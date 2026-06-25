package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// compressToMp3 converts a WAV file to MP3 using ffmpeg.
// Returns the path to the MP3 file, or empty string and error if it fails or if ffmpeg is missing.
func compressToMp3(wavPath string) (string, error) {
	_, err := exec.LookPath("ffmpeg")
	if err != nil {
		return "", fmt.Errorf("ffmpeg not found in PATH: %w", err)
	}

	mp3Path := strings.TrimSuffix(wavPath, ".wav") + ".mp3"
	fmt.Printf("Compressing audio %s to MP3 using FFmpeg...\n", wavPath)

	// -y overwrites existing file
	// -i specifies input file
	// -b:a 32k sets bitrate to 32 kbps (ideal for mono speech, saving space)
	cmd := exec.Command("ffmpeg", "-y", "-i", wavPath, "-b:a", "32k", mp3Path)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("ffmpeg compression failed: %w, stderr: %s", err, stderr.String())
	}

	fmt.Printf("Compression successful: %s\n", mp3Path)
	return mp3Path, nil
}

func getTimestampFromPath(filePath string) string {
	base := filepath.Base(filePath)
	if strings.HasPrefix(base, "pending_") && strings.HasSuffix(base, ".wav") {
		ts := strings.TrimPrefix(base, "pending_")
		ts = strings.TrimSuffix(ts, ".wav")
		return ts
	}
	return time.Now().Format("2006-01-02_15-04-05")
}

func uploadAndTranscribe(ctx context.Context, filePath string, localEvents []string) error {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		return fmt.Errorf("the GEMINI_API_KEY environment variable is not defined")
	}

	timestamp := getTimestampFromPath(filePath)
	queueDir := getQueueDir()
	localMdFile := filepath.Join(queueDir, fmt.Sprintf("transcricao_%s.md", timestamp))
	jsonPath := strings.TrimSuffix(filePath, ".wav") + ".json"

	// Try compressing to MP3 first if ffmpeg is available
	activeAudioPath := filePath
	audioMime := "audio/wav"
	audioExt := ".wav"
	var mp3Path string
	var errMp3 error

	if _, lookErr := exec.LookPath("ffmpeg"); lookErr == nil {
		mp3Path, errMp3 = compressToMp3(filePath)
		if errMp3 == nil {
			activeAudioPath = mp3Path
			audioMime = "audio/mpeg"
			audioExt = ".mp3"
		} else {
			fmt.Printf("Warning: failed to compress to MP3: %v. Falling back to WAV.\n", errMp3)
		}
	} else {
		fmt.Println("FFmpeg not found. Using raw WAV audio.")
	}

	// 1. Generate local markdown transcript if it does not exist already
	if _, err := os.Stat(localMdFile); os.IsNotExist(err) {
		// Initialize the Gemini API client
		client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
		if err != nil {
			return fmt.Errorf("failed to create Gemini client: %w", err)
		}
		defer client.Close()

		// Open the local audio recording file
		file, err := os.Open(activeAudioPath)
		if err != nil {
			return fmt.Errorf("failed to open file %s: %w", activeAudioPath, err)
		}
		defer file.Close()

		fmt.Printf("Uploading audio file %s to Gemini...\n", activeAudioPath)
		// Upload the file using the Gemini File API
		uploadedFile, err := client.UploadFile(ctx, "", file, &genai.UploadFileOptions{
			MIMEType:    audioMime,
			DisplayName: "Coral Recorder Meeting " + timestamp,
		})
		if err != nil {
			return fmt.Errorf("failed to upload file to Gemini File API: %w", err)
		}

		// Ensure the temporary file on Gemini is deleted after processing
		defer func() {
			fmt.Println("Cleaning up temporary file from Gemini...")
			if errDel := client.DeleteFile(ctx, uploadedFile.Name); errDel != nil {
				fmt.Printf("Warning: failed to delete temporary file from Gemini: %v\n", errDel)
			}
		}()

		fmt.Printf("File uploaded successfully. ID: %s. Processing audio...\n", uploadedFile.Name)

		// Wait for Gemini to finish processing the file
		for uploadedFile.State == genai.FileStateProcessing {
			fmt.Println("Waiting for file processing in Gemini...")
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(2 * time.Second):
			}
			uploadedFile, err = client.GetFile(ctx, uploadedFile.Name)
			if err != nil {
				return fmt.Errorf("failed to check file processing status: %w", err)
			}
		}

		if uploadedFile.State != genai.FileStateActive {
			return fmt.Errorf("unexpected file state in Gemini: %v", uploadedFile.State)
		}

		// Build the local acoustic events list to instruct Gemini
		eventsPrompt := ""
		if len(localEvents) > 0 {
			eventsPrompt = "\nThe following acoustic events were identified locally by the Coral Board (Edge TPU) during the recording:\n"
			for _, ev := range localEvents {
				eventsPrompt += fmt.Sprintf("- %s\n", ev)
			}
			eventsPrompt += "\nPlease translate and integrate these relevant moments into the Portuguese transcript using tags like [Risos] (laughter), [Palmas] (clapping), or [Tosse] (cough) at their corresponding timestamps.\n"
		}

		model := client.GenerativeModel("gemini-2.5-flash")

		// Pass the uploaded file's URI inside the prompt
		prompt := []genai.Part{
			genai.FileData{URI: uploadedFile.URI},
			genai.Text("The audio of this meeting is in Brazilian Portuguese. Provide a structured response in Brazilian Portuguese in the following order:\n" +
				"1. A structured executive summary (Sumário Executivo).\n" +
				"2. A clear list of action items/tasks (Itens de Ação).\n" +
				"3. A complete transcription separated by speaker (Transcrição Detalhada), including timestamps (e.g., [MM:SS] or [HH:MM:SS]) for each speaker turn. " +
				"Pay close attention to any names mentioned during the conversation to infer the identity of the speakers; " +
				"if a speaker's name can be determined or reasonably inferred from the conversation (e.g., if someone says 'Olá Alexandre' and the speaker responds), " +
				"use their actual name as the speaker label instead of generic placeholders like 'Speaker 1', 'Speaker 2', etc.\n\n" +
				"IMPORTANT: Completely ignore and omit any trailing voice commands used to stop the recording (such as 'coral, parar gravação', 'coral, terminar gravação', 'coral, encerrar gravação', 'coral, stop', 'stop', 'parar') from the very end of the transcription." + eventsPrompt),
		}

		fmt.Println("Generating summary and action items...")
		resp, err := model.GenerateContent(ctx, prompt...)
		if err != nil {
			return fmt.Errorf("failed to generate content from the model: %w", err)
		}

		// Construct the Markdown file content
		humanTime := timestamp
		if parsedTime, err := time.Parse("2006-01-02_15-04-05", timestamp); err == nil {
			humanTime = parsedTime.Format("02/01/2006 15:04:05")
		}

		mdContent := fmt.Sprintf("# Meeting Transcript\n\n**Date/Time**: %s\n\n", humanTime)

		if len(localEvents) > 0 {
			mdContent += "### Acoustic Events Detected Locally (Coral Edge TPU):\n"
			for _, ev := range localEvents {
				mdContent += fmt.Sprintf("- %s\n", ev)
			}
			mdContent += "\n"
		}

		mdContent += "\n"

		// Print the result to the console and append it to the Markdown file
		fmt.Println("\n==================================================")
		fmt.Println("                 GEMINI RESULT                    ")
		fmt.Println("==================================================")
		for _, cand := range resp.Candidates {
			if cand.Content != nil {
				for _, part := range cand.Content.Parts {
					partText := fmt.Sprint(part)
					fmt.Println(partText)
					mdContent += partText + "\n"
				}
			}
		}
		fmt.Println("==================================================")

		// Save the Markdown file locally
		err = os.WriteFile(localMdFile, []byte(mdContent), 0644)
		if err != nil {
			return fmt.Errorf("failed to save local markdown file %s: %w", localMdFile, err)
		}
		fmt.Printf("Local file saved successfully: %s\n", localMdFile)
	} else {
		fmt.Printf("Local transcript file %s already exists. Skipping Gemini generation.\n", localMdFile)
	}

	// 2. Upload transcript Markdown file to Google Drive
	driveFileName := fmt.Sprintf("%s_Transcript_Meeting.md", timestamp)
	errUpload := uploadToDrive(ctx, localMdFile, driveFileName, "text/markdown")
	if errUpload != nil {
		return fmt.Errorf("failed to upload transcript to Google Drive: %w", errUpload)
	}
	fmt.Printf("Successfully uploaded transcript to Google Drive: %s\n", driveFileName)

	// 3. Upload audio file to Google Drive
	fileInfo, err := os.Stat(activeAudioPath)
	if err == nil && fileInfo.Size() > 35*1024*1024 {
		fmt.Printf("Warning: audio file %s size (%d bytes) exceeds the Google Apps Script upload limit of 35MB. Skipping audio upload to Google Drive.\n", activeAudioPath, fileInfo.Size())
	} else {
		driveAudioName := fmt.Sprintf("%s_Audio_Meeting%s", timestamp, audioExt)
		errAudioUpload := uploadToDrive(ctx, activeAudioPath, driveAudioName, audioMime)
		if errAudioUpload != nil {
			if strings.Contains(errAudioUpload.Error(), "status 413") || strings.Contains(errAudioUpload.Error(), "Request Entity Too Large") {
				fmt.Printf("Warning: audio file %s is too large to upload to Google Drive (HTTP 413). Skipping audio upload to prevent blocking the queue.\n", activeAudioPath)
			} else {
				return fmt.Errorf("failed to upload audio file to Google Drive: %w", errAudioUpload)
			}
		} else {
			fmt.Printf("Successfully uploaded audio to Google Drive: %s\n", driveAudioName)
		}
	}

	// 4. Cleanup all files since everything succeeded!
	if errDel := os.Remove(localMdFile); errDel != nil {
		fmt.Printf("Warning: failed to delete local transcript file %s: %v\n", localMdFile, errDel)
	} else {
		fmt.Printf("Deleted local transcript file %s after successful Google Drive upload.\n", localMdFile)
	}

	if errDel := os.Remove(filePath); errDel != nil {
		fmt.Printf("Warning: failed to delete local audio file %s: %v\n", filePath, errDel)
	} else {
		fmt.Printf("Deleted local audio file %s.\n", filePath)
	}

	if mp3Path != "" {
		if errDel := os.Remove(mp3Path); errDel != nil && !os.IsNotExist(errDel) {
			fmt.Printf("Warning: failed to delete local compressed audio file %s: %v\n", mp3Path, errDel)
		} else {
			fmt.Printf("Deleted local compressed audio file %s.\n", mp3Path)
		}
	}

	if errDel := os.Remove(jsonPath); errDel != nil && !os.IsNotExist(errDel) {
		fmt.Printf("Warning: failed to delete local json metadata file %s: %v\n", jsonPath, errDel)
	} else {
		fmt.Printf("Deleted local json file %s.\n", jsonPath)
	}

	return nil
}
