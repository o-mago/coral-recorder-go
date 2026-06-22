package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

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

	// 1. Generate local markdown transcript if it does not exist already
	if _, err := os.Stat(localMdFile); os.IsNotExist(err) {
		// Initialize the Gemini API client
		client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
		if err != nil {
			return fmt.Errorf("failed to create Gemini client: %w", err)
		}
		defer client.Close()

		// Open the local audio recording file
		file, err := os.Open(filePath)
		if err != nil {
			return fmt.Errorf("failed to open file %s: %w", filePath, err)
		}
		defer file.Close()

		fmt.Printf("Uploading audio file %s to Gemini...\n", filePath)
		// Upload the file using the Gemini File API
		uploadedFile, err := client.UploadFile(ctx, "", file, &genai.UploadFileOptions{
			MIMEType:    "audio/wav",
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
			genai.Text("The audio of this meeting is in Brazilian Portuguese. Provide a structured response in Brazilian Portuguese in the following order:\n1. A structured executive summary (Sumário Executivo).\n2. A clear list of action items/tasks (Itens de Ação).\n3. A complete transcription separated by speaker (Transcrição Detalhada, identifying them by their voices).\n\nIMPORTANT: Completely ignore and omit any trailing voice commands used to stop the recording (such as 'coral, parar gravação', 'coral, terminar gravação', 'coral, encerrar gravação', 'coral, stop', 'stop', 'parar') from the very end of the transcription." + eventsPrompt),
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
	driveAudioName := fmt.Sprintf("%s_Audio_Meeting.wav", timestamp)
	errAudioUpload := uploadToDrive(ctx, filePath, driveAudioName, "audio/wav")
	if errAudioUpload != nil {
		return fmt.Errorf("failed to upload audio file to Google Drive: %w", errAudioUpload)
	}
	fmt.Printf("Successfully uploaded audio to Google Drive: %s\n", driveAudioName)

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

	if errDel := os.Remove(jsonPath); errDel != nil && !os.IsNotExist(errDel) {
		fmt.Printf("Warning: failed to delete local json metadata file %s: %v\n", jsonPath, errDel)
	} else {
		fmt.Printf("Deleted local json file %s.\n", jsonPath)
	}

	return nil
}
