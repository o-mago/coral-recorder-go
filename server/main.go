package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"encoding/json"
)

// loadEnv parses a local .env file and sets environment variables
func loadEnv() {
	file, err := os.Open(".env")
	if err != nil {
		// Ignore if .env is missing and fallback to system environment
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		// Strip wrapping quotes if any
		if len(val) >= 2 && ((val[0] == '"' && val[len(val)-1] == '"') || (val[0] == '\'' && val[len(val)-1] == '\'')) {
			val = val[1 : len(val)-1]
		}
		if key != "" {
			os.Setenv(key, val)
		}
	}
}

// CoralMessage defines the format of JSON messages sent by coral_audio.py
type CoralMessage struct {
	Type      string `json:"type"`
	Value     string `json:"value"`
	Timestamp string `json:"timestamp,omitempty"`
}

func main() {
	loadEnv()

	// Ensure LEDs are turned off on any exit path (normal return or log.Fatalf).
	defer ledAllOff()

	// Catch SIGINT (Ctrl+C) and SIGTERM (systemd stop) to turn off LEDs before exiting.
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigs
		log.Println("Shutdown signal received. Turning off LEDs and exiting...")
		ledAllOff()
		os.Exit(0)
	}()

	networkFlag := flag.Bool("network", false, "Run in network mode, listening for UDP client streams")
	listFlag := flag.Bool("list", false, "List available audio input devices on the server")
	deviceFlag := flag.Int("device", -1, "Select input device index on the server (default is default system input)")
	statusFlag := flag.Bool("status", false, "Show server service status, queue info, and recent error logs")
	flag.Parse()

	if *statusFlag {
		showStatusAndLogs()
		os.Exit(0)
	}

	// Start background upload queue retry worker and internet checker
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go startRetryLoop(ctx)
	go startInternetCheck(ctx)

	localMode := !*networkFlag

	// If we are listing devices or running in local mode, initialize PortAudio
	if *listFlag || localMode {
		err := initializeAudio()
		if err != nil {
			log.Fatalf("Failed to initialize PortAudio on server: %v", err)
		}
		defer terminateAudio()
	}

	if *listFlag {
		err := listAudioDevices()
		if err != nil {
			log.Fatalf("Failed to list audio devices: %v", err)
		}
		os.Exit(0)
	}

	var selectedDevice interface{}
	if localMode {
		var deviceName string
		var err error
		selectedDevice, deviceName, err = selectAudioDevice(*deviceFlag)
		if err != nil {
			log.Fatalf("Failed to select audio device: %v", err)
		}
		log.Printf("Running in Standalone Mode. Using Input Device: %s\n", deviceName)
	}

	addr, err := net.ResolveUDPAddr("udp", ":5000")
	if err != nil {
		log.Fatalf("Failed to resolve UDP address: %v", err)
	}

	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		log.Fatalf("Failed to start UDP server: %v", err)
	}
	defer conn.Close()
	log.Println("UDP Server started on port 5000. Listening for client connections...")

	// Determine the python executable to use (check virtual environments first)
	pythonExe := "python3"
	if _, err := os.Stat("env/bin/python3"); err == nil {
		pythonExe = "env/bin/python3"
	} else if _, err := os.Stat(".venv/bin/python3"); err == nil {
		pythonExe = ".venv/bin/python3"
	} else if _, err := os.Stat("../env/bin/python3"); err == nil {
		pythonExe = "../env/bin/python3"
	} else if _, err := os.Stat("../.venv/bin/python3"); err == nil {
		pythonExe = "../.venv/bin/python3"
	}
	log.Printf("Starting local Python audio processor using: %s\n", pythonExe)

	// Signal that the server is up and waiting for the first recording.
	if localMode {
		SetAudioSource(SourceLocalMic)
	} else {
		SetAudioSource(SourceExternalNet)
	}
	ledReady()

	udpChan := make(chan []byte, 100)
	udpErrChan := make(chan error, 1)

	// Persistent UDP receiver goroutine — runs for the lifetime of the server.
	go func() {
		buf := make([]byte, 4096)
		for {
			n, _, err := conn.ReadFromUDP(buf)
			if err != nil {
				udpErrChan <- err
				return
			}
			if n > 0 {
				data := make([]byte, n)
				copy(data, buf[:n])
				udpChan <- data
			}
		}
	}()

	// Main session loop: each iteration is one full recording session.
	// After transcription completes the server restarts and waits for the next session.
	for sessionNum := 1; ; sessionNum++ {
		if sessionNum > 1 {
			log.Printf("\n=== [Session %d] Ready for new recording. Waiting for 'GO' voice command... ===\n", sessionNum)
			if localMode {
				SetAudioSource(SourceLocalMic)
			} else {
				SetAudioSource(SourceExternalNet)
			}
			ledReady()
		}

		// Per-session state
		var events []string
		var recordingStarted bool
		var recordingDone bool
		doneChan := make(chan bool, 1)

		// Start a fresh Python audio processor for each session
		cmd := exec.Command(pythonExe, "coral_audio.py")
		cmd.Dir = "."
		cmd.Stderr = os.Stderr

		stdin, err := cmd.StdinPipe()
		if err != nil {
			log.Printf("Failed to create stdin pipe for Python processor: %v", err)
			return
		}

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			log.Printf("Failed to create stdout pipe for Python processor: %v", err)
			return
		}

		if err := cmd.Start(); err != nil {
			log.Printf("Failed to start local Python processor: %v. Make sure Python 3 is installed.", err)
			return
		}

		// Goroutine to read stdout from Python in real-time
		go func() {
			scanner := bufio.NewScanner(stdout)
			for scanner.Scan() {
				line := scanner.Text()
				var msg CoralMessage
				if err := json.Unmarshal([]byte(line), &msg); err != nil {
					continue
				}
				switch msg.Type {
				case "control":
					if msg.Value == "start" {
						log.Println(">>> [Coral Edge TPU] 'GO' signal detected! Starting recording...")
						recordingStarted = true
						ledRecording()
					} else if msg.Value == "done" {
						log.Println(">>> [Coral Edge TPU] 'STOP' signal detected! Stopping recording...")
						recordingDone = true
						doneChan <- true
						return
					}
				case "event":
					eventStr := fmt.Sprintf("[%s] Acoustic event: %s", msg.Timestamp, msg.Value)
					log.Printf(">>> [Coral Edge TPU Event] %s", eventStr)
					events = append(events, eventStr)
				}
			}
			doneChan <- true
		}()

		// Per-session local microphone setup
		var localMicChan chan []byte
		var localErrChan chan error
		var localStopChan chan bool
		localMicStopped := false

		if localMode {
			localMicChan = make(chan []byte, 100)
			localErrChan = make(chan error, 1)
			localStopChan = make(chan bool)
			go startLocalAudio(selectedDevice, localMicChan, localErrChan, localStopChan)
			log.Println("Monitoring local microphone. Ready to start recording locally via voice commands.")
		} else {
			log.Println("Waiting for client stream to start (explicit network mode)...")
		}

		// stopLocalMic safely closes localStopChan at most once per session.
		stopLocalMic := func() {
			if localMode && !localMicStopped {
				close(localStopChan)
				localMicStopped = true
			}
		}

		activeSource := "mic"
		if !localMode {
			activeSource = "net"
		}

		var networkInactivityTimer *time.Timer
		var networkTimeoutChan <-chan time.Time
		var startupTimeoutChan <-chan time.Time
		if !localMode {
			startupTimeoutChan = time.After(5 * time.Minute)
		}
		var networkStarted bool

	SessionLoop:
		for {
			select {
			case <-doneChan:
				log.Println("Python processor signaled done.")
				stopLocalMic()
				break SessionLoop

			case err := <-localErrChan:
				log.Printf("Local microphone recording error: %v", err)
				localMicStopped = true // goroutine already exited
				if activeSource == "mic" {
					break SessionLoop
				}

			case err := <-udpErrChan:
				// UDP socket errors are unrecoverable — the listener goroutine has exited.
				log.Printf("UDP server error (unrecoverable): %v", err)
				stopLocalMic()
				stdin.Close()
				_ = cmd.Process.Kill()
				_ = cmd.Wait()
				return

			case data := <-udpChan:
				if activeSource == "mic" {
					// First UDP data received: switch from local mic to network mode.
					log.Println(">>> Client connection detected (usb or network). Switching from Standalone (mic) to Client-Server (network) mode...")
					stopLocalMic()
					activeSource = "net"
					SetAudioSource(SourceExternalNet)
				}
				if activeSource == "net" {
					if !networkStarted {
						log.Println(">>> Client stream started.")
						networkStarted = true
						networkInactivityTimer = time.NewTimer(10 * time.Second)
						networkTimeoutChan = networkInactivityTimer.C
					} else if networkInactivityTimer != nil {
						networkInactivityTimer.Stop()
						networkInactivityTimer = time.NewTimer(10 * time.Second)
						networkTimeoutChan = networkInactivityTimer.C
					}
					if _, err := stdin.Write(data); err != nil {
						log.Printf("Error sending network audio to Python processor: %v", err)
						break SessionLoop
					}
				}

			case data := <-localMicChan:
				if activeSource == "mic" {
					if _, err := stdin.Write(data); err != nil {
						log.Printf("Error sending local audio to Python processor: %v", err)
						stopLocalMic()
						break SessionLoop
					}
				}

			case <-networkTimeoutChan:
				if activeSource == "net" && networkStarted {
					log.Println("End of audio transmission due to network inactivity (timeout).")
					break SessionLoop
				}

			case <-startupTimeoutChan:
				if activeSource == "net" && !networkStarted {
					log.Println("Timeout waiting for client stream to start in explicit network mode.")
					break SessionLoop
				}
			}
		}

		// Signal EOF to Python and wait for it to flush and exit
		stdin.Close()

		select {
		case <-doneChan:
			log.Println("Coral Python processor finished.")
		case <-time.After(5 * time.Second):
			log.Println("Warning: Timeout waiting for Python processor messages to finish.")
		}

		_ = cmd.Process.Kill()
		_ = cmd.Wait()

		// Transcribe if recording was started during this session
		if recordingStarted {
			if !recordingDone && activeSource == "net" {
				log.Println("Warning: Recording interrupted by network timeout before 'STOP' voice command.")
			}

			timestamp := time.Now().Format("2006-01-02_15-04-05")
			queueDir := getQueueDir()
			pendingWav := filepath.Join(queueDir, fmt.Sprintf("pending_%s.wav", timestamp))
			pendingJson := filepath.Join(queueDir, fmt.Sprintf("pending_%s.json", timestamp))

			log.Printf("Saving session to queue files: %s and %s...\n", pendingWav, pendingJson)

			// Rename meeting.wav to pending_<timestamp>.wav
			meetingPath := filepath.Join(queueDir, "meeting.wav")
			if err := os.Rename(meetingPath, pendingWav); err != nil {
				log.Printf("Failed to rename meeting.wav to %s: %v", pendingWav, err)
				// If rename fails, try direct local rename as fallback
				if err2 := os.Rename("meeting.wav", pendingWav); err2 != nil {
					pendingWav = "meeting.wav"
				}
			}

			// Save metadata events to pending_<timestamp>.json
			metadata := PendingSession{
				Timestamp: timestamp,
				Events:    events,
			}
			metadataBytes, err := json.Marshal(metadata)
			if err != nil {
				log.Printf("Failed to marshal pending metadata: %v", err)
			} else {
				if err := os.WriteFile(pendingJson, metadataBytes, 0644); err != nil {
					log.Printf("Failed to write pending json file %s: %v", pendingJson, err)
				}
			}

			// Manually refresh LED state
			TriggerLEDUpdate()

			// Trigger background retry loop to process this new file immediately
			triggerUploadRetry()
		} else {
			log.Println("No audio was recorded. Skipping transcription.")
		}
	}
}

// showStatusAndLogs fetches and formats service logs, systemd status, and the local pending queue.
func showStatusAndLogs() {
	fmt.Println("==================================================")
	fmt.Println("         CORAL RECORDER SERVICE STATUS            ")
	fmt.Println("==================================================")

	// 1. Check systemd service status
	cmd := exec.Command("systemctl", "status", "coral-recorder")
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Systemd service status check error/inactive: %v\n", err)
	}
	fmt.Println(string(output))

	// 2. Check pending queue files
	fmt.Println("==================================================")
	fmt.Println("            PENDING RECORDINGS QUEUE              ")
	fmt.Println("==================================================")
	queueDir := getQueueDir()
	files, err := os.ReadDir(queueDir)
	if err != nil {
		fmt.Printf("Failed to read directory %s: %v\n", queueDir, err)
	} else {
		var pendingWavs []os.DirEntry
		for _, f := range files {
			if !f.IsDir() && strings.HasPrefix(f.Name(), "pending_") && strings.HasSuffix(f.Name(), ".wav") {
				pendingWavs = append(pendingWavs, f)
			}
		}

		if len(pendingWavs) == 0 {
			fmt.Println("Queue is empty. No pending uploads.")
		} else {
			fmt.Printf("Found %d pending recording(s) in queue:\n\n", len(pendingWavs))
			for _, f := range pendingWavs {
				info, err := f.Info()
				if err != nil {
					fmt.Printf("- %s (details unavailable)\n", f.Name())
					continue
				}
				sizeMB := float64(info.Size()) / (1024 * 1024)
				
				// Read matching JSON metadata if present
				jsonPath := filepath.Join(queueDir, strings.TrimSuffix(f.Name(), ".wav")+".json")
				eventCount := 0
				if jsonBytes, err := os.ReadFile(jsonPath); err == nil {
					var metadata PendingSession
					if err := json.Unmarshal(jsonBytes, &metadata); err == nil {
						eventCount = len(metadata.Events)
					}
				}
				
				// Read matching local transcript markdown if present
				mdPath := filepath.Join(queueDir, fmt.Sprintf("transcricao_%s.md", getTimestampFromPath(f.Name())))
				hasMd := "No"
				if _, err := os.Stat(mdPath); err == nil {
					hasMd = "Yes"
				}

				fmt.Printf("- File: %s\n  Size: %.2f MB\n  Local Events: %d\n  Transcript Cached: %s\n\n", 
					f.Name(), sizeMB, eventCount, hasMd)
			}
		}
	}

	// 3. Show recent journal logs
	fmt.Println("==================================================")
	fmt.Println("              RECENT SERVICE LOGS                 ")
	fmt.Println("==================================================")
	cmdLogs := exec.Command("journalctl", "-u", "coral-recorder", "-n", "30", "--no-pager")
	logsOutput, err := cmdLogs.CombinedOutput()
	if err != nil {
		fmt.Printf("Failed to fetch journal logs: %v\n", err)
		if len(logsOutput) > 0 {
			fmt.Println(string(logsOutput))
		}
		fmt.Println("\n* Hint: To apply the 'systemd-journal' group in your current SSH/terminal session, run: newgrp systemd-journal")
	} else {
		fmt.Println(string(logsOutput))
	}
	fmt.Println("==================================================")
}

// getQueueDir resolves the directory where the binary is running, where pending queue files reside.
func getQueueDir() string {
	exePath, err := os.Executable()
	if err == nil {
		return filepath.Dir(exePath)
	}
	return "."
}
