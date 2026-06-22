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

	// Run Google Drive upload test
	fmt.Println("RUNNING GOOGLE DRIVE UPLOAD TEST...")
	dummyFile := "dummy_test.txt"
	err := os.WriteFile(dummyFile, []byte("Hello from Coral board via USB ADB Proxy!"), 0644)
	if err != nil {
		log.Fatalf("Failed to write dummy file: %v", err)
	}
	err = uploadToDrive(context.Background(), dummyFile, "test_coral_usb_proxy.txt", "text/plain")
	os.Remove(dummyFile)
	if err != nil {
		log.Fatalf("GOOGLE DRIVE UPLOAD TEST FAILED: %v", err)
	}
	fmt.Println("GOOGLE DRIVE UPLOAD TEST SUCCEEDED!")
	os.Exit(0)

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
	flag.Parse()

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
			log.Println("Starting upload and transcription of the meeting...")
			ledProcessing()
			ctx := context.Background()
			if err := uploadAndTranscribe(ctx, "meeting.wav", events); err != nil {
				log.Printf("Error during transcription processing: %v", err)
				ledError() // blinks red 5x, then returns to green (ready)
			}
		} else {
			log.Println("No audio was recorded. Skipping transcription.")
		}
	}
}
