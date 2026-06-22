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
	"time"

	"encoding/json"
)

// CoralMessage defines the format of JSON messages sent by coral_audio.py
type CoralMessage struct {
	Type      string `json:"type"`
	Value     string `json:"value"`
	Timestamp string `json:"timestamp,omitempty"`
}

func main() {
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

	cmd := exec.Command(pythonExe, "coral_audio.py")
	cmd.Dir = "."
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		log.Fatalf("Failed to create stdin pipe for Python processor: %v", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("Failed to create stdout pipe for Python processor: %v", err)
	}

	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start local Python processor: %v. Make sure Python 3 is installed.", err)
	}
	defer func() {
		_ = cmd.Process.Kill()
	}()

	// Event list and recording status
	var events []string
	var recordingStarted bool
	var recordingDone bool
	doneChan := make(chan bool)

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

	// Stream execution
	var localMicChan chan []byte
	var localErrChan chan error
	var localStopChan chan bool

	if localMode {
		localMicChan = make(chan []byte, 100)
		localErrChan = make(chan error, 1)
		localStopChan = make(chan bool)
		go startLocalAudio(selectedDevice, localMicChan, localErrChan, localStopChan)
		log.Println("Monitoring local microphone. Ready to start recording locally via voice commands.")
	} else {
		log.Println("Waiting for client stream to start (explicit network mode)...")
	}

	udpChan := make(chan []byte, 100)
	clientConnectedChan := make(chan string, 1)
	udpErrChan := make(chan error, 1)

	go func() {
		buf := make([]byte, 4096)
		first := true
		for {
			n, remoteAddr, err := conn.ReadFromUDP(buf)
			if err != nil {
				udpErrChan <- err
				return
			}
			if n > 0 {
				if first {
					clientConnectedChan <- remoteAddr.String()
					first = false
				}
				data := make([]byte, n)
				copy(data, buf[:n])
				udpChan <- data
			}
		}
	}()

	var activeSource string = "mic"
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

	for {
		select {
		case <-doneChan:
			log.Println("Python processor finished.")
			if localMode && activeSource == "mic" {
				close(localStopChan)
			}
			goto EndLoop

		case err := <-localErrChan:
			log.Printf("Local microphone recording error: %v", err)
			if activeSource == "mic" {
				close(localStopChan)
				goto EndLoop
			}

		case err := <-udpErrChan:
			log.Printf("UDP server error: %v", err)
			if activeSource == "net" {
				goto EndLoop
			}

		case clientAddr := <-clientConnectedChan:
			log.Printf(">>> Client connection detected from %s (usb or network)", clientAddr)
			if activeSource == "mic" {
				log.Println(">>> Switching from Standalone (mic) to Client-Server (network) mode...")
				activeSource = "net"
				close(localStopChan) // Stop the local mic stream
			}
			networkStarted = true
			networkInactivityTimer = time.NewTimer(10 * time.Second)
			networkTimeoutChan = networkInactivityTimer.C

		case data := <-localMicChan:
			if activeSource == "mic" {
				_, err := stdin.Write(data)
				if err != nil {
					log.Printf("Error sending local audio to Python processor: %v", err)
					close(localStopChan)
					goto EndLoop
				}
			}

		case data := <-udpChan:
			if activeSource == "net" {
				if networkInactivityTimer != nil {
					networkInactivityTimer.Stop()
					networkInactivityTimer = time.NewTimer(10 * time.Second)
					networkTimeoutChan = networkInactivityTimer.C
				}
				_, err := stdin.Write(data)
				if err != nil {
					log.Printf("Error sending network audio to Python processor: %v", err)
					goto EndLoop
				}
			}

		case <-networkTimeoutChan:
			if activeSource == "net" && networkStarted {
				log.Println("End of audio transmission due to network inactivity (timeout).")
				goto EndLoop
			}

		case <-startupTimeoutChan:
			if activeSource == "net" && !networkStarted {
				log.Println("Timeout waiting for client stream to start in explicit network mode.")
				goto EndLoop
			}
		}
	}

EndLoop:
	// Close stdin to signal EOF to Python (tells script to close WAV and exit if recording)
	stdin.Close()

	// Wait for the Python event reader to finish
	select {
	case <-doneChan:
		log.Println("Coral Python processor finished.")
	case <-time.After(5 * time.Second):
		log.Println("Warning: Timeout waiting for Python processor messages to finish.")
	}

	// Wait for the Python subprocess to exit completely
	_ = cmd.Wait()

	// If recording was started, run transcription
	if recordingStarted {
		if !recordingDone && activeSource == "net" {
			log.Println("Warning: Recording interrupted by network timeout before 'STOP' voice command.")
		}

		log.Println("Starting upload and transcription of the meeting...")
		ctx := context.Background()
		err = uploadAndTranscribe(ctx, "meeting.wav", events)
		if err != nil {
			log.Fatalf("Error during transcription processing: %v", err)
		}
	} else {
		log.Println("No audio was recorded. Skipping transcription.")
	}
}
