package main

import (
	"os"
	"strings"
	"sync"
	"time"
)

// LED sysfs paths on the Synaptics Coral Dev Board (Astra SL2610).
const (
	ledGreen = "/sys/class/leds/green:status"
	ledRed   = "/sys/class/leds/red:status"
	ledBlue  = "/sys/class/leds/blue:status"
)

type ServerState int

const (
	StateReady ServerState = iota
	StateRecording
	StateProcessing
)

type AudioSource int

const (
	SourceLocalMic AudioSource = iota
	SourceExternalNet
)

var (
	currentServerState     = StateReady
	currentAudioSource     = SourceLocalMic
	blinkingChan           chan struct{}
	blinkingDoneChan       chan struct{}
	blinkingMutex          sync.Mutex
	uploadingInProgress    bool
	reprocessingInProgress bool
	uploadingMutex         sync.Mutex
)

// SetAudioSource updates the active audio source (local mic vs external network) and refreshes LEDs.
func SetAudioSource(source AudioSource) {
	uploadingMutex.Lock()
	currentAudioSource = source
	uploadingMutex.Unlock()
	TriggerLEDUpdate()
}

// SetUploadingProgress updates the background uploading state and triggers an LED refresh.
func SetUploadingProgress(active bool) {
	uploadingMutex.Lock()
	uploadingInProgress = active
	uploadingMutex.Unlock()
	TriggerLEDUpdate()
}

// SetReprocessingProgress updates the background reprocessing state and triggers an LED refresh.
func SetReprocessingProgress(active bool) {
	uploadingMutex.Lock()
	reprocessingInProgress = active
	uploadingMutex.Unlock()
	TriggerLEDUpdate()
}

// ledSet sets a single LED on (brightness=1) or off (brightness=0)
// and disables any kernel trigger so we have full manual control.
func ledSet(path string, on bool) {
	// Disable kernel trigger first
	_ = os.WriteFile(path+"/trigger", []byte("none"), 0644)
	val := "0"
	if on {
		val = "1"
	}
	_ = os.WriteFile(path+"/brightness", []byte(val), 0644)
}

// ledAllOff turns off all three status LEDs.
func ledAllOff() {
	ledSet(ledGreen, false)
	ledSet(ledRed, false)
	ledSet(ledBlue, false)
}

// ledErrorBlinkStart starts a background goroutine to blink the red LED.
func ledErrorBlinkStart() {
	blinkingMutex.Lock()
	defer blinkingMutex.Unlock()
	if blinkingChan != nil {
		return // Already blinking
	}
	blinkingChan = make(chan struct{})
	blinkingDoneChan = make(chan struct{})
	go func(stopChan, doneChan chan struct{}) {
		defer close(doneChan)
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		state := false
		for {
			select {
			case <-stopChan:
				ledSet(ledRed, false)
				return
			case <-ticker.C:
				state = !state
				ledSet(ledRed, state)
			}
		}
	}(blinkingChan, blinkingDoneChan)
}

// ledErrorBlinkStop stops the background blinking goroutine synchronously.
func ledErrorBlinkStop() {
	blinkingMutex.Lock()
	defer blinkingMutex.Unlock()
	if blinkingChan != nil {
		close(blinkingChan)
		<-blinkingDoneChan
		blinkingChan = nil
		blinkingDoneChan = nil
	}
}

// hasPendingFiles checks if there are any pending upload files on disk.
func hasPendingFiles() bool {
	files, err := os.ReadDir(getQueueDir())
	if err != nil {
		return false
	}
	for _, f := range files {
		if !f.IsDir() && strings.HasPrefix(f.Name(), "pending_") && strings.HasSuffix(f.Name(), ".wav") {
			return true
		}
	}
	return false
}

// updateLED refreshes the physical LED states based on current server state and pending uploads.
func updateLED() {
	hasPending := hasPendingFiles()

	if currentServerState == StateRecording {
		// 1. Recording: solid red only
		ledErrorBlinkStop()
		ledAllOff()
		ledSet(ledRed, true)
	} else if reprocessingInProgress {
		// 2. Reprocessing queue files with error: solid blue + blinking red
		ledAllOff()
		ledSet(ledBlue, true)
		ledErrorBlinkStart()
	} else if uploadingInProgress {
		// 3. Normal processing: solid blue only
		ledErrorBlinkStop()
		ledAllOff()
		ledSet(ledBlue, true)
	} else {
		// 4. Idle / Ready: green (or green+blue) solid, and red blinking if there are pending files
		ledAllOff()
		if currentAudioSource == SourceExternalNet {
			ledSet(ledGreen, true)
			ledSet(ledBlue, true)
		} else {
			ledSet(ledGreen, true)
		}

		if hasPending {
			ledErrorBlinkStart()
		} else {
			ledErrorBlinkStop()
		}
	}
}

// SetServerState transitions the server state and updates LEDs.
func SetServerState(state ServerState) {
	currentServerState = state
	updateLED()
}

// TriggerLEDUpdate manually refreshes the LEDs (used when files are added or removed from queue).
func TriggerLEDUpdate() {
	updateLED()
}

// ledReady updates state to ready.
func ledReady() {
	SetServerState(StateReady)
}

// ledRecording updates state to recording.
func ledRecording() {
	SetServerState(StateRecording)
}

// ledProcessing updates state to processing.
func ledProcessing() {
	SetServerState(StateProcessing)
}

// ledError is called when an error occurs.
func ledError() {
	// Re-evaluation will automatically start blinking if queue has pending files.
	updateLED()
}
