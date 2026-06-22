package main

import (
	"os"
	"time"
)

// LED sysfs paths on the Synaptics Coral Dev Board (Astra SL2610).
const (
	ledGreen = "/sys/class/leds/green:status"
	ledRed   = "/sys/class/leds/red:status"
	ledBlue  = "/sys/class/leds/blue:status"
)

// LED state meanings:
//   Green  solid   → server is running, waiting for "GO" command
//   Red    solid   → recording in progress
//   Blue   solid   → uploading / processing (Gemini + Drive)
//   All    off     → server not running / between states
//   Red    blink   → error during transcription

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

// ledReady signals that the server is idle and waiting for the "GO" command.
//   Green solid ON, Red OFF, Blue OFF
func ledReady() {
	ledAllOff()
	ledSet(ledGreen, true)
}

// ledRecording signals that a recording session is active.
//   Red solid ON, Green OFF, Blue OFF
func ledRecording() {
	ledAllOff()
	ledSet(ledRed, true)
}

// ledProcessing signals that the file is being uploaded to Gemini / Google Drive.
//   Blue solid ON, Red OFF, Green OFF
func ledProcessing() {
	ledAllOff()
	ledSet(ledBlue, true)
}

// ledError blinks the red LED continuously to signal a processing or upload error.
// It loops infinitely, keeping the error state active until the server is restarted.
func ledError() {
	ledAllOff()
	for {
		ledSet(ledRed, true)
		time.Sleep(500 * time.Millisecond)
		ledSet(ledRed, false)
		time.Sleep(500 * time.Millisecond)
	}
}
