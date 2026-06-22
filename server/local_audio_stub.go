//go:build no_portaudio

package main

import (
	"errors"
	"log"
)

func initializeAudio() error {
	log.Println("PortAudio support is disabled in this build.")
	return errors.New("portaudio support disabled")
}

func terminateAudio() {
}

func listAudioDevices() error {
	log.Println("PortAudio support is disabled in this build. Cannot list devices.")
	return errors.New("portaudio support disabled")
}

func selectAudioDevice(index int) (interface{}, string, error) {
	return nil, "", errors.New("portaudio support disabled")
}

func startLocalAudio(device interface{}, dataChan chan []byte, errChan chan error, stopChan chan bool) {
	log.Println("PortAudio support is disabled in this build. Cannot start local audio.")
	errChan <- errors.New("portaudio support disabled")
}
