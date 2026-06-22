//go:build !no_portaudio

package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log"

	"github.com/gordonklaus/portaudio"
)

func initializeAudio() error {
	return portaudio.Initialize()
}

func terminateAudio() {
	portaudio.Terminate()
}

func listAudioDevices() error {
	devices, err := portaudio.Devices()
	if err != nil {
		return err
	}
	fmt.Println("Available Audio Input Devices on Server:")
	for i, dev := range devices {
		if dev.MaxInputChannels > 0 {
			fmt.Printf("[%d] Name: %s (Host API: %s, Max Input Channels: %d)\n", i, dev.Name, dev.HostApi.Name, dev.MaxInputChannels)
		}
	}
	return nil
}

func selectAudioDevice(index int) (interface{}, string, error) {
	devices, err := portaudio.Devices()
	if err != nil {
		return nil, "", err
	}
	var selectedDevice *portaudio.DeviceInfo
	if index >= 0 && index < len(devices) {
		selectedDevice = devices[index]
		if selectedDevice.MaxInputChannels == 0 {
			return nil, "", fmt.Errorf("selected device [%d] is not an input device", index)
		}
	} else {
		selectedDevice, err = portaudio.DefaultInputDevice()
		if err != nil {
			return nil, "", err
		}
	}
	return selectedDevice, selectedDevice.Name, nil
}

func startLocalAudio(device interface{}, dataChan chan []byte, errChan chan error, stopChan chan bool) {
	devInfo, ok := device.(*portaudio.DeviceInfo)
	if !ok || devInfo == nil {
		errChan <- errors.New("invalid portaudio device info")
		return
	}
	
	buffer := make([]int16, 1024)
	
	params := portaudio.LowLatencyParameters(devInfo, nil)
	params.Input.Channels = 1
	params.SampleRate = 16000
	params.FramesPerBuffer = len(buffer)

	stream, err := portaudio.OpenStream(params, &buffer)
	if err != nil {
		errChan <- err
		return
	}
	defer stream.Close()

	err = stream.Start()
	if err != nil {
		errChan <- err
		return
	}
	defer stream.Stop()

	log.Println("Local microphone capture loop started...")
	for {
		select {
		case <-stopChan:
			log.Println("Stopping local microphone capture...")
			return
		default:
			err = stream.Read()
			if err != nil {
				errChan <- err
				return
			}
			dataChan <- localInt16ToBytes(buffer)
		}
	}
}

func localInt16ToBytes(input []int16) []byte {
	output := make([]byte, len(input)*2)
	for i, v := range input {
		binary.LittleEndian.PutUint16(output[i*2:], uint16(v))
	}
	return output
}
