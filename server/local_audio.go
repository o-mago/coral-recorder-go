package main

import (
	"encoding/binary"
	"log"

	"github.com/gordonklaus/portaudio"
)

// recordLocalAudio records audio from the selected input device and sends it to dataChan
func recordLocalAudio(device *portaudio.DeviceInfo, dataChan chan []byte, errChan chan error, stopChan chan bool) {
	buffer := make([]int16, 1024)
	
	params := portaudio.LowLatencyParameters(device, nil)
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

// localInt16ToBytes converts an int16 slice to a little-endian byte slice.
func localInt16ToBytes(input []int16) []byte {
	output := make([]byte, len(input)*2)
	for i, v := range input {
		binary.LittleEndian.PutUint16(output[i*2:], uint16(v))
	}
	return output
}
