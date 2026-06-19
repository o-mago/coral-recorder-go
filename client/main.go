package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"net"
	"os"

	"github.com/gordonklaus/portaudio"
)

func main() {
	listFlag := flag.Bool("list", false, "List available audio input devices")
	deviceFlag := flag.Int("device", -1, "Select input device index (default is default system input)")
	serverFlag := flag.String("server", "localhost:5000", "UDP server address (ip:port)")
	usbFlag := flag.Bool("usb", false, "Connect to Coral server via USB-C cable (uses virtual Ethernet IP 192.168.100.2:5000)")
	flag.Parse()

	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Failed to initialize PortAudio: %v", err)
	}
	defer portaudio.Terminate()

	devices, err := portaudio.Devices()
	if err != nil {
		log.Fatalf("Failed to get audio devices: %v", err)
	}

	if *listFlag {
		fmt.Println("Available Audio Input Devices:")
		for i, dev := range devices {
			if dev.MaxInputChannels > 0 {
				fmt.Printf("[%d] Name: %s (Host API: %s, Max Input Channels: %d)\n", i, dev.Name, dev.HostApi.Name, dev.MaxInputChannels)
			}
		}
		os.Exit(0)
	}

	var selectedDevice *portaudio.DeviceInfo
	if *deviceFlag >= 0 && *deviceFlag < len(devices) {
		selectedDevice = devices[*deviceFlag]
		if selectedDevice.MaxInputChannels == 0 {
			log.Fatalf("Selected device [%d] is not an input device", *deviceFlag)
		}
	} else {
		selectedDevice, err = portaudio.DefaultInputDevice()
		if err != nil {
			log.Fatalf("Failed to get default input device: %v", err)
		}
	}

	log.Printf("Using Input Device: %s (Max Input Channels: %d)\n", selectedDevice.Name, selectedDevice.MaxInputChannels)

	serverAddr := *serverFlag
	if *usbFlag {
		serverAddr = "192.168.100.2:5000"
		log.Printf("Connecting to Coral via USB-C (address: %s)\n", serverAddr)
	} else {
		log.Printf("Connecting to Coral via network (address: %s)\n", serverAddr)
	}

	// Connect to the UDP server
	conn, err := net.Dial("udp", serverAddr)
	if err != nil {
		log.Fatalf("Failed to connect via UDP to %s: %v", serverAddr, err)
	}
	defer conn.Close()

	buffer := make([]int16, 1024)

	// Configure stream parameters for input-only stream
	params := portaudio.LowLatencyParameters(selectedDevice, nil)
	params.Input.Channels = 1
	params.SampleRate = 16000
	params.FramesPerBuffer = len(buffer)

	stream, err := portaudio.OpenStream(params, &buffer)
	if err != nil {
		log.Fatalf("Failed to open PortAudio stream: %v", err)
	}
	defer stream.Close()

	err = stream.Start()
	if err != nil {
		log.Fatalf("Failed to start stream: %v", err)
	}
	defer stream.Stop()

	log.Printf("Streaming audio over UDP to %s...\n", serverAddr)
	for {
		err = stream.Read()
		if err != nil {
			log.Printf("Failed to read audio data: %v", err)
			break
		}
		// Convert int16 to bytes and send over the network
		_, err = conn.Write(int16ToBytes(buffer))
		if err != nil {
			log.Printf("Failed to send data over UDP: %v", err)
			break
		}
	}
}

// int16ToBytes converts an int16 slice to a little-endian byte slice.
func int16ToBytes(input []int16) []byte {
	output := make([]byte, len(input)*2)
	for i, v := range input {
		binary.LittleEndian.PutUint16(output[i*2:], uint16(v))
	}
	return output
}
