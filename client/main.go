package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"time"

	"github.com/gordonklaus/portaudio"
)

func main() {
	// Start the local HTTP/HTTPS tunneling proxy in the background
	proxyPort := 8082
	go startProxy(proxyPort)

	// Automatically configure ADB reverse port forwarding to the board
	log.Printf("Configuring ADB reverse port forwarding (tcp:%d -> tcp:%d)...\n", proxyPort, proxyPort)
	cmd := exec.Command("adb", "reverse", fmt.Sprintf("tcp:%d", proxyPort), fmt.Sprintf("tcp:%d", proxyPort))
	if err := cmd.Run(); err != nil {
		log.Printf("Warning: failed to run 'adb reverse': %v (make sure the board is connected via USB and 'adb' is in your PATH)\n", err)
	} else {
		log.Println("ADB reverse port forwarding configured successfully!")
	}

	listFlag := flag.Bool("list", false, "List available audio input devices")
	deviceFlag := flag.Int("device", -1, "Select input device index (default is default system input)")
	serverFlag := flag.String("server", "localhost:5000", "UDP server address (ip:port)")
	usbFlag := flag.Bool("usb", true, "Connect to Coral server via USB-C cable (default: true)")
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
		serverAddr = getUSBIP() + ":5000"
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

	// Use all available input channels (e.g. mic ch1 + BlackHole ch2/ch3 on an Aggregate Device)
	numChannels := selectedDevice.MaxInputChannels
	if numChannels < 1 {
		numChannels = 1
	}
	framesPerBuffer := 1024
	// Interleaved multi-channel buffer: [ch0_f0, ch1_f0, ..., ch0_f1, ch1_f1, ...]
	multiBuffer := make([]int16, framesPerBuffer*numChannels)

	// Configure stream parameters for input-only stream
	params := portaudio.LowLatencyParameters(selectedDevice, nil)
	params.Input.Channels = numChannels
	params.SampleRate = 16000
	params.FramesPerBuffer = framesPerBuffer

	stream, err := portaudio.OpenStream(params, &multiBuffer)
	if err != nil {
		log.Fatalf("Failed to open PortAudio stream: %v", err)
	}
	defer stream.Close()

	err = stream.Start()
	if err != nil {
		log.Fatalf("Failed to start stream: %v", err)
	}
	defer stream.Stop()

	log.Printf("Streaming audio over UDP to %s... (%d input channel(s), downmixed to mono)\n", serverAddr, numChannels)
	for {
		err = stream.Read()
		if err != nil {
			log.Printf("Failed to read audio data: %v", err)
			break
		}
		// Downmix all channels to mono by averaging, then send
		_, err = conn.Write(int16ToBytes(downmixToMono(multiBuffer, numChannels)))
		if err != nil {
			log.Printf("Failed to send data over UDP: %v", err)
			break
		}
	}
}

// getUSBIP scans local network interfaces and returns the matching subnet IP for the Coral board.
// It uses two passes: first checking dedicated USB subnets on physical interfaces, then falling
// back to 192.168.2.x which macOS Internet Sharing (bridge100) uses as its DHCP subnet.
func getUSBIP() string {
	interfaces, err := net.Interfaces()
	if err == nil {
		// First pass: check for dedicated USB Ethernet subnets
		for _, iface := range interfaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}
			for _, addr := range addrs {
				ipnet, ok := addr.(*net.IPNet)
				if ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
					ipStr := ipnet.IP.String()
					// If host is on 192.168.137.x subnet, the board is at 192.168.137.2
					if len(ipStr) >= 12 && ipStr[:12] == "192.168.137." {
						return "192.168.137.2"
					}
					// If host is on 192.168.100.x subnet, the board is at 192.168.100.2
					if len(ipStr) >= 12 && ipStr[:12] == "192.168.100." {
						return "192.168.100.2"
					}
				}
			}
		}
		// Second pass: accept 192.168.2.x on any interface
		for _, iface := range interfaces {
			addrs, err := iface.Addrs()
			if err != nil {
				continue
			}
			for _, addr := range addrs {
				ipnet, ok := addr.(*net.IPNet)
				if ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
					ipStr := ipnet.IP.String()
					if len(ipStr) >= 10 && ipStr[:10] == "192.168.2." {
						return "192.168.2.2"
					}
				}
			}
		}
	}
	// Default fallback to 192.168.100.2 if neither is found
	return "192.168.100.2"
}

// downmixToMono downmixes interleaved channels to a single mono channel.
// To prevent extreme attenuation on virtual multi-channel devices (like Pipewire/BlackHole with 16 or 64 channels),
// we only average the first 2 channels, ignoring extra silent channels.
func downmixToMono(input []int16, numChannels int) []int16 {
	if numChannels <= 1 {
		return input
	}
	useChannels := numChannels
	if useChannels > 2 {
		useChannels = 2
	}
	numFrames := len(input) / numChannels
	mono := make([]int16, numFrames)
	for f := 0; f < numFrames; f++ {
		var sum int32
		for c := 0; c < useChannels; c++ {
			sum += int32(input[f*numChannels+c])
		}
		mono[f] = int16(sum / int32(useChannels))
	}
	return mono
}

// int16ToBytes converts an int16 slice to a little-endian byte slice.
func int16ToBytes(input []int16) []byte {
	output := make([]byte, len(input)*2)
	for i, v := range input {
		binary.LittleEndian.PutUint16(output[i*2:], uint16(v))
	}
	return output
}

// startProxy launches a local HTTP/HTTPS tunneling proxy.
func startProxy(port int) {
	server := &http.Server{
		Addr: fmt.Sprintf(":%d", port),
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodConnect {
				// HTTPS Tunneling (CONNECT method)
				destConn, err := net.DialTimeout("tcp", r.Host, 10*time.Second)
				if err != nil {
					http.Error(w, err.Error(), http.StatusServiceUnavailable)
					return
				}
				w.WriteHeader(http.StatusOK)
				hijacker, ok := w.(http.Hijacker)
				if !ok {
					destConn.Close()
					http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
					return
				}
				clientConn, _, err := hijacker.Hijack()
				if err != nil {
					destConn.Close()
					return
				}
				go transfer(destConn, clientConn)
				go transfer(clientConn, destConn)
			} else {
				// Standard HTTP forwarding
				transport := http.DefaultTransport
				outReq := new(http.Request)
				*outReq = *r
				outReq.RequestURI = ""
				res, err := transport.RoundTrip(outReq)
				if err != nil {
					http.Error(w, err.Error(), http.StatusBadGateway)
					return
				}
				defer res.Body.Close()
				for k, vv := range res.Header {
					for _, v := range vv {
						w.Header().Add(k, v)
					}
				}
				w.WriteHeader(res.StatusCode)
				io.Copy(w, res.Body)
			}
		}),
	}
	log.Printf("HTTP/HTTPS Proxy started on local port %d\n", port)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Printf("Proxy server error: %v\n", err)
	}
}

func transfer(destination io.WriteCloser, source io.ReadCloser) {
	defer destination.Close()
	defer source.Close()
	io.Copy(destination, source)
}
