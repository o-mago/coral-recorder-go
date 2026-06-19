# Coral Recorder Go

This project is a Go-based client-server system designed to capture audio from a microphone, stream it in real-time over UDP, process it locally using the **Google Coral Edge TPU**, and leverage the Gemini 2.5 Flash API to transcribe, identify speakers, and extract action items from the meeting.

## Architecture

1. **Client**: Captures local microphone audio using the `portaudio` library, converts `int16` samples to bytes (`Little Endian`), and streams them over the network using the UDP protocol.
2. **Server**: Listens on a UDP port (default `:5000`) and forwards the raw audio stream to a local Python subprocess (`coral_audio.py`), which runs Edge TPU-accelerated models:
   - **Voice Commands (Keyword Spotting)**: Continuously listens in the background. It starts recording upon hearing **"go"** (or "yes"/"on") and stops upon hearing **"stop"** (or "no"/"off").
   - **Intelligent VAD (Voice Activity Detection)**: Saves only segments containing active human speech to `meeting.wav` (using a hangover buffer to prevent clipping), removing silent periods.
   - **Audio Event Classification (YAMNet)**: Detects and logs background audio events such as laughter, clapping, and coughing in real-time, sending timestamps to Go.
3. **Gemini & Cloud Orchestration**: Once recording is finalized, Go uploads the processed `meeting.wav` file to the **Gemini File API**, appends the list of local events detected by the TPU to the prompt, and requests a speaker-separated transcription integrated with the acoustic events.

---

## Prerequisites

### Client (Recording)
You need the `PortAudio` audio library and the `pkg-config` tool installed on your operating system.

#### macOS
```bash
brew install portaudio pkg-config
```

#### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev pkg-config
```

### Server (Coral Edge TPU / CPU)
The server runs a hybrid pipeline with Python 3. Install the required dependencies:

```bash
# General Python dependencies
pip3 install numpy

# TensorFlow Lite Runtime
# On Coral Dev Board (Linux):
sudo apt-get install python3-tflite-runtime python3-pycoral
# Or on standard computers (macOS/Linux/Windows) to test on CPU:
pip3 install tflite-runtime
```
*Note: The Python script automatically falls back to CPU execution if the Edge TPU (libedgetpu.so.1) is not detected.*

---

## Configuration

### 1. Download Local TFLite Models
Navigate to the server folder and run the download script to fetch the pre-trained YAMNet and Voice Commands models:
```bash
cd server
./download_models.sh
```

### 2. Configure Server IP Address on the Client
You do not need to edit the client code. By default, the client sends audio to `localhost:5000`. You can specify the server address dynamically using the `-server` command-line flag depending on your connection method:

- **Via Local Network (Wi-Fi or Ethernet)**:
  Find your Coral Board's network IP address (e.g. `192.168.1.100`) and run:
  ```bash
  go run . -server 192.168.1.100:5000
  ```
- **Via USB-C Cable (Direct Connection)**:
  Connect your PC directly to the Coral Board's USB-C data port (labeled OTG). Mendel Linux automatically sets up a virtual network interface (USB Ethernet Gadget) assigning the static IP **`192.168.100.2`** to the Coral Board.
  1. Start the server on the Coral Board in network mode:
     ```bash
     go run . -network
     ```
  2. Start the client on the PC pointing to the virtual USB IP:
     - Using the convenient `-usb` flag (recommended):
       ```bash
       go run . -usb
       ```
     - Or by manually specifying the IP and port with the `-server` flag:
       ```bash
       go run . -server 192.168.100.2:5000
       ```

### 3. Gemini API Key
Define the `GEMINI_API_KEY` environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 4. Google Drive Integration (Optional)
Define the variables to automatically save the generated meeting notes to your Google Drive account:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-credentials.json"
export GOOGLE_DRIVE_FOLDER_ID="1A2B3C4D5E...your-folder-id"
```

---

## How to Run

### 1. Start the Server on the Coral Board

#### Auto-Switching Hybrid Mode (Default - Local Mic with Client Auto-Detect)
By default, the server starts in a hybrid mode. It captures audio from the local microphone (using a connected USB/built-in mic) but concurrently listens for client UDP connections on port 5000 (via network or USB-C). As soon as a client connects and starts streaming:
- The server automatically stops capturing local microphone audio.
- It seamlessly transitions to recording the client's incoming PC stream.
- No server restart is needed!

1. (Optional) List local audio devices on the Coral Board:
   ```bash
   cd server
   go run . -list
   ```
2. Start the server (runs in hybrid mode, defaulting to system input mic):
   ```bash
   go run .
   ```
   *(To use a specific microphone, use the `-device <index>` flag, e.g. `go run . -device 1`).*

#### Network-Only Mode
If you want the server to bypass the local microphone entirely and ONLY wait for remote UDP client streams (e.g., when the Coral Board has no microphone attached and you want to save resources):
```bash
cd server
go run . -network
```

### 2. Start the Client on the Recording Machine (Network Mode Only)

#### List available audio input devices (to select microphone or loopback audio):
```bash
cd client
go run . -list
```
This will print a list of all detected devices on your computer.

#### Capture and stream audio to the server:
- **Default Device (Built-in Mic) & Local Server**:
  ```bash
  go run .
  ```
- **Custom Device Index & Remote Server (e.g. Remote meeting audio)**:
  If you are in a remote meeting and want to capture what others say, install a virtual audio loopback driver like **BlackHole** on macOS, list the devices to find its index, and start recording:
  ```bash
  go run . -device 2 -server 192.168.1.100:5000
  ```
  *(Replace `2` with the BlackHole or Aggregate Device index from your `-list` command, and replace the server IP with your Coral Board's IP).*

### 3. Control and Monitor the Recording

* **Start Recording**: Say the word **"go"** (or "yes" / "on"). The server will print in the console:
  `>>> [Coral Edge TPU] Signal 'GO' detected! Starting recording...`
* **Stop Recording**: Say the word **"stop"** (or "no" / "off"). The server will print:
  `>>> [Coral Edge TPU] Signal 'STOP' detected! Stopping recording...`
  The recording will be closed and automatically sent for transcription and uploaded to Google Drive.
* **Fallback (Inactivity - Network Mode Only)**: If you simply stop the client (pressing `Ctrl+C`), the server will detect that the network became inactive for 10 seconds, close the recording safely, and proceed to the transcription.
* **Acoustic Events**: Background sounds like laughter or applause will be logged on the screen in real-time by the Coral board and attached to the final generated Markdown transcript.
