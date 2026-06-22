# Coral Recorder Go

This project is a Go-based client-server system designed to capture audio from a microphone, stream it in real-time over UDP, process it locally using the **Synaptics Coral Dev Board Limited Edition 2GB (Synaptics Astra SL2610 SoC with Torq NPU)**, and leverage the Gemini 2.5 Flash API to transcribe, identify speakers, and extract action items from the meeting.

## Architecture

1. **Client**: Captures local microphone audio using the `portaudio` library, converts `int16` samples to bytes (`Little Endian`), and streams them over the network using the UDP protocol.
2. **Server**: Listens on a UDP port (default `:5000`) and forwards the raw audio stream to a local Python subprocess (`coral_audio.py`), which runs optimized detection models:
   - **Voice Commands (Keyword Spotting)**: Continuously listens in the background. It uses Vosk for offline Portuguese command detection guarded by the wake-word **"coral"** (e.g., *"coral, iniciar gravação"*, *"coral, stop"*) to control recording.
   - **Intelligent VAD (Voice Activity Detection)**: Saves only segments containing active human speech to `meeting.wav` (using a hangover buffer to prevent clipping), removing silent periods.
   - **Audio Event Classification (YAMNet)**: Detects and logs background audio events such as laughter, clapping, and coughing in real-time, sending timestamps to Go.
3. **Gemini & Cloud Orchestration**: Once recording is finalized, Go uploads the processed `meeting.wav` file to the **Gemini File API**, appends the list of local events detected by the NPU to the prompt, and requests a speaker-separated transcription integrated with the acoustic events.


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

### Server (Synaptics Astra SL2610 NPU / CPU Fallback)
The server runs a hybrid pipeline with Python 3. Install the required dependencies:

#### 1. On the Synaptics Coral Dev Board (Astra SL2610)
Because this board runs a modern ARM64 architecture with newer Python versions (like Python 3.11+), the legacy `tflite-runtime` package might not have precompiled wheels available on PyPI. Instead, install **`ai-edge-litert`** (the official modern successor to TFLite runtime from Google) and **`vosk`** (for speech recognition) alongside the other dependencies:

```bash
# Install packages globally (if supported by your system image)
pip3 install ai-edge-litert numpy iree-base-runtime vosk
```

If your system blocks global pip installs with an "externally-managed-environment" error, create a virtual environment:
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install ai-edge-litert numpy iree-base-runtime vosk
```
*(Note: `iree-base-runtime` is required for loading compiled `.vmfb` models to accelerate inference on the Torq NPU).*

#### 2. On standard computers (macOS/Linux/Windows) to test on CPU
If you wish to test the server locally on your host PC:
```bash
pip3 install ai-edge-litert numpy vosk
```
*(Alternatively, you can also use `pip3 install tensorflow numpy vosk` if you already have the full TensorFlow package installed).*


---

## Configuration

### 1. Download and Compile Local Models
Navigate to the server folder and run the download script to fetch the pre-trained YAMNet and Voice Commands models:
```bash
cd server
./download_models.sh
```

#### Compilation Guideline for Synaptics Astra SL2610 NPU:
> [!IMPORTANT]
> **Compilation Workflow Recommendation:**
> Always compile models on a host machine (such as your macOS/Linux workstation) using Docker, and then copy the generated `.vmfb` files to the Coral Board. Do not run or install the compiler directly on the target Coral board due to resource constraints (the compilation process is heavy and can cause Out of Memory errors on the board's 2GB RAM).
> 
> The board only needs the runtime packages (`torq-runtime`, `torq-runtime-python`), which are pre-installed on the official board OS image, to execute the compiled `.vmfb` models.

To compile the models on your developer host machine (using Docker):
1. Authenticate with GitHub Container Registry:
   ```bash
   docker login ghcr.io
   ```
2. Navigate to the `server/` directory and run the download script, which will automatically detect Docker and compile the models:
   ```bash
   cd server
   ./download_models.sh
   ```
3. Once compiled, transfer the `server/models/` folder containing the `.vmfb` and `.tflite` files to your Coral Dev Board.

   If you are connected via USB-C (OTG) and have the **Mendel Development Tool (`mdt`)** installed, you can push the folder directly (from the `server/` directory on your host PC):
   * **Using `mdt` (Direct USB-C):**
     ```bash
     mdt push models/ ~/dev/coral-recorder-go/server/models/
     ```

   Alternatively, if you prefer using SSH over the USB-C virtual Ethernet interface, the board's static IP is typically either **`192.168.100.2`** (standard Mendel default) or **`192.168.2.2`** (depending on how host network interfaces/bridges route the USB gadget link). 
   
   Run one of the following commands from the `server/` directory on your host PC (replacing `192.168.2.2` with `192.168.100.2` if necessary, and `mago` with your board's username):
   * **Using `scp` (USB-C network IP):**
     ```bash
     scp -r models/* mago@192.168.2.2:~/dev/coral-recorder-go/server/models/
     ```
   * **Using `rsync` (Incremental/Faster Sync):**
     ```bash
     rsync -avz models/ mago@192.168.2.2:~/dev/coral-recorder-go/server/models/
     ```

Alternatively, you can run the Docker compilation commands manually on your host machine:
```bash
docker run --rm -v $(pwd):/work -w /work ghcr.io/synaptics-torq/torq-compiler/compiler:main \
    torq-compile --input-model=models/yamnet.tflite --output-file=models/yamnet.vmfb --target-device=synaptics-npu

docker run --rm -v $(pwd):/work -w /work ghcr.io/synaptics-torq/torq-compiler/compiler:main \
    torq-compile --input-model=models/voice_commands.tflite --output-file=models/voice_commands.vmfb --target-device=synaptics-npu
```
*(Note: If no `.vmfb` files are present on the board, the python backend will automatically fall back to CPU execution using the `.tflite` models).*


### 2. Configure Server IP Address on the Client
You do not need to edit the client code. By default, the client sends audio to `localhost:5000`. You can specify the server address dynamically using the `-server` command-line flag depending on your connection method:

- **Via Local Network (Wi-Fi or Ethernet)**:
  Find your Coral Board's network IP address (e.g. `192.168.1.100`) and run:
  ```bash
  go run . -server 192.168.1.100:5000
  ```
- **Via USB-C Cable (Direct Connection)**:
  Connect your PC directly to the Coral Board's USB-C data port (labeled OTG). Mendel Linux automatically sets up a virtual network interface (USB Ethernet Gadget), assigning a static IP (typically **`192.168.100.2`** or **`192.168.2.2`**) to the Coral Board.
  1. Start the server on the Coral Board in network mode:
     ```bash
     go run . -network
     ```
  2. Start the client on the PC pointing to the virtual USB IP:
     - Using the convenient `-usb` flag (recommended - auto-detects the host interface subnet subnet and points to either `192.168.2.2` or `192.168.100.2` dynamically):
       ```bash
       go run . -usb
       ```
     - Or by manually specifying the IP and port with the `-server` flag:
       ```bash
       go run . -server 192.168.2.2:5000
       ```

### 3. Gemini API Key
Define the `GEMINI_API_KEY` environment variable:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

#### How to Generate the API Key:
1. Go to [Google AI Studio](https://aistudio.google.com/).
2. Log in with your Google Account.
3. Click on the **Get API key** (or **Create API key**) button.
4. Click on **Create API key** (Criar chave de API).
5. Select an existing Google Cloud project or create a new one to bind the key to.
6. Copy the generated API key string.
7. Set the `GEMINI_API_KEY` environment variable in your terminal using the copied key.

### 4. Google Drive Integration (Optional)
To save your meeting reports and raw audio recordings to Google Drive, you can configure the server to route uploads through a lightweight **Google Apps Script Web App**. 

This method bypasses the `0-byte` storage quota limit imposed on Google Service Accounts for free `@gmail.com` accounts, utilizing your personal Google Drive storage instead.

#### How to Set Up the Google Apps Script Gateway:

1. Open your browser and go to [Google Apps Script](https://script.google.com/).
2. Click **New Project** (Novo projeto) or open an existing script project.
3. Paste the following JavaScript code in the editor (replacing all code):
   ```javascript
   function doPost(e) {
     try {
       var data = JSON.parse(e.postData.contents);
       var folderId = data.folderId;
       var fileName = data.fileName;
       var content = data.content;
       var mimeType = data.mimeType || "text/markdown";
       
       var folder = DriveApp.getFolderById(folderId);
       
       // Check if the file already exists to avoid duplicates
       var files = folder.getFilesByName(fileName);
       while (files.hasNext()) {
         files.next().setTrashed(true); // Move old version to trash
       }
       
       var file;
       if (mimeType === "audio/wav") {
         // Decode the Base64 binary payload sent by the Go server
         var bytes = Utilities.base64Decode(content);
         var blob = Utilities.newBlob(bytes, mimeType, fileName);
         file = folder.createFile(blob);
       } else {
         // Create a standard text/markdown file
         file = folder.createFile(fileName, content, mimeType);
       }
       
       return ContentService.createTextOutput(JSON.stringify({
         status: "success",
         fileId: file.getId()
       })).setMimeType(ContentService.MimeType.JSON);
       
     } catch (error) {
       return ContentService.createTextOutput(JSON.stringify({
         status: "error",
         message: error.toString()
       })).setMimeType(ContentService.MimeType.JSON);
     }
   }
   ```
4. Click **Save** (diskette icon).
5. Click **Deploy** > **New deployment** (Nova implantação) in the top-right corner.
6. Click the gear icon next to "Select type" and choose **Web app** (Aplicativo da Web).
7. Configure the settings:
   - **Description**: `Coral Audio and Text Upload Gateway`
   - **Execute as** (Executar como): **Me** (your-email@gmail.com)
   - **Who has access** (Quem tem acesso): **Anyone** (Qualquer pessoa)
8. Click **Deploy**, authorize the required permissions with your Google account, and copy the generated **Web app URL** (ends with `/exec`).

#### Environmental Variables:
Define these variables in your shell environment or add them directly to a `server/.env` file:
```bash
export GEMINI_API_KEY="your_api_key_here"
export GOOGLE_DRIVE_FOLDER_ID="your_google_drive_folder_id"
export GOOGLE_DRIVE_WEB_APP_URL="https://script.google.com/macros/s/XXXXX/exec"
```
*(The Folder ID is the long alphanumeric string at the end of the URL when viewing your Google Drive folder in a browser, e.g., `https://drive.google.com/drive/folders/YOUR_FOLDER_ID`)*

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
# On standard systems (with PortAudio installed):
go run . -network

# On the Yocto-based Coral Dev Board (which lacks PortAudio libraries):
go run -tags no_portaudio . -network
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

* **Offline Voice Commands (Vosk)**:
  To prevent false triggers during conversational speech, voice commands are guarded by the wake-word **"coral"**. Say "coral" followed by one of the following commands:
  - **Start Recording**: *"coral, iniciar"* / *"coral, começar gravação"* / *"coral, começar a gravar"* (or English *"coral, start"* / *"coral, go"*).
  - **Stop Recording**: *"coral, parar"* / *"coral, terminar"* / *"coral, encerrar gravação"* (or English *"coral, stop"*).
  *(Single command words spoken in complete isolation may also trigger).*
  
* **CPU Optimization & Stability**:
  - The Vosk speech decoder runs with a restricted JSON grammar vocabulary containing only the control keywords, lowering CPU overhead by 90% (processing chunks in <20ms) and preventing ALSA/PortAudio buffer overflows.
  - The YAMNet acoustic classifier automatically disables when the recorder is in standby, and subsamples inference frames by 2 during recording to keep ARM CPU usage optimal.

* **Outputs & Google Drive Sync**:
  Once recording stops (either via voice command, or automatically after 10 seconds of network inactivity):
  1. The server uploads `meeting.wav` to Gemini File API to generate meeting minutes.
  2. A Markdown meeting report is created. It places the **Executive Summary (Sumário Executivo)** and **Action Items (Itens de Ação)** right at the top for immediate visibility, with the **Detailed Transcription (Transcrição Detalhada)** (with speaker identification) at the bottom.
  3. If Drive integration is configured, both the report and the raw audio are uploaded to Google Drive using a timestamp-prefixed format:
     - `YYYY-MM-DD_HH-MM-SS_Transcript_Meeting.md`
     - `YYYY-MM-DD_HH-MM-SS_Audio_Meeting.wav`
  4. The local temporary `meeting.wav` and the generated markdown transcript files are immediately deleted after successful upload to maintain a **`0-byte` permanent disk footprint** on the Coral board.

* **Acoustic Events**: Background sounds (e.g. laughter, clapping, coughing) detected by the YAMNet model on the Coral Board are printed on the server console in real-time and automatically integrated into the transcription report by Gemini (e.g., `[Risos]`, `[Palmas]`).
