# Coral Recorder Go

This project is a Go-based client-server system designed to capture audio from a microphone, stream it in real-time over UDP, process it locally using the **Synaptics Coral Dev Board Limited Edition 2GB (Synaptics Astra SL2610 SoC with Torq NPU)**, and leverage the Gemini 2.5 Flash API to transcribe, identify speakers, and extract action items from the meeting.

## Architecture

1. **Client**: Captures local microphone audio using the `portaudio` library, converts `int16` samples to bytes (`Little Endian`), and streams them over the network using the UDP protocol.
2. **Server**: Listens on a UDP port (default `:5000`) and forwards the raw audio stream to a local Python subprocess (`coral_audio.py`), which runs NPU-accelerated models:
   - **Voice Commands (Keyword Spotting)**: Continuously listens in the background. It starts recording upon hearing **"go"** (or "yes"/"on") and stops upon hearing **"stop"** (or "no"/"off").
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
Because this board runs a modern ARM64 architecture with newer Python versions (like Python 3.11+), the legacy `tflite-runtime` package might not have precompiled wheels available on PyPI. Instead, install **`ai-edge-litert`** (the official modern successor to TFLite runtime from Google) alongside the other dependencies:

```bash
# Install packages globally (if supported by your system image)
pip3 install ai-edge-litert numpy iree-base-runtime
```

If your system blocks global pip installs with an "externally-managed-environment" error, create a virtual environment:
```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install ai-edge-litert numpy iree-base-runtime
```
*(Note: `iree-base-runtime` is required for loading compiled `.vmfb` models to accelerate inference on the Torq NPU).*

#### 2. On standard computers (macOS/Linux/Windows) to test on CPU
If you wish to test the server locally on your host PC:
```bash
pip3 install ai-edge-litert numpy
```
*(Alternatively, you can also use `pip3 install tensorflow numpy` if you already have the full TensorFlow package installed).*


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
Define the variables to automatically save the generated meeting notes to your Google Drive account:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-credentials.json"
export GOOGLE_DRIVE_FOLDER_ID="1A2B3C4D5E...your-folder-id"
```

#### How to Generate the Credentials:

1. **Create or Select a Google Cloud Project**:
   * Go to the [Google Cloud Console](https://console.cloud.google.com/).
   * Create a new project or select an existing one (e.g., `Coral Recorder`).
2. **Enable the Google Drive API**:
   * Open the left sidebar menu and navigate to **APIs & Services** > **Library** (APIs e Serviços > Biblioteca).
   * Search for **Google Drive API**, click on it, and click **Enable** (Ativar).
3. **Create a Service Account**:
   * Go to **IAM & Admin** > **Service Accounts** (IAM e Administrador > Contas de Serviço).
   * Click **Create Service Account** at the top.
   * Provide a name (e.g., `coral-recorder-drive`) and click **Create and Continue**, then click **Done**.
4. **Generate and Download the JSON Key**:
   * In the Service Accounts list, click on the email of the Service Account you just created.
   * Navigate to the **Keys** (Chaves) tab at the top.
   * Click **Add Key** > **Create new key** (Adicionar Chave > Criar nova chave).
   * Select **JSON** format and click **Create**. A `.json` file containing your private key will be downloaded. Save it securely on your machine (avoid keeping it inside the git repository directory to prevent accidental commits).
5. **Share the Target Folder in Google Drive (Crucial!)**:
   * Service Accounts do not have access to your personal files or folders by default.
   * Open your personal **Google Drive** in your browser and select (or create) the folder you want to use.
   * Right-click the folder and choose **Share** (Compartilhar).
   * Copy the Service Account email address (you can find it in the Google Cloud Console list or inside the downloaded JSON file under the `"client_email"` key, e.g. `coral-recorder-drive@project-id.iam.gserviceaccount.com`).
   * Paste the email in the share field, set the role to **Editor**, and click **Share**.
   * Copy the **Folder ID** from the browser's URL bar (the long alphanumeric string at the end of `https://drive.google.com/drive/folders/YOUR_FOLDER_ID`) and use it for the `GOOGLE_DRIVE_FOLDER_ID` environment variable.

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

* **Start Recording**: Say the word **"go"** (or "yes" / "on"). The server will print in the console:
  `>>> [Coral Edge TPU] Signal 'GO' detected! Starting recording...`
* **Stop Recording**: Say the word **"stop"** (or "no" / "off"). The server will print:
  `>>> [Coral Edge TPU] Signal 'STOP' detected! Stopping recording...`
  The recording will be closed and automatically sent for transcription and uploaded to Google Drive.
* **Fallback (Inactivity - Network Mode Only)**: If you simply stop the client (pressing `Ctrl+C`), the server will detect that the network became inactive for 10 seconds, close the recording safely, and proceed to the transcription.
* **Acoustic Events**: Background sounds like laughter or applause will be logged on the screen in real-time by the Coral board and attached to the final generated Markdown transcript.
