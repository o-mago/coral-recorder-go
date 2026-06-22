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

#### 2. PortAudio on the Coral Dev Board (for local microphone mode)
The Yocto-based OS image does not ship PortAudio. If you want the server to run in hybrid or standalone mode using the board's built-in microphone (`klamath-asoc` DMIC), you must build PortAudio from source:

```bash
# Clone and build PortAudio from source on the board
cd ~/dev
git clone https://github.com/PortAudio/portaudio.git
cd portaudio
./configure
make
make install
```

This installs the library to `/usr/local/lib` and the pkg-config descriptor to `/usr/local/lib/pkgconfig/portaudio-2.0.pc`. Register both with the system:

```bash
# Make the dynamic linker aware of /usr/local/lib
echo '/usr/local/lib' >> /etc/ld.so.conf
ldconfig

# Make pkg-config find PortAudio (add to ~/.bashrc for persistence)
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
echo 'export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH' >> ~/.bashrc
```

Also add your user to the `audio` group so PortAudio can open `/dev/snd/*` devices:
```bash
usermod -aG audio $USER
# Apply immediately in the current shell without logging out:
newgrp audio
```

#### 3. ALSA configuration on the Coral Dev Board
The Astra SL2610 SoC exposes a built-in digital microphone as `card 0` (`klamath-asoc`). By default, `/etc/asound.conf` is empty and ALSA does not know what `default` means, causing PortAudio to fail. Create the mapping:

```bash
printf 'pcm.!default {\n    type hw\n    card 0\n    device 0\n}\n\nctl.!default {\n    type hw\n    card 0\n}\n' > /etc/asound.conf
```

Verify the microphone is visible before starting the server:
```bash
arecord -l
# Expected: card 0: klamathasoc [klamath-asoc], device 0: dummy-dmic
```

#### 4. LED permissions on the Coral Dev Board
The server controls the three onboard status LEDs (`green:status`, `red:status`, `blue:status`) by writing to `/sys/class/leds/*/brightness` via the Linux sysfs interface. By default these files are owned by `root` and only writable by root, so the server process (running as a regular user) must be granted write access.

**Persistent fix — create a udev rule** (applied automatically on every boot):
```bash
cat > /etc/udev/rules.d/99-leds.rules << 'EOF'
SUBSYSTEM=="leds", ACTION=="add", RUN+="/bin/chmod a+w /sys/class/leds/%k/brightness /sys/class/leds/%k/trigger"
EOF
```

**Immediate fix** (takes effect right now, without rebooting):
```bash
chmod a+w \
  /sys/class/leds/green:status/brightness \
  /sys/class/leds/green:status/trigger \
  /sys/class/leds/red:status/brightness \
  /sys/class/leds/red:status/trigger \
  /sys/class/leds/blue:status/brightness \
  /sys/class/leds/blue:status/trigger
```

> [!NOTE]
> The `chmod` above is lost on reboot. The udev rule ensures it is reapplied automatically whenever the LED devices are registered during boot. Both steps together are required for a permanent solution.

#### 5. On standard computers (macOS/Linux/Windows) to test on CPU
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
  Connect your PC directly to the Coral Board's USB-C data port (labeled OTG).

  The recommended setup on **macOS** is to use **Internet Sharing** as the DHCP gateway for the board. This gives the Coral internet access (required for the Gemini API upload) and a stable IP on the `192.168.2.x` subnet:

  > [!IMPORTANT]
  > The Coral Dev Board's Yocto image does not include Wi-Fi support. **macOS Internet Sharing is the only way to give the board internet access via USB-C** so that it can upload recordings to the Gemini File API.

  **macOS Internet Sharing setup (one-time):**
  1. Go to **System Settings > General > Sharing**.
  2. Click the **ⓘ** next to **Internet Sharing**.
  3. Set **Share your connection from:** to your active internet interface (e.g. Wi-Fi).
  4. Set **To computers using:** to the USB Ethernet adapter (the one that appears when the Coral is plugged in — typically listed as an NCM/RNDIS gadget or `en10`).
  5. Enable **Internet Sharing**. macOS will create a `bridge100` interface at `192.168.2.1` and run a DHCP server on it.

  **On the Coral board (one-time):** configure `usb0` to use DHCP so it gets an IP from macOS automatically on every boot:
  ```bash
  # Edit /etc/network/interfaces: replace the static usb0 block with:
  # auto usb0
  # iface usb0 inet dhcp
  #
  # Then request an IP immediately (no reboot needed):
  udhcpc -i usb0
  ```
  After this the board will be reachable at `192.168.2.2` and will route internet traffic through the Mac.

  **Start the client** on the Mac using the `-usb` flag (auto-detects `192.168.2.2`):
  ```bash
  go run . -usb
  # Or explicitly:
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
        
        var blob;
        if (mimeType === "audio/wav") {
          // Decode the Base64 binary payload sent by the Go server
          var bytes = Utilities.base64Decode(content);
          blob = Utilities.newBlob(bytes, mimeType, fileName);
        } else {
          // Create a standard text/markdown file using a blob
          blob = Utilities.newBlob(content, mimeType, fileName);
        }
        
        var file = folder.createFile(blob);
        
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
By default, the server starts in a hybrid mode. It captures audio from the local microphone (using the board's built-in DMIC or a connected USB mic) but concurrently listens for client UDP connections on port 5000. As soon as a client connects and starts streaming:
- The server automatically stops capturing local microphone audio.
- It seamlessly transitions to recording the client's incoming PC stream.
- No server restart is needed!

> [!NOTE]
> On the Coral Dev Board, PortAudio must be built from source and `PKG_CONFIG_PATH` must be set before running `go run` (see Prerequisites). The `audio` group and `/etc/asound.conf` must also be configured (one-time setup).

1. (Optional) List local audio devices on the Coral Board:
   ```bash
   cd server
   PKG_CONFIG_PATH=/usr/local/lib/pkgconfig go run . -list
   ```
2. Start the server (runs in hybrid mode, defaulting to system input mic):
   ```bash
   cd server
   PKG_CONFIG_PATH=/usr/local/lib/pkgconfig go run .
   ```
   *(To use a specific microphone, use the `-device <index>` flag, e.g. `go run . -device 1`.)*
   *(After adding `PKG_CONFIG_PATH` to `~/.bashrc`, you can omit the prefix.)*

#### Network-Only Mode
If you want the server to bypass the local microphone entirely and ONLY wait for remote UDP client streams (e.g., to save resources when no microphone is needed on the board side):
```bash
cd server
# With PortAudio installed from source (Coral Dev Board):
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig go run . -network

# Without PortAudio (build tag disables the dependency entirely):
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

- **Internet-Only Mode (Proxy only, local mic on board)**:
  If you want the board to use its own local microphone for recording, but still need the Mac to provide it with internet access (through the USB ADB proxy) to upload the results, run the client with the `-internet-only` flag:
  ```bash
  go run . -internet-only
  ```
  This will spin up the proxy server and configure ADB reverse port forwarding, then run indefinitely in the background without opening the Mac's audio device or sending any UDP streams.

### 3. Control and Monitor the Recording

* **Offline Voice Commands (Vosk)**:
  To prevent false triggers during conversational speech, voice commands are guarded by the wake-word **"coral"**. Say "coral" followed by one of the following commands:
  - **Start Recording**: *"coral, iniciar"* / *"coral, começar gravação"* / *"coral, começar a gravar"* (or English *"coral, start"* / *"coral, go"*).
  - **Stop Recording**: *"coral, parar"* / *"coral, terminar"* / *"coral, encerrar gravação"* (or English *"coral, stop"*).
  *(Single command words spoken in complete isolation may also trigger).*

* **LED Visual Feedback**:
  The server drives the three onboard status LEDs of the Astra SL2610 via the Linux sysfs interface (`/sys/class/leds/`) to give a clear at-a-glance view of what the board is doing — no terminal required.

  | LED | Pattern | State | Description |
  |---|---|---|---|
  | 🟢 **Green** | Solid ON | **Idle (Local)** | Standalone mode — monitoring local microphone, waiting for **"GO"** command (no errors) |
  | 🟢+🔵 **Green+Blue** | Solid ON | **Idle (Network)** | Client-Server mode — waiting for client UDP stream, waiting for **"GO"** command (no errors) |
  | 🟢 + 🔴 **Green + Blinking Red** | Solid Green, Blinking Red | **Idle (Local with Queue Errors)** | Standalone mode ready to record, but failed recordings are waiting in the queue |
  | 🟢+🔵 + 🔴 **Cyan + Blinking Red** | Solid Cyan, Blinking Red | **Idle (Network with Queue Errors)** | Client-Server mode ready to connect, but failed recordings are waiting in the queue |
  | 🔴 **Red** | Solid ON | **Recording** | Recording in progress (takes absolute priority) |
  | 🔵 + 🔴 **Blue + Blinking Red** | Solid Blue, Blinking Red | **Retrying Queue** | Actively re-processing and uploading failed recordings in the background |
  | 🔵 **Blue** | Solid ON | **Processing** | Actively uploading and transcribing a fresh session |
  | All | OFF | **Offline** | Server not running |

  After a successful session finishes, the server automatically resets to its idle color (🟢 **Green** or 🟢+🔵 **Cyan**) and waits for the next **"GO"** command. If there are pending/failed uploads in the queue, the red LED will blink alongside the solid idle colors to indicate that it is ready for a new recording but still has unsent data. During background queue retries, the blue LED stays solid while the red LED blinks.

* **Diagnostics and Logs Flag (`-status`)**:
  To inspect the service status, local upload queue, and logs directly from the command line, run:
  ```bash
  /home/mago/dev/coral-recorder-go/server/coral-recorder -status
  ```
  *(To run as a non-root user and view journal logs, make sure you are in the `systemd-journal` group or run `newgrp systemd-journal` in your session).*

* **Continuous Session Loop**:
  The server never exits after a recording. Once transcription completes, it immediately restarts the Python audio processor and returns to the idle state, ready for the next meeting. Each new session is logged as `=== [Session N] Ready for new recording... ===`.

* **CPU Optimization & Stability**:
  - The Vosk speech decoder runs with a restricted JSON grammar vocabulary containing only the control keywords, lowering CPU overhead by 90% (processing chunks in <20ms) and preventing ALSA/PortAudio buffer overflows.
  - The YAMNet acoustic classifier automatically disables when the recorder is in standby, and subsamples inference frames by 2 during recording to keep ARM CPU usage optimal.

* **Outputs, Safety Queue & Google Drive Sync**:
  Once recording stops (either via voice command, or automatically after 10 seconds of network inactivity):
  1. The server immediately renames `meeting.wav` to `pending_<timestamp>.wav` and marshals Edge-TPU events to `pending_<timestamp>.json` inside the queue directory. This ensures the files survive reboots/power cycles and are not overwritten by the next recording.
  2. The main loop immediately starts a new recording session, allowing you to record a new meeting while the previous one is uploaded asynchronously in the background.
  3. The background retry worker uploads `pending_<timestamp>.wav` to the Gemini File API to generate the meeting minutes report `transcricao_<timestamp>.md`. (If this markdown file already exists from a previous successful Gemini run, the system reuses it to save API usage).
  4. Both the report and the raw audio are uploaded to Google Drive using a timestamp-prefixed format:
     - `YYYY-MM-DD_HH-MM-SS_Transcript_Meeting.md`
     - `YYYY-MM-DD_HH-MM-SS_Audio_Meeting.wav`
     - The local files are **only** deleted after successful upload of *both* files, ensuring 100% data safety.

* **Acoustic Events**: Background sounds (e.g. laughter, clapping, coughing) detected by the YAMNet model on the Coral Board are printed on the server console in real-time and automatically integrated into the transcription report by Gemini (e.g., `[Risos]`, `[Palmas]`).


---

## Autostart on Boot (Systemd)

To make the server run automatically when the Coral Dev Board boots up, you can configure a systemd service.

### 1. Compile the Server Binary on the Board
The systemd service runs the precompiled `coral-recorder` binary. Build it in the repository directory on the board:
```bash
cd ~/dev/coral-recorder-go/server
PKG_CONFIG_PATH=/usr/local/lib/pkgconfig /home/mago/go/bin/go build -o coral-recorder .
```

### 2. Create the Systemd Service File
Create a new unit file at `/etc/systemd/system/coral-recorder.service` with root privileges (using `sudo`):

```ini
[Unit]
Description=Coral Recorder - Audio capture and Gemini transcription server
After=sound.target

[Service]
Type=simple
User=mago
Group=mago
SupplementaryGroups=audio
WorkingDirectory=/home/mago/dev/coral-recorder-go/server
Environment=PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
Environment=http_proxy=http://127.0.0.1:8082
Environment=https_proxy=http://127.0.0.1:8082
EnvironmentFile=-/home/mago/dev/coral-recorder-go/server/.env
ExecStartPre=+/bin/chmod a+w /sys/class/leds/green:status/brightness /sys/class/leds/green:status/trigger /sys/class/leds/red:status/brightness /sys/class/leds/red:status/trigger /sys/class/leds/blue:status/brightness /sys/class/leds/blue:status/trigger
ExecStart=/home/mago/dev/coral-recorder-go/server/coral-recorder
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=coral-recorder

[Install]
WantedBy=multi-user.target
```

> [!IMPORTANT]
> - Do **not** set `Environment=LD_LIBRARY_PATH=/usr/local/lib` in this service file. The custom-compiled PortAudio library in `/usr/local/lib` may lack ALSA support, whereas the system library `/usr/lib/libportaudio.so.2` has full ALSA support and is loaded correctly by default.
> - The `+` prefix before `/bin/chmod` in `ExecStartPre` tells systemd to run that specific permission-granting command with root privileges even though the main service runs as user `mago`.
> - `SupplementaryGroups=audio` ensures the process inherits the permissions of the `audio` group to access ALSA soundcard device nodes in `/dev/snd/*`.
> - `Environment=*_proxy=http://127.0.0.1:8082` routes internet traffic through the USB-C ADB reverse tunnel to the host PC proxy, allowing Gemini API uploads and Google Drive sync without complex network configuration on the host.

### 3. Control the Service

Reload the systemd daemon configuration, enable the service (to start on boot), and start it immediately:
```bash
sudo systemctl daemon-reload
sudo systemctl enable coral-recorder
sudo systemctl start coral-recorder
```

To stop the service and turn off the status LEDs:
```bash
sudo systemctl stop coral-recorder
```

To disable autostart:
```bash
sudo systemctl disable coral-recorder
```

### 4. Monitor and Troubleshoot

Check the service status:
```bash
sudo systemctl status coral-recorder
```

Follow the logs in real-time (useful for checking transcription/upload progress and voice commands):
```bash
sudo journalctl -u coral-recorder -f
```

