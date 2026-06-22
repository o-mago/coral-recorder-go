# System Context & Coding Rules for AI Agents

Welcome, Agent! This document provides the complete technical context, architecture specifications, and repository-specific coding rules for the **Coral Recorder Go** project. Please review and adhere to these guidelines during any code modifications or additions.

---

## 1. Project Overview

**Coral Recorder Go** is a hybrid Go-Python client-server system designed to:
1. Capture real-time microphone audio from a client machine.
2. Stream raw audio packages over UDP to a server.
3. Process the audio on the **Synaptics Coral Dev Board Limited Edition 2GB** (Synaptics Astra SL2610 SoC) leveraging Edge-TPU/NPU accelerated models.
4. Detect local speech commands (wake-words like "go" / "stop") and acoustic events (laughter, clapping, cough) in real-time.
5. Upload the captured audio segment to the **Gemini 2.5 Flash API** using the Gemini File API to generate a speaker-separated transcription, executive summary, and action items in Portuguese, integrating the NPU-detected acoustic events.
6. Sync the resulting Markdown report to **Google Drive**.

---

## 2. Directory & Component Architecture

### `client/`
- **Main Client Application ([client/main.go](file:///Users/mago/dev/coral-recorder-go/client/main.go))**: Captures local microphone audio using the `portaudio` library, packs 16-bit PCM samples into a little-endian byte stream, and streams them over a UDP connection to the server.
- Supports CLI flags:
  - `-list`: Lists all input audio devices on the host PC.
  - `-device <index>`: Selects a specific audio device.
  - `-server <ip:port>`: Explicit server address.
  - `-usb`: Auto-points to the virtual network interface static IP (`192.168.100.2:5000`) assigned when connecting to the Coral board via USB-C OTG.

### `server/`
- **Go Orchestrator ([server/main.go](file:///Users/mago/dev/coral-recorder-go/server/main.go))**: Establishes the UDP socket receiver, launches the Python AI processing subprocess, writes raw incoming audio frames to the Python stdin, reads JSON events from Python stdout in real-time, and handles recording states.
- **Python Audio Processor ([server/coral_audio.py](file:///Users/mago/dev/coral-recorder-go/server/coral_audio.py))**: Runs TFLite/IREE runtime model inference on the incoming audio buffer. Detects voice commands (to start/stop recording) and tags background events.
- **Orchestration & Uploads ([server/upload.go](file:///Users/mago/dev/coral-recorder-go/server/upload.go))**: Manages the upload of `meeting.wav` to the Gemini File API, queries `gemini-2.5-flash` with the prompt and local events log, writes the markdown result, and triggers Drive sync.
- **Google Drive Sync ([server/drive.go](file:///Users/mago/dev/coral-recorder-go/server/drive.go))**: Syncs the markdown transcript to a specified Google Drive folder.

---

## 3. Communication Protocol

The Python subprocess (`coral_audio.py`) communicates status and acoustic events back to the Go Orchestrator (`main.go`) by writing single-line JSON messages to its `stdout`:

- **Control Commands:**
  `{"type": "control", "value": "start"}` (wake-word "go" detected - starts recording)
  `{"type": "control", "value": "done"}` (wake-word "stop" detected - stops recording and triggers transcription)
- **Acoustic Events:**
  `{"type": "event", "value": "laughter" | "clapping" | "cough", "timestamp": "HH:MM:SS"}`

Ensure any modification to these events is reflected in both the Python parser and the Go JSON structs (`CoralMessage`).

---

## 4. Hardware Specifications & Python Runtime Target

- **Target Board:** **Synaptics Coral Dev Board Limited Edition 2GB** (Synaptics Astra SL2610 SoC with Dual Core Arm Cortex-A55 & Cortex-M52), housing the integrated **Synaptics Torq NPU** (1 TOPS).
- **Model Compilation:** The NPU expects compiled neural networks in `.vmfb` format using the `torq-compile` toolchain (derived from MLIR/IREE). AI Agents should **always** guide users to compile models via Docker on their host development workstation (Mac/PC) rather than on the target board itself, owing to the board's 2GB RAM limits and lack of target compiler packages. The compiled `.vmfb` files should then be transferred to the board's `server/models/` directory (e.g., using `mdt push` for direct USB-C, or `scp`/`rsync` over the virtual network IP `192.168.100.2`).
  - *NPU Limitations:*
    - **YAMNet** cannot be compiled to `.vmfb` because it contains an RFFT (Real Fast Fourier Transform) yielding `complex<f32>` values, which are not supported by the TOSA dialect standard.
    - **Voice Commands** cannot be compiled to `.vmfb` because it relies on legacy TFLite custom ops (`AudioSpectrogram` and `Mfcc`) that cannot be legalized to standard TOSA NPU instructions.
- **Libraries & Dependencies:**
  - **`ai-edge-litert`**: Because the Synaptics board runs a modern ARM64 Linux image (with Python 3.11+), the legacy `tflite-runtime` package is deprecated and lacks PyPI wheels. Always use `ai-edge-litert` (LiteRT, the modern successor to TensorFlow Lite) and import `ai_edge_litert.interpreter` for CPU/fallback inference.
  - **`iree-base-runtime`**: Required to run compiled `.vmfb` models on the Torq NPU.
- **Compatibility Failbacks:** Keep CPU-fallback execution of standard `.tflite` models (via `ai-edge-litert` or standard `tensorflow`) and energy-based signal detection (`calculate_rms`) inside `coral_audio.py`. Because the audio models have NPU-incompatible preprocessing, they will always run on the board's ARM CPU via `ai-edge-litert`, which is highly optimized. This ensures developer portability to standard PCs and legacy platforms.

---

## 5. Repository Preferences & Coding Rules

- **English-Only Content:** All codebase contents—including variable names, function names, logging outputs, error logs, code comments, docstrings, and commit messages—**must** be written in English.
  - *Exception:* The final Markdown report and the Gemini prompt instruction template can include Portuguese phrasing to request transcription in Brazilian Portuguese.
- **Resource Management:** Ensure all pipes, socket channels, files, and audio streams are explicitly closed. Temporary files uploaded to the Gemini File API must be deleted immediately after transcription completes using `client.DeleteFile`.
- **Environment Isolation:** Do not suggest global pip package installations. Always instruct the user to run within a virtual Python environment (`python3 -m venv .venv`).

---
