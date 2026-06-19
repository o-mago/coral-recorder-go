# Repository Preferences and Coding Rules

These rules are enforced for any agent working in this repository.

## 1. Language Rule
- **English-Only Content**: All code comments, docstrings, variable definitions, error messages, logging outputs, commit messages, and README files **must** be written in English.
  - *Exception*: The final generated Markdown transcript sent to the user can include translations or Portuguese output if specifically requested in the prompt instructions to Gemini.

## 2. Hybrid Architecture (Go + Python)
- **Go Server (`server/main.go`)**: Manages the UDP socket, pipes raw audio data to the Python subprocess's `stdin`, and reads JSON lines from its `stdout` in a background thread. It orchestrates Gemini uploads and Google Drive sync.
- **Python Processor (`server/coral_audio.py`)**: Runs Edge TPU-accelerated TFLite models (YAMNet, Speech Commands) on standard input audio frames. It operates in streaming mode, manages local WAV writing, and emits status/events to `stdout`.
- **Go Client (`client/main.go`)**: Captures local audio and streams it to the server. Must support command-line arguments to list devices (`-list`), select devices (`-device <index>`), and configure the destination server address (`-server`).

## 3. Communication Protocol
- The Python script communicates with the Go server by writing JSON lines to `stdout` in the following format:
  - **Control Commands**: `{"type": "control", "value": "start" | "done"}`
  - **Acoustic Events**: `{"type": "event", "value": "laughter" | "clapping" | "cough", "timestamp": "HH:MM:SS"}`

## 4. Resource Management
- All WAV writers, files, pipes, and network streams must be closed explicitly on finalization or exit to prevent resource leaks and guarantee files are complete before upload.
- Temporary files uploaded to the Gemini File API must be deleted immediately after transcription via `client.DeleteFile`.

## 5. Coral Edge TPU Optimization
- **Edge TPU Prioritization**: Always prioritize offloading machine learning inferences (such as TFLite models for audio/speech commands and classifications) to the Coral Edge TPU whenever possible and when it makes technical sense.
- **Hardware Limitations & Fallbacks**: Respect the hardware resource limits of the Coral Board (memory, thermals, and compiler constraint limits). Always design modular fallback layers (CPU models, RMS calculations) so the application remains robustly functional even if TPU libraries or hardware acceleration is not present.
