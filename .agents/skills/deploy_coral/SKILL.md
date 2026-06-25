---
name: deploy-coral
description: Explains how to push code changes, cross-compile, build, and restart the Coral Recorder Go server on the Synaptics Coral Dev Board.
---

# Deploy Coral Recorder Go Server to the Coral Dev Board

This skill guides you through compiling, pushing, and restarting the Go-based Orchestrator (`server/`) on the Synaptics Coral Dev Board (Synaptics Astra SL2610 SoC).

## Pre-requisites
1. The Coral Board must be connected via USB-C OTG or be reachable on the local network (default virtual network IP is `192.168.100.2` or `192.168.137.2`).
2. `adb` must be installed and running on the host system. Validate connectivity using `adb devices`.

## Step-by-Step Deployment

### 1. Push Code Changes to the Board
Use `adb push` to copy local Go code changes in the `server/` directory to the board's workspace `/home/mago/dev/coral-recorder-go/server/`.

Example for `server/upload.go`:
```bash
adb push server/upload.go /home/mago/dev/coral-recorder-go/server/upload.go
```

If multiple files in the server package changed, push the whole directory structure (avoiding `.git` or database files):
```bash
adb push server/ /home/mago/dev/coral-recorder-go/
```

### 2. Compile/Rebuild the Binary on the Board
Because of library dependencies (e.g. `portaudio` package dependencies if enabled, or local compiler configs), always build the Go orchestrator directly on the board.

Run the build command over `adb shell`. Use `-buildvcs=false` to avoid VCS git configuration errors on the board:
```bash
adb shell "cd /home/mago/dev/coral-recorder-go/server && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig /home/mago/go/bin/go build -buildvcs=false -o coral-recorder ."
```

### 3. Restart the Service
Restart the background systemd service (`coral-recorder.service`) to run the newly compiled binary:
```bash
adb shell "systemctl restart coral-recorder"
```

### 4. Verify Service Health
Monitor the logs to confirm the service restarted successfully and the Python co-processor initialized correctly:
```bash
adb shell "journalctl -u coral-recorder.service -n 30"
```
Ensure you see the message: `Ready to receive audio stream over stdin...`.
