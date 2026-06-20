#!/bin/bash
set -e

# Create models directory if it doesn't exist
mkdir -p models

echo "=== Downloading Audio Models for Coral Recorder ==="

# 1. YAMNet (Audio Event Classification and Speech Detection)
if [ ! -f "models/yamnet.tflite" ]; then
    echo "Downloading YAMNet (CPU TFLite)..."
    curl -L -o models/yamnet.tflite "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/classification/tflite/1.tflite"
else
    echo "YAMNet model already exists. Skipping..."
fi

# 2. Speech Commands (Wake Word / Commands Model) - CPU
if [ ! -f "models/voice_commands.tflite" ]; then
    echo "Downloading Speech Commands (CPU TFLite)..."
    curl -L -o models/voice_commands.tflite "https://raw.githubusercontent.com/google-coral/project-keyword-spotter/master/models/voice_commands_v0.7.tflite"
else
    echo "Speech Commands CPU model already exists. Skipping..."
fi

# 3. Speech Commands (Wake Word / Commands Model) - Edge TPU
if [ ! -f "models/voice_commands_edgetpu.tflite" ]; then
    echo "Downloading Speech Commands (Edge TPU TFLite)..."
    curl -L -o models/voice_commands_edgetpu.tflite "https://raw.githubusercontent.com/google-coral/project-keyword-spotter/master/models/voice_commands_v0.7_edgetpu.tflite"
else
    echo "Speech Commands Edge TPU model already exists. Skipping..."
fi

# 4. Vosk Portuguese Speech Model (Optional, for Portuguese Voice Commands)
if [ ! -d "model" ]; then
    echo "Downloading Vosk Portuguese Voice Model (Small)..."
    curl -L -o models/vosk-model.zip "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip"
    echo "Extracting model..."
    unzip -q models/vosk-model.zip -d models/
    mv models/vosk-model-small-pt-0.3 model
    rm models/vosk-model.zip
    echo "Vosk Portuguese Model configured successfully!"
else
    echo "Vosk Portuguese Model directory 'model' already exists. Skipping..."
fi

# 5. Compile to Synaptics Astra NPU format (.vmfb) if torq-compile is available
if command -v torq-compile &> /dev/null; then
    echo "=== Detected torq-compile tool. Compiling models for Synaptics Astra NPU... ==="
    
    if [ ! -f "models/yamnet.vmfb" ]; then
        echo "Compiling YAMNet for Synaptics Astra NPU..."
        torq-compile --input-model=models/yamnet.tflite --output-file=models/yamnet.vmfb --target-device=synaptics-npu
    else
        echo "yamnet.vmfb already exists. Skipping..."
    fi

    if [ ! -f "models/voice_commands.vmfb" ]; then
        echo "Compiling Speech Commands for Synaptics Astra NPU..."
        torq-compile --input-model=models/voice_commands.tflite --output-file=models/voice_commands.vmfb --target-device=synaptics-npu
    else
        echo "voice_commands.vmfb already exists. Skipping..."
    fi
elif command -v docker &> /dev/null; then
    echo "=== 'torq-compile' not found, but 'docker' is available. Attempting to compile using Torq compiler Docker container... ==="
    
    # Try compiling YAMNet first
    if [ ! -f "models/yamnet.vmfb" ]; then
        echo "Compiling YAMNet via Docker..."
        if docker run --rm -v "$(pwd):/work" -w /work ghcr.io/synaptics-torq/torq-compiler/compiler:main \
            torq-compile --input-model=models/yamnet.tflite --output-file=models/yamnet.vmfb --target-device=synaptics-npu; then
            echo "YAMNet compiled successfully!"
        else
            echo "Failed to compile YAMNet. You may need to log in to GitHub Container Registry first (docker login ghcr.io)."
        fi
    else
        echo "yamnet.vmfb already exists. Skipping..."
    fi

    # Try compiling Speech Commands
    if [ ! -f "models/voice_commands.vmfb" ]; then
        echo "Compiling Speech Commands via Docker..."
        if docker run --rm -v "$(pwd):/work" -w /work ghcr.io/synaptics-torq/torq-compiler/compiler:main \
            torq-compile --input-model=models/voice_commands.tflite --output-file=models/voice_commands.vmfb --target-device=synaptics-npu; then
            echo "Speech Commands compiled successfully!"
        fi
    else
        echo "voice_commands.vmfb already exists. Skipping..."
    fi
else
    echo "=== Note: 'torq-compile' and 'docker' not found on system path. ==="
    echo "To utilize the Synaptics Astra SL2610 NPU, ensure you run 'torq-compile' manually"
    echo "on a machine with the Synaptics toolchain or Docker installed."
fi

echo "=== All models downloaded and configured successfully! ==="
