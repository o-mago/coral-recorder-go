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

echo "=== All models downloaded and configured successfully! ==="
