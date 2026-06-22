#!/bin/bash
set -e

# Create models directory if it doesn't exist
mkdir -p models

echo "=== Downloading Audio Models for Coral Recorder ==="

# 1. YAMNet (Audio Event Classification and Speech Detection)
if [ ! -f "models/yamnet.tflite" ]; then
    echo "Downloading YAMNet (CPU TFLite)..."
    curl -L -o models/yamnet.tflite "https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite"
else
    echo "YAMNet model already exists. Skipping..."
fi

# 2. Speech Commands (Wake Word / Commands Model) - CPU
if [ ! -f "models/voice_commands.tflite" ]; then
    echo "Downloading Speech Commands (CPU TFLite)..."
    curl -L -o models/conv_actions_tflite.zip "https://storage.googleapis.com/download.tensorflow.org/models/tflite/conv_actions_tflite.zip"
    unzip -q models/conv_actions_tflite.zip -d models/
    mv models/conv_actions_frozen.tflite models/voice_commands.tflite
    rm -f models/conv_actions_tflite.zip models/conv_actions_labels.txt
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

# 5. NPU Compilation Compatibility Note for Synaptics Astra SL2610
echo ""
echo "=== Synaptics Astra SL2610 NPU Compilation Status ==="
echo "Note: The downloaded models contain signal pre-processing operations:"
echo "  - YAMNet utilizes an RFFT (Real Fast Fourier Transform) resulting in complex numbers (complex<f32>),"
echo "    which is not supported by the TOSA dialect standard."
echo "  - Speech Commands utilizes custom TFLite operators ('AudioSpectrogram' and 'Mfcc') which are not"
echo "    legalized to TOSA/NPU hardware instructions."
echo ""
echo "Therefore, these models cannot be compiled to .vmfb NPU format and will run on the board's"
echo "ARM CPU using Google's highly optimized 'ai-edge-litert' runtime (or Edge TPU delegate for Coral)."
echo "====================================================="
echo ""

echo "=== All models downloaded and configured successfully! ==="
