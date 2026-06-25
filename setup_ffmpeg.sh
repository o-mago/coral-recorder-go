#!/bin/bash
# Script to download a static build of FFmpeg for ARM64 and deploy it to the Coral Board.
# This enables audio compression (WAV -> MP3) to reduce file sizes before uploading to Google Drive.

set -e

echo "=== Downloading and deploying static FFmpeg (ARM64) to Coral Board ==="

# Create a temporary directory inside the workspace
TEMP_DIR=$(mktemp -d -p .)
cd "$TEMP_DIR"

echo "Downloading FFmpeg static build (approx 20MB)..."
wget -q --show-progress https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz

echo "Extracting..."
tar -xf ffmpeg-release-arm64-static.tar.xz

FFMPEG_PATH=$(find . -name ffmpeg -type f)

if [ -z "$FFMPEG_PATH" ]; then
    echo "Error: Could not find ffmpeg binary in the extracted archive."
    cd ..
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "Pushing ffmpeg to the Coral Board..."
# Push to /usr/bin/ (since we are root on adb shell, this is perfect)
adb push "$FFMPEG_PATH" /usr/bin/ffmpeg

echo "Setting permissions..."
adb shell "chmod +x /usr/bin/ffmpeg"

cd ..
rm -rf "$TEMP_DIR"

echo ""
echo "=== FFmpeg deployed successfully to Coral Board! ==="
adb shell "ffmpeg -version | head -n 1"
