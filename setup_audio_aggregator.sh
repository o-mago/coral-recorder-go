#!/bin/bash
# Script to setup/restore a virtual audio aggregator device on Linux using PipeWire/PulseAudio.
# This device will combine the microphone input and desktop/system output monitor,
# allowing the coral-recorder to record both sides of a meeting/call.

set -e

SINK_NAME="aggregator"
SINK_DESC="Agregador_de_Audio"

echo "=== System Audio Aggregator Configuration ==="

# 1. Cleanup existing aggregator modules if any (prevents duplicates)
echo "Checking and cleaning up any existing aggregator modules..."

pactl list modules | awk '/Module #/{id=$2} /Name: module-loopback/{is_loopback=1} /Argument:.*sink=aggregator/{if(is_loopback) print id; is_loopback=0} !/Module #/ && !/Name:/ {is_loopback=0}' | while read -r mod_id; do
    mod_id=${mod_id#\#}
    echo "  - Unloading old loopback module #$mod_id"
    pactl unload-module "$mod_id" || true
done

pactl list modules | awk '/Module #/{id=$2} /Name: module-null-sink/{is_null_sink=1} /Argument:.*sink_name=aggregator/{if(is_null_sink) print id; is_null_sink=0} !/Module #/ && !/Name:/ {is_null_sink=0}' | while read -r mod_id; do
    mod_id=${mod_id#\#}
    echo "  - Unloading old null-sink module #$mod_id"
    pactl unload-module "$mod_id" || true
done

# Give PipeWire/PulseAudio a short moment to update defaults after unloading
sleep 0.5

# 2. Detect default sink and source
DEFAULT_SINK=$(pactl get-default-sink)
DEFAULT_SOURCE=$(pactl get-default-source)

# If default source is aggregator or empty, find a physical capture device
if [[ "$DEFAULT_SOURCE" == *"aggregator"* ]] || [ -z "$DEFAULT_SOURCE" ]; then
    # Try to find a USB input source
    DEFAULT_SOURCE=$(pactl list sources short | awk '{print $2}' | grep "alsa_input" | grep -i "usb" | head -n 1)
    if [ -z "$DEFAULT_SOURCE" ]; then
        # Fallback to any non-aggregator input source
        DEFAULT_SOURCE=$(pactl list sources short | awk '{print $2}' | grep "alsa_input" | grep -v "aggregator" | head -n 1)
    fi
fi

# If default sink is aggregator or empty, find a physical playback device
if [[ "$DEFAULT_SINK" == *"aggregator"* ]] || [ -z "$DEFAULT_SINK" ]; then
    # Try to find a USB output sink
    DEFAULT_SINK=$(pactl list sinks short | awk '{print $2}' | grep "alsa_output" | grep -i "usb" | head -n 1)
    if [ -z "$DEFAULT_SINK" ]; then
        # Fallback to any non-aggregator output sink
        DEFAULT_SINK=$(pactl list sinks short | awk '{print $2}' | grep "alsa_output" | grep -v "aggregator" | head -n 1)
    fi
fi

if [ -z "$DEFAULT_SINK" ] || [ -z "$DEFAULT_SOURCE" ]; then
    echo "Error: Could not detect default sink or source via pactl."
    exit 1
fi

echo "Detected default playback sink:   $DEFAULT_SINK"
echo "Detected default capture source:  $DEFAULT_SOURCE"

# 3. Load Null Sink
echo "Creating Null Sink '${SINK_NAME}'..."
NULL_SINK_ID=$(pactl load-module module-null-sink sink_name="$SINK_NAME" sink_properties=device.description="$SINK_DESC")
echo "  - Null Sink loaded with ID: $NULL_SINK_ID"

# 4. Load Loopbacks (Left: Mic, Right: Desktop)
echo "Routing microphone ($DEFAULT_SOURCE) to left channel of aggregator..."
MIC_LOOPBACK_ID=$(pactl load-module module-loopback source="$DEFAULT_SOURCE" sink="$SINK_NAME" channel_map=front-left)
echo "  - Microphone loopback loaded with ID: $MIC_LOOPBACK_ID"

echo "Routing desktop audio (${DEFAULT_SINK}.monitor) to right channel of aggregator..."
DESKTOP_LOOPBACK_ID=$(pactl load-module module-loopback source="${DEFAULT_SINK}.monitor" sink="$SINK_NAME" channel_map=front-right)
echo "  - Desktop audio loopback loaded with ID: $DESKTOP_LOOPBACK_ID"

# 5. Set default source to aggregator's monitor
echo "Setting default audio capture source to 'aggregator.monitor'..."
pactl set-default-source "${SINK_NAME}.monitor"

echo ""
echo "=== Configuration Successful! ==="
echo "The virtual audio aggregator has been successfully configured and activated."
echo "Default source is now: aggregator.monitor"
echo "You can start the client and it will automatically capture both microphone and desktop audio."
echo "To check if it's listed, run: cd client && go run . -list"
echo "To tear down this virtual device, run:"
echo "  pactl unload-module $MIC_LOOPBACK_ID"
echo "  pactl unload-module $DESKTOP_LOOPBACK_ID"
echo "  pactl unload-module $NULL_SINK_ID"
