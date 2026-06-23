#!/bin/bash
set -e

RULE_FILE="/etc/udev/rules.d/99-coral-network.rules"
INTERFACE_NAME="coralusb"
IP_ADDR="192.168.137.1/24"
BOARD_IP="192.168.137.2"
CON_NAME="CoralUSB"

echo "=== Coral Recorder - Permanent Network Configuration ==="

# 1. Write the udev rule to match the Coral board and give it a persistent interface name
echo "Creating udev rule in ${RULE_FILE}..."
sudo bash -c "cat > ${RULE_FILE}" <<EOF
# Udev rule for Synaptics Coral Dev Board USB Ethernet Gadget
SUBSYSTEM=="net", ACTION=="add", ATTRS{idVendor}=="1d6b", ATTRS{idProduct}=="0104", ATTRS{serial}=="grinn-astra-2619-coral", NAME="${INTERFACE_NAME}"
EOF

# 2. Find any active dynamic enx* interface and bring it down so udev can rename it
OLD_INTF=$(ip link show | grep -o -E 'enx[0-9a-f]{12}' | head -n 1)
if [ -n "$OLD_INTF" ] && [ "$OLD_INTF" != "$INTERFACE_NAME" ]; then
    echo "Found interface ${OLD_INTF}. Bringing it down to allow renaming..."
    sudo ip link set dev "${OLD_INTF}" down || true
fi

# 3. Reload udev rules and trigger the change
echo "Reloading udev rules and triggering device renaming..."
sudo udevadm control --reload-rules
sudo udevadm trigger --attr-match=subsystem=net

# Wait a moment for the interface renaming to complete
sleep 2

# 4. Check if the interface is renamed successfully
if ip link show "${INTERFACE_NAME}" > /dev/null 2>&1; then
    echo "Success: Interface renamed to ${INTERFACE_NAME}!"
else
    echo "Error: Failed to rename interface to ${INTERFACE_NAME}."
    echo "Please disconnect the Coral board, reconnect it, and run again."
    exit 1
fi

# 5. Create or recreate the NetworkManager profile for the renamed interface
echo "Configuring NetworkManager connection '${CON_NAME}'..."
if nmcli connection show "${CON_NAME}" > /dev/null 2>&1; then
    echo "Removing existing NetworkManager connection '${CON_NAME}'..."
    sudo nmcli connection delete "${CON_NAME}"
fi

echo "Adding new NetworkManager connection '${CON_NAME}' with static IP ${IP_ADDR}..."
sudo nmcli connection add type ethernet con-name "${CON_NAME}" ifname "${INTERFACE_NAME}" ip4 "${IP_ADDR}" ipv4.method manual

# 6. Bring up the network connection
echo "Activating network connection..."
sudo nmcli connection up "${CON_NAME}"

# 7. Test connectivity to the board
echo "Testing connection to Coral Board (${BOARD_IP})..."
if ping -c 3 "${BOARD_IP}" > /dev/null 2>&1; then
    echo "=== Configuration Successful! ==="
    echo "PC Interface: ${INTERFACE_NAME} (IP: ${IP_ADDR})"
    echo "Coral Board: ${BOARD_IP} is reachable!"
else
    echo "Warning: Configuration applied but Coral Board (${BOARD_IP}) did not respond to ping yet."
    echo "Make sure the board is powered on, fully booted, and try running: ping ${BOARD_IP}"
fi
