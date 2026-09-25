#!/bin/bash
# ==========================================
# RLVR Step 1: OS-Level Sandbox Initialization
# ==========================================
# This script creates a secure, unprivileged user environment
# for the SubprocessGrader to execute untrusted LLM-generated code.
# Run this as root (e.g., sudo bash setup_sandbox.sh) before starting the MDP loop.

echo "[Step 1] Initializing RLVR Secure Sandbox..."

# 1. Create the unprivileged sandbox user (if it doesn't exist)
if id "sandboxuser" &>/dev/null; then
    echo "User 'sandboxuser' already exists."
else
    echo "Creating 'sandboxuser'..."
    useradd -m -s /bin/bash sandboxuser
fi

# 2. Setup the sandbox execution directory
SANDBOX_DIR="/tmp/rlvr_sandbox"
echo "Setting up sandbox directory at $SANDBOX_DIR..."
mkdir -p $SANDBOX_DIR

# 3. Restrict permissions
# Only sandboxuser can read/write inside this specific directory
chown -R sandboxuser:sandboxuser $SANDBOX_DIR
chmod -R 700 $SANDBOX_DIR

# 4. (Optional) Network isolation using iptables
# In a full production environment, we drop all outbound packets for sandboxuser
# to prevent the LLM from making network calls during code execution.
echo "Applying iptables network isolation for sandboxuser..."
iptables -A OUTPUT -m owner --uid-owner sandboxuser -j DROP || echo "Warning: iptables not available (normal in some containers)."

# 5. Verify Python availability for the sandbox user
echo "Verifying Python access for sandboxuser..."
sudo -u sandboxuser bash -c "python3 -c 'print(\"Sandbox Python is working!\")'"

echo "[Step 1] Complete! The execution environment is now secure."
echo "You can now run Phase 2 (MDP and GRPO) with use_sandbox=True."
