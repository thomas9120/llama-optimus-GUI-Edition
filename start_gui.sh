#!/usr/bin/env bash
# llama-optimus: GUI launcher (macOS / Linux)
# Runs llama-optimus-gui. If you just cloned the repo, run install.sh first.

# --- prefer the installed entry point ---
if command -v llama-optimus-gui >/dev/null 2>&1; then
    exec llama-optimus-gui
fi

# --- fall back to running the module with the first usable python ---
for PYTHON in python3 python; do
    if command -v "$PYTHON" >/dev/null 2>&1 && "$PYTHON" -c "import llama_optimus.gui" 2>/dev/null; then
        exec "$PYTHON" -m llama_optimus.gui
    fi
done

echo "[ERROR] Could not launch the llama-optimus GUI."
echo
echo "Is llama-optimus installed? Run ./install.sh first (requires Python 3.10+)."
echo "If it is installed but fails with 'No module named tkinter', install tkinter:"
echo "        macOS:          brew install python-tk"
echo "        Debian/Ubuntu:  sudo apt install python3-tk"
echo "Once installed, the GUI is also available as the 'llama-optimus-gui' command."
exit 1
