#!/usr/bin/env bash
# llama-optimus: install script (macOS / Linux)
# Installs llama-optimus from this repository into the current Python environment.
# Requires Python 3.10+ (https://www.python.org/downloads/).

set -u
cd "$(dirname "$0")"

echo
echo "=== llama-optimus installer ==="
echo

# --- check that a working Python 3.10+ is available ---
PYTHON=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 \
       && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        PYTHON="$cand"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3.10+ not found."
    echo "        Install Python 3.10+ (https://www.python.org/downloads/), e.g.:"
    echo "        macOS:  brew install python"
    echo "        Debian/Ubuntu:  sudo apt install python3 python3-pip"
    exit 1
fi

echo "Using Python:"
"$PYTHON" --version
echo

echo "Installing llama-optimus (this may take a minute)..."
if ! "$PYTHON" -m pip install --upgrade .; then
    echo
    echo "[ERROR] Installation failed. See the messages above."
    echo "Hint: on some systems you need --user or a virtualenv. Try:"
    echo "      $PYTHON -m pip install --user ."
    exit 1
fi

echo
echo "=== Installation complete ==="
echo
echo " - Command line :  llama-optimus --help"
echo " - GUI          :  llama-optimus-gui      (or run ./start_gui.sh)"
echo
echo " Tip: drop a llama.cpp build into ~/.llama-optimus/llama/bin"
echo "      and llama-optimus will find it automatically."
echo
echo " If the 'llama-optimus' command is not on your PATH, add the pip"
echo " scripts directory shown above to PATH, or run:"
echo "      $PYTHON -m llama_optimus.cli --help"
echo
