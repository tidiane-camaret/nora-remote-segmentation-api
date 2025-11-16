#!/bin/bash

# ==============================================================================
#  Segmentation Server Environment Setup
# ==============================================================================
#
#  This script prepares the Python environment for the segmentation server.
#  It should be run once before starting the server for the first time.
#
#  It will:
#  1. Check for Python 3.10+.
#  2. Create a virtual environment in './.venv'.
#  3. Install all required packages from 'pyproject.toml'.
#
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the script's own directory to make it runnable from anywhere
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$DIR/.venv"

echo "--- Starting Segmentation Server Setup ---"
echo "Project directory: $DIR"

# --- Prerequisite Check ---
if ! command -v python3 &> /dev/null || ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' &> /dev/null; then
    echo "Error: Python 3.10 or newer is required but not found."
    echo "Please load a Python 3.10+ module before running this script."
    exit 1
fi
echo "Python 3.10+ found."

# --- Create Virtual Environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists. Skipping creation."
fi

# --- Install Dependencies ---
echo "Activating environment and installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
# Install the project in editable mode, including all dependencies
pip install -e "$DIR"


# --- Download Model Weights ---
echo "Downloading model weights..."
cd "$DIR"
"$VENV_DIR/bin/python" "src/utils.py"
deactivate

echo ""
echo "Setup complete!"
echo "You can now launch the segmentation server (e.g. from a job or a terminal) using the start.sh script."
