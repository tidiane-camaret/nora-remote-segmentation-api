#!/bin/bash

# ==============================================================================
#  Segmentation Server start script
# ==============================================================================
#
#  This script starts the segmentation server.
#  Submit it to a queue using: sbatch [options] start.sh
#
#  It assumes that the environment has already been set up using 'setup.sh'.
#
#  Requires Nvidia GPU. 10GB of VRAM are recommended for the current version for the model.
#
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# Find the script's own directory to make it runnable from anywhere
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$DIR/.venv"
PYTHON_EXEC="$VENV_DIR/bin/python"

echo "--- Starting Segmentation Server script ---"
echo "Running on node: $(hostname)"
echo "Project Directory: $DIR"

# --- Environment Validation ---
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $PYTHON_EXEC."
    echo "Please run the 'setup.sh' script on a login node first."
    exit 1
fi

# --- Activate and Launch ---
echo "Activating Python environment..."
source "$VENV_DIR/bin/activate"

# Set MOCK_MODE to 0 to use the real model and GPU
export MOCK_MODE="0"
echo "MOCK_MODE is set to $MOCK_MODE"

echo "Launching FastAPI server..."
cd "$DIR"
# Use python -u for unbuffered output, which is better for logging
$PYTHON_EXEC -u "main.py"

echo "--- Segmentation server has shut down. ---"