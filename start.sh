#!/bin/bash

# ==============================================================================
#  Segmentation Server start script with Auto-Restart
# ==============================================================================
#
#  This script starts the segmentation server with automatic restart on crash,
#  including CUDA out-of-memory errors. GPU memory is cleared before each restart.
#
#  Usage:
#    ./start.sh                                        # Auto-restart enabled, no log file
#    AUTO_RESTART=0 ./start.sh                         # Disable auto-restart
#    MAX_RESTARTS=10 ./start.sh                        # Limit restart attempts
#    LOG_DIR=/path/to/logs ./start.sh                  # Enable file logging
#    CACHE_DIR=/path/to/cache ./start.sh               # Enable persistent cache
#
#  Environment variables:
#    AUTO_RESTART=1       Enable auto-restart on crash (default: 1)
#    MAX_RESTARTS=-1      Max restart attempts, -1 = unlimited (default: -1)
#    RESTART_DELAY=5      Seconds between restarts (default: 5)
#    LOG_DIR=""           Directory for log files. If empty, no file logging (default: "")
#    CACHE_DIR="none"     Directory for persistent cache. Use 'none' for RAM-only (default: "none")
#
#  Requires: Python environment from setup.sh, Nvidia GPU (10GB+ VRAM)
#
# ==============================================================================

# Don't exit on error - we handle crashes ourselves
set +e

# Find the script's own directory to make it runnable from anywhere
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$DIR/.venv"
PYTHON_EXEC="$VENV_DIR/bin/python"

# --- Configuration ---
AUTO_RESTART=${AUTO_RESTART:-1}
MAX_RESTARTS=${MAX_RESTARTS:--1}
RESTART_DELAY=${RESTART_DELAY:-5}
LOG_DIR=${LOG_DIR:-""}
CACHE_DIR=${CACHE_DIR:-"none"}

echo "--- Starting Segmentation Server script ---"
echo "Running on node: $(hostname)"
echo "Project Directory: $DIR"
echo "Auto-restart: $([ $AUTO_RESTART -eq 1 ] && echo 'enabled' || echo 'disabled')"
if [ $AUTO_RESTART -eq 1 ]; then
    echo "Max restarts: $([ $MAX_RESTARTS -eq -1 ] && echo 'unlimited' || echo $MAX_RESTARTS)"
    echo "Restart delay: ${RESTART_DELAY}s"
fi

# --- Environment Validation ---
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "Error: Python executable not found at $PYTHON_EXEC."
    echo "Please run the 'setup.sh' script on a login node first."
    exit 1
fi

# --- GPU Cleanup Function ---
clear_gpu_memory() {
    echo "[$(date '+%H:%M:%S')] Clearing GPU memory..."
    # Kill any zombie Python processes from this server
    pkill -9 -f "python.*main.py" 2>/dev/null || true
    sleep 2
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | \
            awk -F', ' '{printf "  GPU Memory: %s / %s\n", $1, $2}'
    fi
}

# --- Activate Environment ---
echo "Activating Python environment..."
source "$VENV_DIR/bin/activate"

# Set MOCK_MODE to 0 to use the real model and GPU
export MOCK_MODE="0"
echo "MOCK_MODE is set to $MOCK_MODE"

# --- Main Server Loop ---
restart_count=0

while true; do
    # Check max restarts limit
    if [ $MAX_RESTARTS -ne -1 ] && [ $restart_count -ge $MAX_RESTARTS ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: Max restart attempts ($MAX_RESTARTS) reached. Exiting."
        exit 1
    fi

    # Clear GPU on restarts (but not on first start)
    if [ $restart_count -gt 0 ]; then
        clear_gpu_memory
    fi

    # Launch server
    if [ $restart_count -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Launching FastAPI server..."
    else
        echo "[$(date '+%H:%M:%S')] Restarting FastAPI server (attempt $restart_count)..."
    fi

    cd "$DIR"
    start_time=$(date +%s)

    # Build command with optional arguments
    cmd="$PYTHON_EXEC -u main.py"
    if [ -n "$LOG_DIR" ]; then
        cmd="$cmd --log-dir $LOG_DIR"
    fi
    cmd="$cmd --cache-dir $CACHE_DIR"

    # Execute the command
    eval $cmd
    exit_code=$?

    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    # Check exit status
    if [ $exit_code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Server exited cleanly. Shutting down."
        exit 0
    fi

    # Server crashed
    echo "========================================="
    echo "[$(date '+%H:%M:%S')] SERVER CRASHED!"
    echo "  Exit code: $exit_code"
    echo "  Runtime: ${runtime}s"
    echo "========================================="

    # Check if auto-restart is disabled
    if [ $AUTO_RESTART -ne 1 ]; then
        echo "Auto-restart disabled. Exiting."
        exit $exit_code
    fi

    # Increase delay if crashed quickly
    if [ $runtime -lt 10 ]; then
        actual_delay=$((RESTART_DELAY * 2))
        echo "WARNING: Quick crash detected. Using ${actual_delay}s delay."
    else
        actual_delay=$RESTART_DELAY
    fi

    echo "Waiting ${actual_delay}s before restart..."
    sleep $actual_delay

    restart_count=$((restart_count + 1))
done