#!/usr/bin/env bash

# RUN WITH:
# steam-run bash run_controller_nixos_snap.sh ./thymio/controllers/p1_controller_simple/p1_controller_simple.py

set -e

# Configure Webots paths
export WEBOTS_HOME="/snap/webots/current/usr/share/webots"
export LD_LIBRARY_PATH="$WEBOTS_HOME/lib/controller:$LD_LIBRARY_PATH"

# Use direnv's environment
echo "Loading direnv environment..."
direnv allow .
eval "$(direnv export bash)"

VENV_DIR="$(pwd)/.venv"
source "$VENV_DIR/bin/activate"

# Create virtual environment
# VENV_DIR="$(pwd)/.venv"
#if [ ! -d "$VENV_DIR" ]; then
#    echo "Creating virtual environment..."
#    python -m venv "$VENV_DIR"
#    source "$VENV_DIR/bin/activate"
#    pip install --upgrade pip
#    pip install -r requirements.txt
#else
#    source "$VENV_DIR/bin/activate"
#fi

# Run controller script with Webots arguments
$WEBOTS_HOME/webots-controller "$@"