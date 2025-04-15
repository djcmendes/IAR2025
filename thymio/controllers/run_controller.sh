#!/usr/bin/env bash

# RUN WITH:
# steam-run bash run_controller.sh ./aula1/aula1.py

# run_controller.sh
set -e

# Configure Webots paths
export WEBOTS_HOME="/snap/webots/current/usr/share/webots"
export LD_LIBRARY_PATH="$WEBOTS_HOME/lib/controller:$LD_LIBRARY_PATH"

# Virtual environment path
# VENV_DIR="$(pwd)/.venv"

# Use direnv's environment if available
echo "Loading direnv environment..."
direnv allow .
eval "$(direnv export bash)"

# Create virtual environment if not exists
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