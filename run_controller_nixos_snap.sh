#!/usr/bin/env bash

# RUN WITH:
# steam-run bash run_controller_nixos_snap.sh ./thymio/controllers/p1_controller_simple/p1_controller_simple.py

set -e

# Default port
PORT=1234
CONTROLLER_FILE=""

# Processar argumentos
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        *)
            CONTROLLER_FILE="$1"
            shift
            ;;
    esac
done

if [ -z "$CONTROLLER_FILE" ]; then
    echo "ERRO: Arquivo do controlador não especificado!"
    exit 1
fi

# Configurar paths do Webots
export WEBOTS_HOME="/snap/webots/current/usr/share/webots"
export LD_LIBRARY_PATH="$WEBOTS_HOME/lib/controller:$LD_LIBRARY_PATH"

# Usar ambiente direnv
echo "Loading direnv environment..."
direnv allow .
eval "$(direnv export bash)"

VENV_DIR="$(pwd)/.venv"
source "$VENV_DIR/bin/activate"

# Executar controlador
echo "Executando controlador: $CONTROLLER_FILE na porta $PORT"
$WEBOTS_HOME/webots-controller --port="$PORT" "$CONTROLLER_FILE"