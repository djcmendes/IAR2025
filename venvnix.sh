# https://acalustra.com/optimizing-python-development-virtualenv-kernels-with-nix-and-jupyter.html
echo "Creating venv $1"
# VENV_FOLDER="$HOME/venvs/$1"
VENV_FOLDER=".venv"

if test -d "$VENV_FOLDER"; then
    echo "${VENV_FOLDER} is already created"
    nix shell github:GuillaumeDesforges/fix-python --command fix-python --venv "$VENV_FOLDER"
    exit 0
fi

echo "creating venv $1 on folder $VENV_FOLDER"
virtualenv "$VENV_FOLDER"

# Upgrade pip and install requirements in the new venv
"$VENV_FOLDER/bin/pip" install --upgrade pip
"$VENV_FOLDER/bin/pip" install -r requirements.txt

"$VENV_FOLDER/bin/pip" install ipykernel

nix shell github:GuillaumeDesforges/fix-python --command fix-python --venv "$VENV_FOLDER"
"$VENV_FOLDER/bin/python" -m ipykernel install --prefix="$HOME/.local/share" --name "venv-$1"
jupyter kernelspec install --user "$HOME/.local/share/share/jupyter/kernels/venv-$1"