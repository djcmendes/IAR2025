# IAR2025
Inteligência Artificial para a Robótica

# Make virtual environment python

- If in NixOS, you should have flakes, activate the environment (`nix develop` or `direnv allow`) and run `sh venvnix.sh` e `source .venv/bin/activate`.

1. python -m venv .venv --copies
2. Source the environment:
   1. **Linux:** source .venv/bin/activate
   2. **Windows CMD** .venv\Scripts\activate
   3. **Windows Powershell:** .venv\Scripts\Activate.ps1
3. pip install --upgrade pip
4. pip install -r requirements.txt