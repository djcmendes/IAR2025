# https://github.com/nix-community/dream2nix/tree/main

{
  inputs = {
    #nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    nixpkgs.url = "github:NixOS/nixpkgs/d5e44742e9c935f215995850169944da7dd50b75";
    systems.url = "github:nix-systems/default";
  };

  outputs = { systems, nixpkgs, ... } @ inputs:
  let
    eachSystem = f:
      nixpkgs.lib.genAttrs (import systems) (
        system:
          let
            # Configure pkgs for this specific system
            pkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true; # <<< IMPORTANT for NVIDIA drivers/CUDA
                # cudaSupport = true; # <<< IMPORTANT for CUDA-enabled packages
                # You might need to specify a cudaMajorVersion if default is not what you need
                # cudaMajorVersion = "11"; # or "12" - check what your TF version expects
              };
            };
          in f pkgs
      );
  in {
    devShells = eachSystem (pkgs: 
    let

    in {
      default = pkgs.mkShell {
        packages = [
          pkgs.texlive.combined.scheme-full  # Full TeX Live installation
          #pkgs.jupyter-all
          # pkgs.cudaPackages.cudatoolkit
          # (pkgs.python312.withPackages (python-pkgs: with python-pkgs; [
          #   # tensorflowWithCuda
          #   tensorflow
          #   tensorboard
          #   protobuf
          #   pandas
          #   numpy
          #   jupyter
          #   matplotlib
          #   sympy
          #   virtualenv
          #   gymnasium
          #   stable-baselines3
          #   #torch
          # ]))
          pkgs.virtualenv
          # pkgs.steam-run
        ];

        shellHook = ''
          export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
          # echo "Development shell ready!"
          # echo "Run with > jupyter notebook"

          # jupyter notebook

          #echo "TeX Live full environment is ready!"
          #echo "Build with \`latexmk --shell-escape -f -synctex=1 -interaction=nonstopmode -file-line-error -pdf ./main\`"

          sh venvnix.sh 
          source .venv/bin/activate
        '';
      };
    });
  };
}
