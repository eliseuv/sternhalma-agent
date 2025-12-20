{
  description = "Python development environment with libraries";

  nixConfig = {
    extra-substituters = [
      "https://cache.nixos.org"
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.systems.follows = "systems";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let

        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        # Python script to test PyTorch CUDA support
        testTorchCudaScript = pkgs.writeText ".test_torch_cuda.py" ''
          import torch
          print('CUDA support:', torch.cuda.is_available())
          print('CUDA devices count:', torch.cuda.device_count())
          print('CUDA devices name:', torch.cuda.get_device_name(0))
        '';

      in
      {
        # Python environment with packages
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python3

            ruff
            basedpyright
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                # Add other libraries if needed, e.g. glib, libGL
              ]
            }:$LD_LIBRARY_PATH
            export UV_PYTHON_DOWNLOADS=never
            uv sync
            source .venv/bin/activate
            python ${testTorchCudaScript}
          '';

        };
      }
    );

}
