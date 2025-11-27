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
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [

            ruff
            basedpyright

            # Python
            (python3.withPackages (
              pypkgs: with pypkgs; [
                # CBOR
                cbor2
                # Numerical
                numpy
                scipy
                # Machine Learning
                scikit-learn
                pytorch-bin
              ]
            ))

            # CUDA
            # cudatoolkit
            # cudaPackages.cudnn

          ];

          # shellHook = ''
          #   export CUDA_PATH=${pkgs.cudatoolkit}
          #   export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
          #   export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
          #   export EXTRA_CCFLAGS="-I/usr/include"
          # '';

        };
      }
    );

}
