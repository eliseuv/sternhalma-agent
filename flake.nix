{
  description = "Python env";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      flake-utils,
      nixpkgs,
      ...
    }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {

        packages = with pkgs; [

          # CUDA
          # cudatoolkit
          # cudaPackages.cudnn

          # Python
          (python3.withPackages (
            pypkgs: with pypkgs; [
              # Numerical
              numpy
              scipy
              # Machine Learning
              scikit-learn
              pytorch
            ]
          ))
        ];

        # shellHook = ''
        #   export CUDA_PATH=${pkgs.cudatoolkit}
        #   export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
        #   export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        #   export EXTRA_CCFLAGS="-I/usr/include"
        # '';

      };

    };

}
