{
  description = "Python development environment with libraries";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {

        packages = with pkgs; [

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
              pytorch
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

    };

}
