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
      pkgs = nixpkgs.legacyPackages."x86_64-linux";
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {

        packages = [
          (pkgs.python3.withPackages (
            pypkgs: with pypkgs; [
              numpy
            ]
          ))
        ];

      };

    };

}
