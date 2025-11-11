{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { flake-utils, nixpkgs, ... }:
  flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      diffjpeg = pkgs.python3Packages.buildPythonPackage {
        pname = "DiffJPEG";
        version = "0.1";

        src = pkgs.fetchFromGitHub {
          owner = "necla-ml";
          repo = "Diff-JPEG";
          rev = "e81f082896ba145e35cc129bc7743244e10881e5";
          sha256 = "sha256-2fiifHDJFhduZIGj/Y320+DKJYjrj9umuvEf4VtlobA=";
        };

        propagatedBuildInputs = with pkgs.python3Packages; [
          kornia
          numpy
        ];
      };
    in
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python3
          python3Packages.numpy
          python3Packages.pillow
          python3Packages.torchWithCuda
          python3Packages.torchvision

          diffjpeg
        ];
      };
    }
  );
}
