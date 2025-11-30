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

      python = pkgs.python3.override {
        packageOverrides = self: super: {
          torch = super.torch.override {
            cudaSupport = true;
          };
        };
      };

      diffjpeg = python.pkgs.buildPythonPackage {
        pname = "DiffJPEG";
        version = "0.1";

        src = pkgs.fetchFromGitHub {
          owner = "necla-ml";
          repo = "Diff-JPEG";
          rev = "e81f082896ba145e35cc129bc7743244e10881e5";
          sha256 = "sha256-2fiifHDJFhduZIGj/Y320+DKJYjrj9umuvEf4VtlobA=";
        };

        propagatedBuildInputs = with python.pkgs; [
          kornia
          numpy
        ];
      };

      lpips = python.pkgs.buildPythonPackage {
        pname = "lpips";
        version = "0.1.4";

        src = pkgs.fetchFromGitHub {
          owner = "richzhang";
          repo = "PerceptualSimilarity";
          rev = "082bb24f84c091ea94de2867d34c4544f68e0963";
          sha256 = "sha256-EYpb/toYLz9w6vzIq4M40Q2DH6uHcDSB5q8PSv/7PM8=";
        };
      };
    in
    {
      devShells.default = pkgs.mkShell {
        packages = [
          (python.withPackages (ps: with ps; [
            datasets
            diffjpeg
            lpips
            matplotlib
            numpy
            pillow
            pytorch-msssim
            torch
            torchvision
            tqdm
          ]))
        ];
      };
    }
  );
}
