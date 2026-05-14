{
  description = "Ternary neural network implementation in Rust";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
        lib = pkgs.lib;
        rustToolchain = pkgs.rust-bin.stable.latest.default;

        cudaPkgs = pkgs.cudaPackages_12;

        cudaLibs = with cudaPkgs; [
          cuda_nvcc
          cuda_cudart
          cuda_nvrtc
          libcublas
          cuda_nvtx
        ];

        cudaToolkit = pkgs.symlinkJoin {
          name = "cuda-toolkit";
          paths = cudaLibs ++ builtins.map (p: lib.getLib p) cudaLibs;
        };

        cudaLibPath = lib.concatStringsSep ":" (
          builtins.map (p: "${lib.getLib p}/lib") cudaLibs
        );

      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "tternary";
          version = "0.2.0";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            rustToolchain
            pkg-config
          ];

          buildInputs = with pkgs; [
            openssl
            openblas
          ];

          env.OPENBLAS_NUM_THREADS = "1";
          doCheck = false;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustToolchain
            pkg-config
            openssl
            openblas
            cargo-watch
            cargo-edit
          ];

          env.OPENBLAS_NUM_THREADS = "1";

          shellHook = ''
            echo "tternary dev shell (CPU/NdArray)"
            echo "rustc: $(rustc --version)"
            echo "cargo: $(cargo --version)"
          '';
        };

        devShells.cuda = pkgs.mkShell {
          packages = with pkgs; [
            rustToolchain
            pkg-config
            openssl
            openblas
            cargo-watch
            cargo-edit
          ] ++ cudaLibs;

          env = {
            OPENBLAS_NUM_THREADS = "1";

            CUDA_PATH = "${cudaToolkit}";

            CUDARC_CUDA_VERSION = "12000";

            CUDA_VISIBLE_DEVICES = "0";

            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.openssl
              pkgs.openblas
            ] + ":" + cudaLibPath + ":/run/opengl-driver/lib";
          };

          shellHook = ''
            echo "tternary dev shell (CUDA)"
            echo "rustc: $(rustc --version)"
            echo "cargo: $(cargo --version)"
            echo "CUDA: device 0 = RTX 2060 (sm_75), device 1 = Tesla P100 (sm_60)"
            echo ""
            echo "Usage:"
            echo "  cargo r -r --features cuda -- --steps 500"
            echo "  CUDA_VISIBLE_DEVICES=1 cargo r -r --features cuda -- --steps 500  # P100"
          '';
        };
      }
    );
}
