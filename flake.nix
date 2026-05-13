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
        pkgs = import nixpkgs { inherit system overlays; };
        rustToolchain = pkgs.rust-bin.stable.latest.default;
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

          env = {
            OPENBLAS_NUM_THREADS = "1";
          };

          # Tests require a GPU/dataset, skip them
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
            echo "tternary dev shell"
            echo "rustc: $(rustc --version)"
            echo "cargo: $(cargo --version)"
          '';
        };
      }
    );
}
