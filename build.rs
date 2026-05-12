fn main() {
    println!("cargo:rustc-link-search=/usr/lib/x86_64-linux-gnu/openblas-pthread");
    println!("cargo:rustc-link-lib=dylib=openblas");
}
