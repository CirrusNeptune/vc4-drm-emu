fn main() {
    cc::Build::new().file("inject.c").compile("inject");
    println!("cargo:rerun-if-changed=inject.c");
    for sym in [
        "_ioctl_inject",
        "_open_inject",
        "_close_inject",
        "_mmap_inject",
        "_munmap_inject",
    ] {
        println!("cargo:rustc-link-arg=-Wl,-exported_symbol");
        println!("cargo:rustc-link-arg=-Wl,{sym}");
    }
}
