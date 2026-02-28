/// Returns the default IPC socket/pipe path for client daemon communication.
pub fn default_ipc_path() -> String {
    #[cfg(unix)]
    {
        let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
            .unwrap_or_else(|_| "/tmp".to_string());
        format!("{}/rgpu.sock", runtime_dir)
    }
    #[cfg(windows)]
    {
        r"\\.\pipe\rgpu".to_string()
    }
}

/// Resolve the IPC address for interpose libraries to connect to the daemon.
/// Checks `RGPU_IPC_ADDRESS` env var first (for Docker/container support),
/// then falls back to the default local IPC path.
///
/// If `RGPU_IPC_ADDRESS` contains a `host:port` pattern, TCP is used.
/// Otherwise, Unix sockets (Linux/macOS) or named pipes (Windows) are used.
pub fn resolve_ipc_address() -> String {
    if let Ok(addr) = std::env::var("RGPU_IPC_ADDRESS") {
        if !addr.is_empty() {
            return addr;
        }
    }
    default_ipc_path()
}

/// Returns true if the given IPC address is a TCP address (host:port format).
pub fn is_tcp_address(addr: &str) -> bool {
    // TCP addresses contain ':' but are NOT Windows named pipes (\\.\pipe\...)
    // and NOT Unix socket paths (starting with /)
    addr.contains(':') && !addr.starts_with(r"\\") && !addr.starts_with('/')
}

/// Returns the platform name string.
pub fn platform_name() -> &'static str {
    #[cfg(target_os = "windows")]
    { "windows" }
    #[cfg(target_os = "linux")]
    { "linux" }
    #[cfg(target_os = "macos")]
    { "macos" }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    { "unknown" }
}
