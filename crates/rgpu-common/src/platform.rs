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
