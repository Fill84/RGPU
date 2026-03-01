use tracing_subscriber::{fmt, EnvFilter};

/// Initialize structured logging with environment filter.
/// Set RGPU_LOG=debug (or trace, info, warn, error) for verbosity control.
/// Set RGPU_LOG_FILE to a file path to also write logs to a file (useful for services).
pub fn init_logging() {
    let filter = EnvFilter::try_from_env("RGPU_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // If RGPU_LOG_FILE is set, write logs to a file instead of stderr
    if let Ok(log_path) = std::env::var("RGPU_LOG_FILE") {
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = fmt()
                .with_env_filter(filter)
                .with_target(true)
                .with_thread_ids(true)
                .with_ansi(false)
                .with_writer(file)
                .try_init();
            return;
        }
    }

    let _ = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .try_init();
}
