use tracing_subscriber::{fmt, EnvFilter};

/// Initialize structured logging with environment filter.
/// Set RGPU_LOG=debug (or trace, info, warn, error) for verbosity control.
pub fn init_logging() {
    let filter = EnvFilter::try_from_env("RGPU_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .init();
}
