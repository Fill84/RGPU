#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("handle not found: {0}")]
    HandleNotFound(String),

    #[error("configuration error: {0}")]
    ConfigError(String),

    #[error("IPC error: {0}")]
    IpcError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
