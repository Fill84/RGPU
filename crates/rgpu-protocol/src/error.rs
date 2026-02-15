use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ProtocolError {
    #[error("connection failed: {0}")]
    ConnectionFailed(String),

    #[error("authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("invalid handle: {0}")]
    InvalidHandle(String),

    #[error("GPU error: code={code}, message={message}")]
    GpuError { code: i32, message: String },

    #[error("unsupported command: {0}")]
    UnsupportedCommand(String),

    #[error("serialization error: {0}")]
    SerializationError(String),

    #[error("timeout")]
    Timeout,

    #[error("server disconnected")]
    Disconnected,

    #[error("out of memory: requested {requested} bytes")]
    OutOfMemory { requested: u64 },

    #[error("not implemented: {0}")]
    NotImplemented(String),
}
