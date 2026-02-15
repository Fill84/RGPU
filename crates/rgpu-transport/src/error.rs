#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TLS error: {0}")]
    Tls(#[from] rustls::Error),

    #[error("wire format error: {0}")]
    Wire(#[from] rgpu_protocol::wire::WireError),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("QUIC error: {0}")]
    Quic(String),

    #[error("connection closed")]
    ConnectionClosed,

    #[error("authentication failed: {0}")]
    AuthFailed(String),

    #[error("timeout")]
    Timeout,

    #[error("server not found: {0}")]
    ServerNotFound(String),
}
