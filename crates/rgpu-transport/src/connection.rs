use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_rustls::client::TlsStream as ClientTlsStream;
use tokio_rustls::server::TlsStream as ServerTlsStream;
use tracing::{debug, error};

use rgpu_protocol::messages::{Message, RequestId};
use rgpu_protocol::wire::{self, HEADER_SIZE};

use crate::error::TransportError;

/// Whether this side of the connection is the server or client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionRole {
    Server,
    Client,
}

/// An established, authenticated connection to a remote RGPU peer.
/// Handles framing, sending, and receiving messages.
pub struct RgpuConnection {
    role: ConnectionRole,
    /// Sender half for outgoing messages
    tx: mpsc::Sender<Vec<u8>>,
    /// Receiver for incoming messages (consumed by the message loop)
    rx: Arc<Mutex<mpsc::Receiver<Message>>>,
    /// Request ID counter
    next_request_id: AtomicU64,
    /// Pending responses: request_id -> oneshot sender
    pending: Arc<dashmap::DashMap<u64, oneshot::Sender<Message>>>,
}

impl RgpuConnection {
    /// Create a connection from a raw TLS stream (server-side).
    pub async fn from_server_stream(
        stream: ServerTlsStream<TcpStream>,
    ) -> Result<Self, TransportError> {
        let (read_half, write_half) = tokio::io::split(stream);
        Self::setup(ConnectionRole::Server, read_half, write_half).await
    }

    /// Create a connection from a raw TLS stream (client-side).
    pub async fn from_client_stream(
        stream: ClientTlsStream<TcpStream>,
    ) -> Result<Self, TransportError> {
        let (read_half, write_half) = tokio::io::split(stream);
        Self::setup(ConnectionRole::Client, read_half, write_half).await
    }

    async fn setup<R, W>(role: ConnectionRole, read_half: R, write_half: W) -> Result<Self, TransportError>
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
        W: tokio::io::AsyncWrite + Unpin + Send + 'static,
    {
        let pending: Arc<dashmap::DashMap<u64, oneshot::Sender<Message>>> =
            Arc::new(dashmap::DashMap::new());

        // Channel for outgoing raw frames
        let (out_tx, mut out_rx) = mpsc::channel::<Vec<u8>>(256);
        // Channel for incoming decoded messages
        let (in_tx, in_rx) = mpsc::channel::<Message>(256);

        // Writer task: sends framed bytes to the network
        let mut write_half = write_half;
        tokio::spawn(async move {
            while let Some(frame) = out_rx.recv().await {
                if let Err(e) = AsyncWriteExt::write_all(&mut write_half, &frame).await {
                    error!("write error: {}", e);
                    break;
                }
            }
        });

        // Reader task: reads frames from the network and dispatches
        let pending_clone = pending.clone();
        let mut read_half = read_half;
        tokio::spawn(async move {
            let mut header_buf = [0u8; HEADER_SIZE];
            loop {
                // Read frame header
                match AsyncReadExt::read_exact(&mut read_half, &mut header_buf).await {
                    Ok(_) => {}
                    Err(e) => {
                        debug!("connection closed: {}", e);
                        break;
                    }
                }

                let (flags, _stream_id, payload_len) = match wire::decode_header(&header_buf) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("invalid frame header: {}", e);
                        break;
                    }
                };

                // Read payload
                let mut payload = vec![0u8; payload_len as usize];
                if let Err(e) = AsyncReadExt::read_exact(&mut read_half, &mut payload).await {
                    error!("payload read error: {}", e);
                    break;
                }

                // Decode message
                let msg = match wire::decode_message(&payload, flags) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("message decode error: {}", e);
                        continue;
                    }
                };

                // Check if this is a response to a pending request
                let is_response = match &msg {
                    Message::CudaResponse { request_id, .. } => Some(request_id.0),
                    Message::VulkanResponse { request_id, .. } => Some(request_id.0),
                    Message::AuthResult { .. } => None, // handled differently
                    _ => None,
                };

                if let Some(req_id) = is_response {
                    if let Some((_, sender)) = pending_clone.remove(&req_id) {
                        let _ = sender.send(msg);
                        continue;
                    }
                }

                // Otherwise, send to the general incoming channel
                if in_tx.send(msg).await.is_err() {
                    debug!("incoming channel closed");
                    break;
                }
            }
        });

        Ok(Self {
            role,
            tx: out_tx,
            rx: Arc::new(Mutex::new(in_rx)),
            next_request_id: AtomicU64::new(1),
            pending,
        })
    }

    /// Send a message without waiting for a response.
    pub async fn send(&self, msg: Message) -> Result<(), TransportError> {
        let frame = wire::encode_message(&msg, 0)?;
        self.tx
            .send(frame)
            .await
            .map_err(|_| TransportError::ConnectionClosed)
    }

    /// Send a CUDA command and wait for its response.
    pub async fn send_cuda_command(
        &self,
        command: rgpu_protocol::cuda_commands::CudaCommand,
    ) -> Result<rgpu_protocol::cuda_commands::CudaResponse, TransportError> {
        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));

        let (tx, rx) = oneshot::channel();
        self.pending.insert(request_id.0, tx);

        let msg = Message::CudaCommand {
            request_id,
            command,
        };
        self.send(msg).await?;

        let response = rx.await.map_err(|_| TransportError::ConnectionClosed)?;

        match response {
            Message::CudaResponse { response, .. } => Ok(response),
            Message::Error(e) => Err(TransportError::AuthFailed(e.to_string())),
            _ => Err(TransportError::ConnectionClosed),
        }
    }

    /// Receive the next incoming message (non-response messages).
    pub async fn recv(&self) -> Result<Message, TransportError> {
        self.rx
            .lock()
            .await
            .recv()
            .await
            .ok_or(TransportError::ConnectionClosed)
    }

    pub fn role(&self) -> ConnectionRole {
        self.role
    }
}
