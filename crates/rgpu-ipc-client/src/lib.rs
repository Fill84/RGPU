//! Shared synchronous IPC client for RGPU interpose libraries.
//!
//! This crate provides the common IPC transport layer used by all interpose
//! libraries (CUDA, Vulkan, NVENC, NVDEC, NVML, NVAPI) to communicate with
//! the RGPU client daemon. Each interpose library builds a thin wrapper
//! around this shared infrastructure.
//!
//! The client is synchronous because GPU API calls are synchronous from the
//! application's perspective.

use std::fmt;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use rgpu_protocol::messages::{Message, RequestId};
use rgpu_protocol::wire;

// ── Error Type ─────────────────────────────────────────────────────────

/// Errors that can occur during IPC communication with the daemon.
#[derive(Debug)]
pub enum IpcError {
    /// Failed to connect to the daemon.
    Connect { path: String, cause: String },
    /// IO error during read/write.
    Io(std::io::Error),
    /// Wire protocol error (serialization/deserialization).
    Wire(String),
    /// Mutex was poisoned (another thread panicked while holding it).
    LockPoisoned,
    /// The daemon returned an error message.
    DaemonError(String),
    /// Unexpected response type from daemon.
    UnexpectedResponse(String),
}

impl fmt::Display for IpcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IpcError::Connect { path, cause } => {
                write!(f, "failed to connect to RGPU daemon at {}: {}", path, cause)
            }
            IpcError::Io(e) => write!(f, "IPC IO error: {}", e),
            IpcError::Wire(e) => write!(f, "IPC wire error: {}", e),
            IpcError::LockPoisoned => write!(f, "IPC client lock poisoned"),
            IpcError::DaemonError(e) => write!(f, "daemon error: {}", e),
            IpcError::UnexpectedResponse(msg) => write!(f, "unexpected response: {}", msg),
        }
    }
}

impl std::error::Error for IpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IpcError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for IpcError {
    fn from(e: std::io::Error) -> Self {
        IpcError::Io(e)
    }
}

// ── IPC Transport ──────────────────────────────────────────────────────

enum IpcTransport {
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixStream),
    #[cfg(windows)]
    Pipe(std::fs::File),
    Tcp(std::net::TcpStream),
}

/// Low-level IPC connection to the RGPU client daemon.
pub struct IpcConnection {
    transport: IpcTransport,
}

impl IpcConnection {
    /// Connect to the daemon with retry logic.
    pub fn connect(path: &str) -> Result<Self, IpcError> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY_MS: u64 = 500;

        let mut last_err = String::new();
        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_millis(RETRY_DELAY_MS));
            }

            match Self::try_connect(path) {
                Ok(conn) => return Ok(conn),
                Err(e) => {
                    last_err = e.to_string();
                }
            }
        }

        Err(IpcError::Connect {
            path: path.to_string(),
            cause: format!("after {} attempts: {}", MAX_RETRIES, last_err),
        })
    }

    fn try_connect(path: &str) -> Result<Self, IpcError> {
        // Check for TCP address (host:port format)
        if rgpu_common::platform::is_tcp_address(path) {
            let stream = std::net::TcpStream::connect(path)?;
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                .ok();
            return Ok(Self {
                transport: IpcTransport::Tcp(stream),
            });
        }

        #[cfg(unix)]
        {
            let stream = std::os::unix::net::UnixStream::connect(path)?;
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                .ok();
            Ok(Self {
                transport: IpcTransport::Unix(stream),
            })
        }

        #[cfg(windows)]
        {
            let pipe = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)?;
            Ok(Self {
                transport: IpcTransport::Pipe(pipe),
            })
        }
    }

    /// Write all data to the transport.
    pub fn write_all(&mut self, data: &[u8]) -> Result<(), IpcError> {
        match &mut self.transport {
            #[cfg(unix)]
            IpcTransport::Unix(s) => s.write_all(data)?,
            #[cfg(windows)]
            IpcTransport::Pipe(p) => p.write_all(data)?,
            IpcTransport::Tcp(s) => s.write_all(data)?,
        }
        Ok(())
    }

    /// Read exactly `buf.len()` bytes from the transport.
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), IpcError> {
        match &mut self.transport {
            #[cfg(unix)]
            IpcTransport::Unix(s) => s.read_exact(buf)?,
            #[cfg(windows)]
            IpcTransport::Pipe(p) => p.read_exact(buf)?,
            IpcTransport::Tcp(s) => s.read_exact(buf)?,
        }
        Ok(())
    }

    /// Read a complete protocol message from the transport.
    pub fn read_message(&mut self) -> Result<Message, IpcError> {
        let mut header_buf = [0u8; wire::HEADER_SIZE];
        self.read_exact(&mut header_buf)?;

        let (flags, _stream_id, payload_len) =
            wire::decode_header(&header_buf).map_err(|e| IpcError::Wire(e.to_string()))?;

        let mut payload = vec![0u8; payload_len as usize];
        self.read_exact(&mut payload)?;

        wire::decode_message(&payload, flags).map_err(|e| IpcError::Wire(e.to_string()))
    }
}

// ── Generic IPC Client ─────────────────────────────────────────────────

/// Synchronous IPC client with lazy connection and reconnection support.
///
/// This provides the shared `send_and_receive` logic used by all interpose
/// crate IPC clients.
pub struct BaseIpcClient {
    path: String,
    next_request_id: AtomicU64,
    connection: Mutex<Option<IpcConnection>>,
}

impl Drop for BaseIpcClient {
    fn drop(&mut self) {
        // Explicitly close the connection on drop
        let mut guard = self.connection.lock();
        *guard = None;
    }
}

impl BaseIpcClient {
    /// Create a new client targeting the given IPC path.
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            next_request_id: AtomicU64::new(1),
            connection: Mutex::new(None),
        }
    }

    /// Get the next request ID.
    pub fn next_request_id(&self) -> RequestId {
        RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Send a message and wait for the response.
    ///
    /// Handles lazy connection establishment and automatic reconnection
    /// on connection loss.
    pub fn send_and_receive(&self, msg: Message) -> Result<Message, IpcError> {
        let frame = wire::encode_message(&msg, 0).map_err(|e| IpcError::Wire(e.to_string()))?;

        let mut conn_guard = self.connection.lock();

        // Try to reuse existing connection, or create a new one
        let conn = if let Some(ref mut c) = *conn_guard {
            c
        } else {
            let new_conn = IpcConnection::connect(&self.path)?;
            *conn_guard = Some(new_conn);
            conn_guard
                .as_mut()
                .expect("connection was just set to Some")
        };

        // Send frame
        if let Err(_e) = conn.write_all(&frame) {
            // Connection lost, try reconnecting once
            drop(conn_guard);
            let mut conn_guard =
                self.connection.lock();
            let new_conn = IpcConnection::connect(&self.path)?;
            *conn_guard = Some(new_conn);
            let conn = conn_guard
                .as_mut()
                .expect("connection was just set to Some");
            conn.write_all(&frame)?;

            // Read response
            let response = conn.read_message()?;
            return Ok(response);
        }

        // Read response
        let response = conn.read_message()?;
        Ok(response)
    }
}