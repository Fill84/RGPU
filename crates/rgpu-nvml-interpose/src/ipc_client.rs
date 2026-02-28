//! Synchronous IPC client for communicating with the RGPU client daemon.
//! Used by the NVML interpose to query GPU information from the daemon.
//! No command pipelining needed — NVML queries are simple request/response.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use rgpu_protocol::gpu_info::GpuInfo;
use rgpu_protocol::messages::{Message, RequestId};
use rgpu_protocol::wire;

/// Synchronous IPC client that connects to the RGPU client daemon.
pub struct NvmlIpcClient {
    path: String,
    next_request_id: AtomicU64,
    connection: Mutex<Option<IpcConnection>>,
}

enum IpcTransport {
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixStream),
    #[cfg(windows)]
    Pipe(std::fs::File),
    Tcp(std::net::TcpStream),
}

struct IpcConnection {
    transport: IpcTransport,
}

impl NvmlIpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            next_request_id: AtomicU64::new(1),
            connection: Mutex::new(None),
        }
    }

    /// Query the daemon for the list of available GPUs.
    pub fn query_gpus(&self) -> Result<Vec<GpuInfo>, String> {
        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let _ = request_id; // QueryGpus doesn't use request_id
        let msg = Message::QueryGpus;

        let response = self.send_and_receive(msg)?;

        match response {
            Message::GpuList(gpus) => Ok(gpus),
            Message::Error(e) => Err(e.to_string()),
            other => Err(format!("unexpected response: {:?}", other)),
        }
    }

    fn send_and_receive(&self, msg: Message) -> Result<Message, String> {
        let frame = wire::encode_message(&msg, 0).map_err(|e| e.to_string())?;

        let mut conn_guard = self.connection.lock().map_err(|e| e.to_string())?;

        let conn = if let Some(ref mut c) = *conn_guard {
            c
        } else {
            let new_conn = IpcConnection::connect(&self.path)?;
            *conn_guard = Some(new_conn);
            conn_guard
                .as_mut()
                .expect("connection was just set to Some")
        };

        if let Err(_e) = conn.write_all(&frame) {
            // Connection lost, try reconnecting once
            drop(conn_guard);
            let mut conn_guard = self.connection.lock().map_err(|e| e.to_string())?;
            let new_conn = IpcConnection::connect(&self.path)?;
            *conn_guard = Some(new_conn);
            let conn = conn_guard
                .as_mut()
                .expect("connection was just set to Some");
            conn.write_all(&frame).map_err(|e| e.to_string())?;
            let response = conn.read_message()?;
            return Ok(response);
        }

        let response = conn.read_message()?;
        Ok(response)
    }
}

impl IpcConnection {
    fn connect(path: &str) -> Result<Self, String> {
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
                    last_err = e;
                }
            }
        }

        Err(format!(
            "failed to connect to RGPU daemon at {} after {} attempts: {}",
            path, MAX_RETRIES, last_err
        ))
    }

    fn try_connect(path: &str) -> Result<Self, String> {
        // Check for TCP address (contains ':' and no pipe/socket prefix)
        if path.contains(':') && !path.starts_with(r"\\") && !path.starts_with('/') {
            let stream =
                std::net::TcpStream::connect(path).map_err(|e| format!("TCP connect: {}", e))?;
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                .ok();
            return Ok(Self {
                transport: IpcTransport::Tcp(stream),
            });
        }

        #[cfg(unix)]
        {
            let stream = std::os::unix::net::UnixStream::connect(path)
                .map_err(|e| format!("{}", e))?;
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
                .open(path)
                .map_err(|e| format!("{}", e))?;
            Ok(Self {
                transport: IpcTransport::Pipe(pipe),
            })
        }
    }

    fn write_all(&mut self, data: &[u8]) -> Result<(), String> {
        match &mut self.transport {
            #[cfg(unix)]
            IpcTransport::Unix(stream) => stream
                .write_all(data)
                .map_err(|e| format!("IPC write error: {}", e)),
            #[cfg(windows)]
            IpcTransport::Pipe(pipe) => pipe
                .write_all(data)
                .map_err(|e| format!("IPC write error: {}", e)),
            IpcTransport::Tcp(stream) => stream
                .write_all(data)
                .map_err(|e| format!("IPC write error: {}", e)),
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), String> {
        match &mut self.transport {
            #[cfg(unix)]
            IpcTransport::Unix(stream) => stream
                .read_exact(buf)
                .map_err(|e| format!("IPC read error: {}", e)),
            #[cfg(windows)]
            IpcTransport::Pipe(pipe) => pipe
                .read_exact(buf)
                .map_err(|e| format!("IPC read error: {}", e)),
            IpcTransport::Tcp(stream) => stream
                .read_exact(buf)
                .map_err(|e| format!("IPC read error: {}", e)),
        }
    }

    fn read_message(&mut self) -> Result<Message, String> {
        let mut header_buf = [0u8; wire::HEADER_SIZE];
        self.read_exact(&mut header_buf)?;

        let (flags, _stream_id, payload_len) =
            wire::decode_header(&header_buf).map_err(|e| e.to_string())?;

        let mut payload = vec![0u8; payload_len as usize];
        self.read_exact(&mut payload)?;

        wire::decode_message(&payload, flags).map_err(|e| e.to_string())
    }
}
