//! Synchronous IPC client for communicating with the RGPU client daemon.
//! Adapted from the CUDA interpose IPC client for Vulkan commands.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use rgpu_protocol::messages::{Message, RequestId};
use rgpu_protocol::vulkan_commands::{VulkanCommand, VulkanResponse};
use rgpu_protocol::wire;

/// Synchronous IPC client that connects to the RGPU client daemon.
pub struct IpcClient {
    path: String,
    next_request_id: AtomicU64,
    connection: Mutex<Option<IpcConnection>>,
}

struct IpcConnection {
    #[cfg(unix)]
    stream: std::os::unix::net::UnixStream,
    #[cfg(windows)]
    pipe: std::fs::File,
}

impl IpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            next_request_id: AtomicU64::new(1),
            connection: Mutex::new(None),
        }
    }

    /// Send a Vulkan command to the daemon and wait for the response.
    pub fn send_command(&self, cmd: VulkanCommand) -> Result<VulkanResponse, String> {
        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));

        let msg = Message::VulkanCommand {
            request_id,
            command: cmd,
        };

        let response = self.send_and_receive(msg)?;

        match response {
            Message::VulkanResponse { response, .. } => Ok(response),
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
            conn_guard.as_mut().expect("connection was just set to Some")
        };

        if let Err(_e) = conn.write_all(&frame) {
            drop(conn_guard);
            let mut conn_guard = self.connection.lock().map_err(|e| e.to_string())?;
            let new_conn = IpcConnection::connect(&self.path)?;
            *conn_guard = Some(new_conn);
            let conn = conn_guard.as_mut().expect("connection was just set to Some");
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
        #[cfg(unix)]
        {
            let stream = std::os::unix::net::UnixStream::connect(path)
                .map_err(|e| format!("{}", e))?;
            stream
                .set_read_timeout(Some(std::time::Duration::from_secs(30)))
                .ok();
            Ok(Self { stream })
        }

        #[cfg(windows)]
        {
            let pipe = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(path)
                .map_err(|e| format!("{}", e))?;
            Ok(Self { pipe })
        }
    }

    fn write_all(&mut self, data: &[u8]) -> Result<(), String> {
        #[cfg(unix)]
        {
            self.stream
                .write_all(data)
                .map_err(|e| format!("IPC write error: {}", e))
        }

        #[cfg(windows)]
        {
            self.pipe
                .write_all(data)
                .map_err(|e| format!("IPC write error: {}", e))
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), String> {
        #[cfg(unix)]
        {
            self.stream
                .read_exact(buf)
                .map_err(|e| format!("IPC read error: {}", e))
        }

        #[cfg(windows)]
        {
            self.pipe
                .read_exact(buf)
                .map_err(|e| format!("IPC read error: {}", e))
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
