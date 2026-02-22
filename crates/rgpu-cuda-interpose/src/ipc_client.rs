//! Synchronous IPC client for communicating with the RGPU client daemon.
//! This must be synchronous because CUDA API calls are synchronous.
//!
//! Supports command pipelining: void CUDA commands (memcpy, memset, free, etc.)
//! are batched and sent as a single `Message::CudaBatch` at the next sync point.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};
use rgpu_protocol::messages::{Message, RequestId};
use rgpu_protocol::wire;

/// Maximum number of void commands to buffer before auto-flushing.
const PIPELINE_BATCH_SIZE: usize = 32;

/// Synchronous IPC client that connects to the RGPU client daemon.
pub struct IpcClient {
    path: String,
    next_request_id: AtomicU64,
    /// Connection is lazily established and reused.
    connection: Mutex<Option<IpcConnection>>,
    /// Buffered void CUDA commands waiting to be flushed.
    pipeline_buffer: Mutex<Vec<CudaCommand>>,
}

struct IpcConnection {
    #[cfg(unix)]
    stream: std::os::unix::net::UnixStream,
    #[cfg(windows)]
    pipe: std::fs::File,
}

/// Returns true if this CUDA command is "void" — it always returns Success
/// and doesn't produce data the caller needs immediately.
fn is_void_command(cmd: &CudaCommand) -> bool {
    matches!(
        cmd,
        // Memory transfers (fire-and-forget, errors surface at next sync)
        CudaCommand::MemcpyHtoD { .. }
        | CudaCommand::MemcpyHtoDAsync { .. }
        | CudaCommand::MemcpyDtoD { .. }
        | CudaCommand::MemcpyDtoDAsync { .. }
        // Memset operations
        | CudaCommand::MemsetD8 { .. }
        | CudaCommand::MemsetD16 { .. }
        | CudaCommand::MemsetD32 { .. }
        | CudaCommand::MemsetD8Async { .. }
        | CudaCommand::MemsetD16Async { .. }
        | CudaCommand::MemsetD32Async { .. }
        // Free operations
        | CudaCommand::MemFree { .. }
        | CudaCommand::MemFreeHost { .. }
        | CudaCommand::MemFreeAsync { .. }
        // Context state changes
        | CudaCommand::CtxSetCurrent { .. }
        | CudaCommand::CtxPushCurrent { .. }
        | CudaCommand::CtxSetCacheConfig { .. }
        | CudaCommand::CtxSetLimit { .. }
        | CudaCommand::CtxSetFlags { .. }
        | CudaCommand::CtxResetPersistingL2Cache
        // Event recording
        | CudaCommand::EventRecord { .. }
        | CudaCommand::EventRecordWithFlags { .. }
        // Stream wait
        | CudaCommand::StreamWaitEvent { .. }
        // Function config
        | CudaCommand::FuncSetAttribute { .. }
        | CudaCommand::FuncSetCacheConfig { .. }
        | CudaCommand::FuncSetSharedMemConfig { .. }
        // Pointer/pool/primary ctx config
        | CudaCommand::PointerSetAttribute { .. }
        | CudaCommand::DevicePrimaryCtxSetFlags { .. }
        | CudaCommand::MemPoolTrimTo { .. }
        | CudaCommand::MemPoolSetAttribute { .. }
    )
}

impl IpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            next_request_id: AtomicU64::new(1),
            connection: Mutex::new(None),
            pipeline_buffer: Mutex::new(Vec::new()),
        }
    }

    /// Send a CUDA command to the daemon and wait for the response.
    /// Void commands are batched and sent at the next sync point.
    pub fn send_command(&self, cmd: CudaCommand) -> Result<CudaResponse, String> {
        if is_void_command(&cmd) {
            let mut buf = self.pipeline_buffer.lock().map_err(|e| e.to_string())?;
            buf.push(cmd);

            // Auto-flush when buffer is full
            if buf.len() >= PIPELINE_BATCH_SIZE {
                self.flush_pipeline_locked(&mut buf)?;
            }

            return Ok(CudaResponse::Success);
        }

        // Sync point: flush any buffered commands first
        self.flush_pipeline()?;

        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let msg = Message::CudaCommand {
            request_id,
            command: cmd,
        };

        let response = self.send_and_receive(msg)?;

        match response {
            Message::CudaResponse { response, .. } => Ok(response),
            Message::Error(e) => Err(e.to_string()),
            other => Err(format!("unexpected response: {:?}", other)),
        }
    }

    /// Flush any buffered pipeline commands.
    fn flush_pipeline(&self) -> Result<(), String> {
        let mut buf = self.pipeline_buffer.lock().map_err(|e| e.to_string())?;
        if buf.is_empty() {
            return Ok(());
        }
        self.flush_pipeline_locked(&mut buf)
    }

    /// Flush pipeline buffer (caller already holds the lock).
    fn flush_pipeline_locked(&self, buf: &mut Vec<CudaCommand>) -> Result<(), String> {
        if buf.is_empty() {
            return Ok(());
        }

        let batch = Message::CudaBatch(buf.drain(..).collect());
        let response = self.send_and_receive(batch)?;

        // Check for errors in the batch response
        match response {
            Message::CudaResponse {
                response: CudaResponse::Error { code, message },
                ..
            } => Err(format!("batch error (code {}): {}", code, message)),
            _ => Ok(()),
        }
    }

    fn send_and_receive(&self, msg: Message) -> Result<Message, String> {
        let frame = wire::encode_message(&msg, 0).map_err(|e| e.to_string())?;

        let mut conn_guard = self.connection.lock().map_err(|e| e.to_string())?;

        // Try to reuse existing connection, or create a new one
        let conn = if let Some(ref mut c) = *conn_guard {
            c
        } else {
            let new_conn = IpcConnection::connect(&self.path)?;
            *conn_guard = Some(new_conn);
            conn_guard.as_mut().expect("connection was just set to Some")
        };

        // Send frame
        if let Err(_e) = conn.write_all(&frame) {
            // Connection lost, try reconnecting once
            drop(conn_guard);
            let mut conn_guard = self.connection.lock().map_err(|e| e.to_string())?;
            let new_conn = IpcConnection::connect(&self.path)?;
            *conn_guard = Some(new_conn);
            let conn = conn_guard.as_mut().expect("connection was just set to Some");
            conn.write_all(&frame).map_err(|e| e.to_string())?;

            // Read response
            let response = conn.read_message()?;
            return Ok(response);
        }

        // Read response
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
