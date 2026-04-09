//! CUDA IPC client — wraps the shared BaseIpcClient with command pipelining.
//!
//! Supports command pipelining: void CUDA commands (memcpy, memset, free, etc.)
//! are batched and sent as a single `Message::CudaBatch` at the next sync point.

use parking_lot::Mutex;

use rgpu_ipc_client::{BaseIpcClient, IpcError};
use rgpu_protocol::cuda_commands::{CudaCommand, CudaResponse};
use rgpu_protocol::messages::Message;

/// Maximum number of void commands to buffer before auto-flushing.
const PIPELINE_BATCH_SIZE: usize = 32;

/// Synchronous IPC client with CUDA command pipelining.
pub struct IpcClient {
    inner: BaseIpcClient,
    /// Buffered void CUDA commands waiting to be flushed.
    pipeline_buffer: Mutex<Vec<CudaCommand>>,
}

/// Returns true if this CUDA command is "void" — it always returns Success
/// and doesn't produce data the caller needs immediately.
fn is_void_command(cmd: &CudaCommand) -> bool {
    matches!(
        cmd,
        CudaCommand::MemcpyHtoD { .. }
        | CudaCommand::MemcpyHtoDAsync { .. }
        | CudaCommand::MemcpyDtoD { .. }
        | CudaCommand::MemcpyDtoDAsync { .. }
        | CudaCommand::MemsetD8 { .. }
        | CudaCommand::MemsetD16 { .. }
        | CudaCommand::MemsetD32 { .. }
        | CudaCommand::MemsetD8Async { .. }
        | CudaCommand::MemsetD16Async { .. }
        | CudaCommand::MemsetD32Async { .. }
        | CudaCommand::MemFree { .. }
        | CudaCommand::MemFreeHost { .. }
        | CudaCommand::MemFreeAsync { .. }
        | CudaCommand::CtxSetCurrent { .. }
        | CudaCommand::CtxPushCurrent { .. }
        | CudaCommand::CtxSetCacheConfig { .. }
        | CudaCommand::CtxSetLimit { .. }
        | CudaCommand::CtxSetFlags { .. }
        | CudaCommand::CtxResetPersistingL2Cache
        | CudaCommand::EventRecord { .. }
        | CudaCommand::EventRecordWithFlags { .. }
        | CudaCommand::StreamWaitEvent { .. }
        | CudaCommand::FuncSetAttribute { .. }
        | CudaCommand::FuncSetCacheConfig { .. }
        | CudaCommand::FuncSetSharedMemConfig { .. }
        | CudaCommand::PointerSetAttribute { .. }
        | CudaCommand::DevicePrimaryCtxSetFlags { .. }
        | CudaCommand::MemPoolTrimTo { .. }
        | CudaCommand::MemPoolSetAttribute { .. }
    )
}

impl IpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            inner: BaseIpcClient::new(path),
            pipeline_buffer: Mutex::new(Vec::new()),
        }
    }

    /// Send a CUDA command to the daemon and wait for the response.
    /// Void commands are batched and sent at the next sync point.
    pub fn send_command(&self, cmd: CudaCommand) -> Result<CudaResponse, IpcError> {
        if is_void_command(&cmd) {
            let mut buf = self.pipeline_buffer.lock();
            buf.push(cmd);

            if buf.len() >= PIPELINE_BATCH_SIZE {
                self.flush_pipeline_locked(&mut buf)?;
            }

            return Ok(CudaResponse::Success);
        }

        // Sync point: flush any buffered commands first
        self.flush_pipeline()?;

        let request_id = self.inner.next_request_id();
        let msg = Message::CudaCommand {
            request_id,
            command: cmd,
        };

        let response = self.inner.send_and_receive(msg)?;

        match response {
            Message::CudaResponse { response, .. } => Ok(response),
            Message::Error(e) => Err(IpcError::DaemonError(e.to_string())),
            other => Err(IpcError::UnexpectedResponse(format!("{:?}", other))),
        }
    }

    /// Flush any buffered pipeline commands.
    fn flush_pipeline(&self) -> Result<(), IpcError> {
        let mut buf = self.pipeline_buffer.lock();
        if buf.is_empty() {
            return Ok(());
        }
        self.flush_pipeline_locked(&mut buf)
    }

    /// Flush pipeline buffer (caller already holds the lock).
    fn flush_pipeline_locked(&self, buf: &mut Vec<CudaCommand>) -> Result<(), IpcError> {
        if buf.is_empty() {
            return Ok(());
        }

        let batch = Message::CudaBatch(buf.drain(..).collect());
        let response = self.inner.send_and_receive(batch)?;

        match response {
            Message::CudaResponse {
                response: CudaResponse::Error { code, message },
                ..
            } => Err(IpcError::DaemonError(format!(
                "batch error (code {}): {}",
                code, message
            ))),
            _ => Ok(()),
        }
    }
}
