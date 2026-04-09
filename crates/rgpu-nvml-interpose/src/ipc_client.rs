//! NVML IPC client — thin wrapper around the shared BaseIpcClient.

use rgpu_ipc_client::{BaseIpcClient, IpcError};
use rgpu_protocol::gpu_info::GpuInfo;
use rgpu_protocol::messages::Message;

/// Synchronous IPC client for NVML GPU queries.
pub struct NvmlIpcClient {
    inner: BaseIpcClient,
}

impl NvmlIpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            inner: BaseIpcClient::new(path),
        }
    }

    /// Query the daemon for the list of available GPUs.
    pub fn query_gpus(&self) -> Result<Vec<GpuInfo>, IpcError> {
        let msg = Message::QueryGpus;
        let response = self.inner.send_and_receive(msg)?;

        match response {
            Message::GpuList(gpus) => Ok(gpus),
            Message::Error(e) => Err(IpcError::DaemonError(e.to_string())),
            other => Err(IpcError::UnexpectedResponse(format!("{:?}", other))),
        }
    }
}
