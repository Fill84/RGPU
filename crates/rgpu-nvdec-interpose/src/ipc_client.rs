//! NVDEC IPC client — thin wrapper around the shared BaseIpcClient.

use rgpu_ipc_client::{BaseIpcClient, IpcError};
use rgpu_protocol::messages::Message;
use rgpu_protocol::nvdec_commands::{NvdecCommand, NvdecResponse};

/// Synchronous IPC client for NVDEC decoding commands.
pub struct NvdecIpcClient {
    inner: BaseIpcClient,
}

impl NvdecIpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            inner: BaseIpcClient::new(path),
        }
    }

    /// Send an NVDEC command to the daemon and wait for the response.
    pub fn send_command(&self, cmd: NvdecCommand) -> Result<NvdecResponse, IpcError> {
        let request_id = self.inner.next_request_id();
        let msg = Message::NvdecCommand {
            request_id,
            command: cmd,
        };

        let response = self.inner.send_and_receive(msg)?;

        match response {
            Message::NvdecResponse { response, .. } => Ok(response),
            Message::Error(e) => Err(IpcError::DaemonError(e.to_string())),
            other => Err(IpcError::UnexpectedResponse(format!("{:?}", other))),
        }
    }
}
