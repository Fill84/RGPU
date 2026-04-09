//! NVENC IPC client — thin wrapper around the shared BaseIpcClient.

use rgpu_ipc_client::{BaseIpcClient, IpcError};
use rgpu_protocol::messages::Message;
use rgpu_protocol::nvenc_commands::{NvencCommand, NvencResponse};

/// Synchronous IPC client for NVENC encoding commands.
pub struct NvencIpcClient {
    inner: BaseIpcClient,
}

impl NvencIpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            inner: BaseIpcClient::new(path),
        }
    }

    /// Send an NVENC command to the daemon and wait for the response.
    pub fn send_command(&self, cmd: NvencCommand) -> Result<NvencResponse, IpcError> {
        let request_id = self.inner.next_request_id();
        let msg = Message::NvencCommand {
            request_id,
            command: cmd,
        };

        let response = self.inner.send_and_receive(msg)?;

        match response {
            Message::NvencResponse { response, .. } => Ok(response),
            Message::Error(e) => Err(IpcError::DaemonError(e.to_string())),
            other => Err(IpcError::UnexpectedResponse(format!("{:?}", other))),
        }
    }
}
