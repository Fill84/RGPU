//! Vulkan IPC client — thin wrapper around the shared BaseIpcClient.

use rgpu_ipc_client::{BaseIpcClient, IpcError};
use rgpu_protocol::messages::Message;
use rgpu_protocol::vulkan_commands::{VulkanCommand, VulkanResponse};

/// Synchronous IPC client for Vulkan commands.
pub struct IpcClient {
    inner: BaseIpcClient,
}

impl IpcClient {
    pub fn new(path: &str) -> Self {
        Self {
            inner: BaseIpcClient::new(path),
        }
    }

    /// Send a Vulkan command to the daemon and wait for the response.
    pub fn send_command(&self, cmd: VulkanCommand) -> Result<VulkanResponse, IpcError> {
        let request_id = self.inner.next_request_id();
        let msg = Message::VulkanCommand {
            request_id,
            command: cmd,
        };

        let response = self.inner.send_and_receive(msg)?;

        match response {
            Message::VulkanResponse { response, .. } => Ok(response),
            Message::Error(e) => Err(IpcError::DaemonError(e.to_string())),
            other => Err(IpcError::UnexpectedResponse(format!("{:?}", other))),
        }
    }
}
