use ash::vk;
use tracing::debug;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_buffer(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, size, usage, sharing_mode, queue_family_indices) = match cmd {
            VulkanCommand::CreateBuffer {
                device,
                size,
                usage,
                sharing_mode,
                queue_family_indices,
            } => (device, size, usage, sharing_mode, queue_family_indices),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };

        let mut create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::from_raw(usage))
            .sharing_mode(vk::SharingMode::from_raw(sharing_mode as i32));

        if !queue_family_indices.is_empty() {
            create_info =
                create_info.queue_family_indices(&queue_family_indices);
        }

        match unsafe { dev.create_buffer(&create_info, None) } {
            Ok(buffer) => {
                let handle = session.alloc_handle(ResourceType::VkBuffer);
                self.buffer_handles.insert(handle, buffer);
                self.buffer_to_device.insert(handle, device);
                debug!("created buffer ({} bytes): {:?}", size, handle);
                VulkanResponse::BufferCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_buffer(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, buffer) = match cmd {
            VulkanCommand::DestroyBuffer { device, buffer } => (device, buffer),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, buf)) = self.buffer_handles.remove(&buffer) {
            unsafe { dev.destroy_buffer(buf, None) };
            self.buffer_to_device.remove(&buffer);
            session.remove_handle(&buffer);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_bind_buffer_memory(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, buffer, memory, memory_offset) = match cmd {
            VulkanCommand::BindBufferMemory {
                device,
                buffer,
                memory,
                memory_offset,
            } => (device, buffer, memory, memory_offset),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid device handle".to_string(),
                }
            }
        };
        let buf = match self.buffer_handles.get(&buffer) {
            Some(b) => *b.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid buffer handle".to_string(),
                }
            }
        };
        let mem = match self.memory_handles.get(&memory) {
            Some(m) => *m.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid memory handle".to_string(),
                }
            }
        };
        match unsafe { dev.bind_buffer_memory(buf, mem, memory_offset) } {
            Ok(()) => VulkanResponse::Success,
            Err(e) => Self::vk_err(e),
        }
    }
}
