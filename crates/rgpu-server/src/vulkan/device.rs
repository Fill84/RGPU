use ash::vk;
use tracing::{debug, info};

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_device(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (physical_device, queue_create_infos, enabled_features) = match cmd {
            VulkanCommand::CreateDevice {
                physical_device,
                queue_create_infos,
                enabled_extensions: _,
                enabled_features,
            } => (physical_device, queue_create_infos, enabled_features),
            _ => unreachable!(),
        };

        let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
            Some(e) => *e.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid physical device handle".to_string(),
                }
            }
        };
        let wrapper = match self.instance_wrappers.get(&inst_handle) {
            Some(w) => w,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                    message: "instance wrapper not found".to_string(),
                }
            }
        };

        // Build queue create infos
        let queue_priorities: Vec<Vec<f32>> = queue_create_infos
            .iter()
            .map(|qi| qi.queue_priorities.clone())
            .collect();

        let vk_queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_create_infos
            .iter()
            .enumerate()
            .map(|(i, qi)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(qi.queue_family_index)
                    .queue_priorities(&queue_priorities[i])
            })
            .collect();

        // Optionally set features
        let features = enabled_features.and_then(|raw| {
            if raw.len() == std::mem::size_of::<vk::PhysicalDeviceFeatures>() {
                Some(unsafe {
                    std::ptr::read(raw.as_ptr() as *const vk::PhysicalDeviceFeatures)
                })
            } else {
                None
            }
        });

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&vk_queue_create_infos);

        if let Some(ref f) = features {
            device_create_info = device_create_info.enabled_features(f);
        }

        match unsafe { wrapper.create_device(pd, &device_create_info, None) } {
            Ok(device) => {
                let handle = session.alloc_handle(ResourceType::VkDevice);
                let raw = device.handle();
                self.device_handles.insert(handle, raw);
                self.device_wrappers.insert(handle, device);
                self.device_to_instance.insert(handle, inst_handle);
                info!("created Vulkan device: {:?}", handle);
                VulkanResponse::DeviceCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_device(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let device = match cmd {
            VulkanCommand::DestroyDevice { device } => device,
            _ => unreachable!(),
        };

        if let Some((_, dev)) = self.device_wrappers.remove(&device) {
            unsafe { dev.destroy_device(None) };
            self.device_handles.remove(&device);
            self.device_to_instance.remove(&device);
            session.remove_handle(&device);
            debug!("destroyed Vulkan device: {:?}", device);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_device_wait_idle(&self, cmd: VulkanCommand) -> VulkanResponse {
        let device = match cmd {
            VulkanCommand::DeviceWaitIdle { device } => device,
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
        match unsafe { dev.device_wait_idle() } {
            Ok(()) => VulkanResponse::Success,
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_get_device_queue(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, queue_family_index, queue_index) = match cmd {
            VulkanCommand::GetDeviceQueue {
                device,
                queue_family_index,
                queue_index,
            } => (device, queue_family_index, queue_index),
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
        let queue =
            unsafe { dev.get_device_queue(queue_family_index, queue_index) };
        let handle = session.alloc_handle(ResourceType::VkQueue);
        self.queue_handles.insert(handle, queue);
        debug!(
            "got queue family={} index={}: {:?}",
            queue_family_index, queue_index, handle
        );
        VulkanResponse::QueueRetrieved { handle }
    }
}
