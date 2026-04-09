use ash::vk;
use tracing::debug;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_image(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, create_info) = match cmd {
            VulkanCommand::CreateImage { device, create_info } => (device, create_info),
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

        let ci = &create_info;
        let qfi = &ci.queue_family_indices;
        let mut image_ci = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::from_raw(ci.flags))
            .image_type(vk::ImageType::from_raw(ci.image_type))
            .format(vk::Format::from_raw(ci.format))
            .extent(vk::Extent3D {
                width: ci.extent[0],
                height: ci.extent[1],
                depth: ci.extent[2],
            })
            .mip_levels(ci.mip_levels)
            .array_layers(ci.array_layers)
            .samples(vk::SampleCountFlags::from_raw(ci.samples))
            .tiling(vk::ImageTiling::from_raw(ci.tiling))
            .usage(vk::ImageUsageFlags::from_raw(ci.usage))
            .sharing_mode(vk::SharingMode::from_raw(ci.sharing_mode))
            .initial_layout(vk::ImageLayout::from_raw(ci.initial_layout));
        if !qfi.is_empty() {
            image_ci = image_ci.queue_family_indices(qfi);
        }

        match unsafe { dev.create_image(&image_ci, None) } {
            Ok(image) => {
                let handle = session.alloc_handle(ResourceType::VkImage);
                self.image_handles.insert(handle, image);
                self.image_to_device.insert(handle, device);
                debug!("created image: {:?}", handle);
                VulkanResponse::ImageCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_image(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, image) = match cmd {
            VulkanCommand::DestroyImage { device, image } => (device, image),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, img)) = self.image_handles.remove(&image) {
            unsafe { dev.destroy_image(img, None) };
            self.image_to_device.remove(&image);
            session.remove_handle(&image);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_bind_image_memory(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, image, memory, memory_offset) = match cmd {
            VulkanCommand::BindImageMemory {
                device,
                image,
                memory,
                memory_offset,
            } => (device, image, memory, memory_offset),
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
        let img = match self.image_handles.get(&image) {
            Some(i) => *i.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid image handle".to_string(),
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
        match unsafe { dev.bind_image_memory(img, mem, memory_offset) } {
            Ok(()) => VulkanResponse::Success,
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_create_image_view(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, image, view_type, format, components, subresource_range) = match cmd {
            VulkanCommand::CreateImageView {
                device,
                image,
                view_type,
                format,
                components,
                subresource_range,
            } => (device, image, view_type, format, components, subresource_range),
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
        let img = match self.image_handles.get(&image) {
            Some(i) => *i.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid image handle".to_string(),
                }
            }
        };
        let ci = vk::ImageViewCreateInfo::default()
            .image(img)
            .view_type(vk::ImageViewType::from_raw(view_type))
            .format(vk::Format::from_raw(format))
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::from_raw(components.r),
                g: vk::ComponentSwizzle::from_raw(components.g),
                b: vk::ComponentSwizzle::from_raw(components.b),
                a: vk::ComponentSwizzle::from_raw(components.a),
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::from_raw(subresource_range.aspect_mask),
                base_mip_level: subresource_range.base_mip_level,
                level_count: subresource_range.level_count,
                base_array_layer: subresource_range.base_array_layer,
                layer_count: subresource_range.layer_count,
            });

        match unsafe { dev.create_image_view(&ci, None) } {
            Ok(view) => {
                let handle = session.alloc_handle(ResourceType::VkImageView);
                self.image_view_handles.insert(handle, view);
                self.image_view_to_device.insert(handle, device);
                debug!("created image view: {:?}", handle);
                VulkanResponse::ImageViewCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_image_view(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, image_view) = match cmd {
            VulkanCommand::DestroyImageView { device, image_view } => (device, image_view),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, view)) = self.image_view_handles.remove(&image_view) {
            unsafe { dev.destroy_image_view(view, None) };
            self.image_view_to_device.remove(&image_view);
            session.remove_handle(&image_view);
        }
        VulkanResponse::Success
    }
}
