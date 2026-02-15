//! Image and ImageView functions for the Vulkan ICD.

use ash::vk;
use ash::vk::Handle;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{
    SerializedComponentMapping, SerializedImageCreateInfo, SerializedImageSubresourceRange,
    VulkanCommand, VulkanResponse,
};

// ── vkCreateImage ────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateImage(
    device: vk::Device,
    p_create_info: *const vk::ImageCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_image: *mut vk::Image,
) -> vk::Result {
    if p_create_info.is_null() || p_image.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;
    let queue_family_indices = if !ci.p_queue_family_indices.is_null()
        && ci.queue_family_index_count > 0
    {
        std::slice::from_raw_parts(ci.p_queue_family_indices, ci.queue_family_index_count as usize)
            .to_vec()
    } else {
        Vec::new()
    };

    let cmd = VulkanCommand::CreateImage {
        device: dev_handle,
        create_info: SerializedImageCreateInfo {
            flags: ci.flags.as_raw(),
            image_type: ci.image_type.as_raw(),
            format: ci.format.as_raw(),
            extent: [ci.extent.width, ci.extent.height, ci.extent.depth],
            mip_levels: ci.mip_levels,
            array_layers: ci.array_layers,
            samples: ci.samples.as_raw(),
            tiling: ci.tiling.as_raw(),
            usage: ci.usage.as_raw(),
            sharing_mode: ci.sharing_mode.as_raw(),
            queue_family_indices,
            initial_layout: ci.initial_layout.as_raw(),
        },
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::ImageCreated { handle }) => {
            let local_id = handle_store::store_image(handle);
            *p_image = vk::Image::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    }
}

// ── vkDestroyImage ───────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkDestroyImage(
    device: vk::Device,
    image: vk::Image,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if image == vk::Image::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = image.as_raw();
    if let Some(handle) = handle_store::remove_image(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyImage {
            device: dev_handle,
            image: handle,
        });
    }
}

// ── vkGetImageMemoryRequirements ─────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetImageMemoryRequirements(
    device: vk::Device,
    image: vk::Image,
    p_memory_requirements: *mut vk::MemoryRequirements,
) {
    if p_memory_requirements.is_null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let img_handle = match handle_store::get_image(image.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetImageMemoryRequirements {
        device: dev_handle,
        image: img_handle,
    };

    if let Ok(VulkanResponse::MemoryRequirements {
        size,
        alignment,
        memory_type_bits,
    }) = send_vulkan_command(cmd)
    {
        let mr = &mut *p_memory_requirements;
        mr.size = size;
        mr.alignment = alignment;
        mr.memory_type_bits = memory_type_bits;
    }
}

// ── vkBindImageMemory ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkBindImageMemory(
    device: vk::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    memory_offset: vk::DeviceSize,
) -> vk::Result {
    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let img_handle = match handle_store::get_image(image.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let mem_handle = match handle_store::get_memory(memory.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let cmd = VulkanCommand::BindImageMemory {
        device: dev_handle,
        image: img_handle,
        memory: mem_handle,
        memory_offset,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

// ── vkCreateImageView ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateImageView(
    device: vk::Device,
    p_create_info: *const vk::ImageViewCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_view: *mut vk::ImageView,
) -> vk::Result {
    if p_create_info.is_null() || p_view.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;
    let img_handle = match handle_store::get_image(ci.image.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let cmd = VulkanCommand::CreateImageView {
        device: dev_handle,
        image: img_handle,
        view_type: ci.view_type.as_raw(),
        format: ci.format.as_raw(),
        components: SerializedComponentMapping {
            r: ci.components.r.as_raw(),
            g: ci.components.g.as_raw(),
            b: ci.components.b.as_raw(),
            a: ci.components.a.as_raw(),
        },
        subresource_range: SerializedImageSubresourceRange {
            aspect_mask: ci.subresource_range.aspect_mask.as_raw(),
            base_mip_level: ci.subresource_range.base_mip_level,
            level_count: ci.subresource_range.level_count,
            base_array_layer: ci.subresource_range.base_array_layer,
            layer_count: ci.subresource_range.layer_count,
        },
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::ImageViewCreated { handle }) => {
            let local_id = handle_store::store_image_view(handle);
            *p_view = vk::ImageView::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    }
}

// ── vkDestroyImageView ───────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkDestroyImageView(
    device: vk::Device,
    image_view: vk::ImageView,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if image_view == vk::ImageView::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = image_view.as_raw();
    if let Some(handle) = handle_store::remove_image_view(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyImageView {
            device: dev_handle,
            image_view: handle,
        });
    }
}
