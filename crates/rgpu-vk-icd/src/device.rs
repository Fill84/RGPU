//! Logical device and queue functions for the Vulkan ICD.

use ash::vk;
use ash::vk::Handle;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{DeviceQueueCreateInfo, VulkanCommand, VulkanResponse};

#[no_mangle]
pub unsafe extern "C" fn vkCreateDevice(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_device: *mut vk::Device,
) -> vk::Result {
    if p_create_info.is_null() || p_device.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let disp = physical_device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let pd_handle = match handle_store::get_physical_device(local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;

    // Read queue create infos
    let mut queue_create_infos = Vec::new();
    if !ci.p_queue_create_infos.is_null() {
        for i in 0..ci.queue_create_info_count as usize {
            let qci = &*ci.p_queue_create_infos.add(i);
            let priorities = if !qci.p_queue_priorities.is_null() {
                std::slice::from_raw_parts(qci.p_queue_priorities, qci.queue_count as usize)
                    .to_vec()
            } else {
                vec![1.0; qci.queue_count as usize]
            };
            queue_create_infos.push(DeviceQueueCreateInfo {
                queue_family_index: qci.queue_family_index,
                queue_priorities: priorities,
            });
        }
    }

    // Read enabled extensions
    let enabled_extensions = read_string_array(
        ci.pp_enabled_extension_names,
        ci.enabled_extension_count,
    );

    // Read enabled features (raw bytes)
    let enabled_features = if !ci.p_enabled_features.is_null() {
        let features = &*ci.p_enabled_features;
        let bytes = std::slice::from_raw_parts(
            features as *const vk::PhysicalDeviceFeatures as *const u8,
            std::mem::size_of::<vk::PhysicalDeviceFeatures>(),
        );
        Some(bytes.to_vec())
    } else {
        None
    };

    let cmd = VulkanCommand::CreateDevice {
        physical_device: pd_handle,
        queue_create_infos,
        enabled_extensions,
        enabled_features,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::DeviceCreated { handle }) => {
            let dev_local_id = handle_store::store_device(handle);
            let dev_disp = DispatchableHandle::new(dev_local_id);
            *p_device = std::mem::transmute(dev_disp);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_INITIALIZATION_FAILED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyDevice(
    device: vk::Device,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if device == vk::Device::null() {
        return;
    }
    let disp = device.as_raw() as *mut DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    if let Some(handle) = handle_store::remove_device(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyDevice { device: handle });
    }
    DispatchableHandle::destroy(disp);
}

#[no_mangle]
pub unsafe extern "C" fn vkGetDeviceQueue(
    device: vk::Device,
    queue_family_index: u32,
    queue_index: u32,
    p_queue: *mut vk::Queue,
) {
    if p_queue.is_null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(local_id) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetDeviceQueue {
        device: dev_handle,
        queue_family_index,
        queue_index,
    };

    if let Ok(VulkanResponse::QueueRetrieved { handle }) = send_vulkan_command(cmd) {
        let q_local_id = handle_store::store_queue(handle);
        let q_disp = DispatchableHandle::new(q_local_id);
        *p_queue = std::mem::transmute(q_disp);
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDeviceWaitIdle(device: vk::Device) -> vk::Result {
    let disp = device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let cmd = VulkanCommand::DeviceWaitIdle {
        device: dev_handle,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_DEVICE_LOST,
    }
}

// ── Helpers ─────────────────────────────────────────────────

unsafe fn read_string_array(ptrs: *const *const std::os::raw::c_char, count: u32) -> Vec<String> {
    if ptrs.is_null() || count == 0 {
        return Vec::new();
    }
    (0..count as usize)
        .filter_map(|i| {
            let ptr = *ptrs.add(i);
            if ptr.is_null() {
                None
            } else {
                Some(
                    std::ffi::CStr::from_ptr(ptr)
                        .to_string_lossy()
                        .into_owned(),
                )
            }
        })
        .collect()
}
