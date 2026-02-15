//! Vulkan instance and enumeration functions.

use ash::vk;
use ash::vk::Handle;
use std::ffi::CStr;
use std::os::raw::c_char;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{VulkanCommand, VulkanResponse};

#[no_mangle]
pub unsafe extern "C" fn vkCreateInstance(
    p_create_info: *const vk::InstanceCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    if p_create_info.is_null() || p_instance.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let ci = &*p_create_info;

    let (app_name, app_version, engine_name, engine_version, api_version) =
        if !ci.p_application_info.is_null() {
            let ai = &*ci.p_application_info;
            let app = if !ai.p_application_name.is_null() {
                Some(CStr::from_ptr(ai.p_application_name).to_string_lossy().into_owned())
            } else {
                None
            };
            let eng = if !ai.p_engine_name.is_null() {
                Some(CStr::from_ptr(ai.p_engine_name).to_string_lossy().into_owned())
            } else {
                None
            };
            (app, ai.application_version, eng, ai.engine_version, ai.api_version)
        } else {
            (None, 0, None, 0, 0)
        };

    let enabled_extensions = read_string_array(ci.pp_enabled_extension_names, ci.enabled_extension_count);
    let enabled_layers = read_string_array(ci.pp_enabled_layer_names, ci.enabled_layer_count);

    let cmd = VulkanCommand::CreateInstance {
        app_name,
        app_version,
        engine_name,
        engine_version,
        api_version,
        enabled_extensions,
        enabled_layers,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::InstanceCreated { handle }) => {
            let local_id = handle_store::store_instance(handle);
            let disp = DispatchableHandle::new(local_id);
            *p_instance = std::mem::transmute(disp);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_INITIALIZATION_FAILED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyInstance(
    instance: vk::Instance,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if instance == vk::Instance::null() {
        return;
    }
    let disp = instance.as_raw() as *mut DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    if let Some(handle) = handle_store::remove_instance(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyInstance { instance: handle });
    }
    DispatchableHandle::destroy(disp);
}

#[no_mangle]
pub unsafe extern "C" fn vkEnumeratePhysicalDevices(
    instance: vk::Instance,
    p_physical_device_count: *mut u32,
    p_physical_devices: *mut vk::PhysicalDevice,
) -> vk::Result {
    if p_physical_device_count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let disp = instance.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let inst_handle = match handle_store::get_instance(local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_INITIALIZATION_FAILED,
    };

    let cmd = VulkanCommand::EnumeratePhysicalDevices { instance: inst_handle };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::PhysicalDevices { handles }) => {
            if p_physical_devices.is_null() {
                *p_physical_device_count = handles.len() as u32;
                return vk::Result::SUCCESS;
            }

            let requested = *p_physical_device_count as usize;
            let available = handles.len();
            let count = std::cmp::min(requested, available);

            for i in 0..count {
                let pd_local_id = handle_store::store_physical_device(handles[i]);
                // Physical devices are dispatchable handles
                let pd_disp = DispatchableHandle::new(pd_local_id);
                *p_physical_devices.add(i) = std::mem::transmute(pd_disp);
            }
            *p_physical_device_count = count as u32;

            if count < available {
                vk::Result::INCOMPLETE
            } else {
                vk::Result::SUCCESS
            }
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_INITIALIZATION_FAILED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkEnumerateInstanceExtensionProperties(
    p_layer_name: *const c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    if p_property_count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let layer_name = if !p_layer_name.is_null() {
        Some(CStr::from_ptr(p_layer_name).to_string_lossy().into_owned())
    } else {
        None
    };

    let cmd = VulkanCommand::EnumerateInstanceExtensionProperties { layer_name };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::ExtensionProperties { extensions }) => {
            if p_properties.is_null() {
                *p_property_count = extensions.len() as u32;
                return vk::Result::SUCCESS;
            }

            let requested = *p_property_count as usize;
            let count = std::cmp::min(requested, extensions.len());

            for i in 0..count {
                let prop = &mut *p_properties.add(i);
                write_c_string(&extensions[i].extension_name, &mut prop.extension_name);
                prop.spec_version = extensions[i].spec_version;
            }
            *p_property_count = count as u32;

            if count < extensions.len() {
                vk::Result::INCOMPLETE
            } else {
                vk::Result::SUCCESS
            }
        }
        _ => {
            *p_property_count = 0;
            vk::Result::SUCCESS
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkEnumerateInstanceLayerProperties(
    p_property_count: *mut u32,
    _p_properties: *mut vk::LayerProperties,
) -> vk::Result {
    if p_property_count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    // No layers
    *p_property_count = 0;
    vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn vkEnumerateDeviceExtensionProperties(
    physical_device: vk::PhysicalDevice,
    p_layer_name: *const c_char,
    p_property_count: *mut u32,
    p_properties: *mut vk::ExtensionProperties,
) -> vk::Result {
    if p_property_count.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }

    let disp = physical_device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);
    let pd_handle = match handle_store::get_physical_device(local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let layer_name = if !p_layer_name.is_null() {
        Some(CStr::from_ptr(p_layer_name).to_string_lossy().into_owned())
    } else {
        None
    };

    let cmd = VulkanCommand::EnumerateDeviceExtensionProperties {
        physical_device: pd_handle,
        layer_name,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::ExtensionProperties { extensions }) => {
            if p_properties.is_null() {
                *p_property_count = extensions.len() as u32;
                return vk::Result::SUCCESS;
            }

            let requested = *p_property_count as usize;
            let count = std::cmp::min(requested, extensions.len());

            for i in 0..count {
                let prop = &mut *p_properties.add(i);
                write_c_string(&extensions[i].extension_name, &mut prop.extension_name);
                prop.spec_version = extensions[i].spec_version;
            }
            *p_property_count = count as u32;

            if count < extensions.len() {
                vk::Result::INCOMPLETE
            } else {
                vk::Result::SUCCESS
            }
        }
        _ => {
            *p_property_count = 0;
            vk::Result::SUCCESS
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────

unsafe fn read_string_array(ptrs: *const *const c_char, count: u32) -> Vec<String> {
    if ptrs.is_null() || count == 0 {
        return Vec::new();
    }
    (0..count as usize)
        .filter_map(|i| {
            let ptr = *ptrs.add(i);
            if ptr.is_null() {
                None
            } else {
                Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
            }
        })
        .collect()
}

fn write_c_string(src: &str, dst: &mut [std::os::raw::c_char]) {
    let bytes = src.as_bytes();
    let len = std::cmp::min(bytes.len(), dst.len() - 1);
    for i in 0..len {
        dst[i] = bytes[i] as std::os::raw::c_char;
    }
    dst[len] = 0;
}
