//! Physical device query functions for the Vulkan ICD.

use ash::vk;
use ash::vk::Handle;
use std::os::raw::c_void;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{VulkanCommand, VulkanResponse};

// ── vkGetPhysicalDeviceProperties ───────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceProperties(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties,
) {
    if p_properties.is_null() {
        return;
    }

    let disp = physical_device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let pd_handle = match handle_store::get_physical_device(local_id) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetPhysicalDeviceProperties {
        physical_device: pd_handle,
    };

    if let Ok(VulkanResponse::PhysicalDeviceProperties {
        api_version,
        driver_version,
        vendor_id,
        device_id,
        device_type,
        device_name,
        pipeline_cache_uuid,
        limits_raw,
        sparse_properties_raw,
    }) = send_vulkan_command(cmd)
    {
        let props = &mut *p_properties;
        props.api_version = api_version;
        props.driver_version = driver_version;
        props.vendor_id = vendor_id;
        props.device_id = device_id;
        props.device_type = vk::PhysicalDeviceType::from_raw(device_type as i32);

        // Write device name
        let name_bytes = device_name.as_bytes();
        let len = std::cmp::min(name_bytes.len(), props.device_name.len() - 1);
        for i in 0..len {
            props.device_name[i] = name_bytes[i] as std::os::raw::c_char;
        }
        props.device_name[len] = 0;

        props.pipeline_cache_uuid = pipeline_cache_uuid;

        // Restore limits from raw bytes
        if limits_raw.len() == std::mem::size_of::<vk::PhysicalDeviceLimits>() {
            std::ptr::copy_nonoverlapping(
                limits_raw.as_ptr(),
                &mut props.limits as *mut vk::PhysicalDeviceLimits as *mut u8,
                limits_raw.len(),
            );
        }

        // Restore sparse properties from raw bytes
        if sparse_properties_raw.len()
            == std::mem::size_of::<vk::PhysicalDeviceSparseProperties>()
        {
            std::ptr::copy_nonoverlapping(
                sparse_properties_raw.as_ptr(),
                &mut props.sparse_properties as *mut vk::PhysicalDeviceSparseProperties
                    as *mut u8,
                sparse_properties_raw.len(),
            );
        }
    }
}

// ── vkGetPhysicalDeviceProperties2 ─────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceProperties2(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties2<'_>,
) {
    if p_properties.is_null() {
        return;
    }
    // Fill the core properties struct; ignore pNext chain for Phase 3
    vkGetPhysicalDeviceProperties(physical_device, &mut (*p_properties).properties);
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceProperties2KHR(
    physical_device: vk::PhysicalDevice,
    p_properties: *mut vk::PhysicalDeviceProperties2<'_>,
) {
    vkGetPhysicalDeviceProperties2(physical_device, p_properties);
}

// ── vkGetPhysicalDeviceFeatures ────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceFeatures(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures,
) {
    if p_features.is_null() {
        return;
    }

    let disp = physical_device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let pd_handle = match handle_store::get_physical_device(local_id) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetPhysicalDeviceFeatures {
        physical_device: pd_handle,
    };

    if let Ok(VulkanResponse::PhysicalDeviceFeatures { features_raw }) =
        send_vulkan_command(cmd)
    {
        if features_raw.len() == std::mem::size_of::<vk::PhysicalDeviceFeatures>() {
            std::ptr::copy_nonoverlapping(
                features_raw.as_ptr(),
                p_features as *mut u8,
                features_raw.len(),
            );
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceFeatures2(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    if p_features.is_null() {
        return;
    }
    vkGetPhysicalDeviceFeatures(physical_device, &mut (*p_features).features);
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceFeatures2KHR(
    physical_device: vk::PhysicalDevice,
    p_features: *mut vk::PhysicalDeviceFeatures2<'_>,
) {
    vkGetPhysicalDeviceFeatures2(physical_device, p_features);
}

// ── vkGetPhysicalDeviceMemoryProperties ────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceMemoryProperties(
    physical_device: vk::PhysicalDevice,
    p_memory_properties: *mut vk::PhysicalDeviceMemoryProperties,
) {
    if p_memory_properties.is_null() {
        return;
    }

    let disp = physical_device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let pd_handle = match handle_store::get_physical_device(local_id) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetPhysicalDeviceMemoryProperties {
        physical_device: pd_handle,
    };

    if let Ok(VulkanResponse::PhysicalDeviceMemoryProperties {
        memory_type_count,
        memory_types,
        memory_heap_count,
        memory_heaps,
    }) = send_vulkan_command(cmd)
    {
        let mem_props = &mut *p_memory_properties;
        mem_props.memory_type_count = memory_type_count;
        for (i, mt) in memory_types.iter().enumerate() {
            if i >= vk::MAX_MEMORY_TYPES {
                break;
            }
            mem_props.memory_types[i] = vk::MemoryType {
                property_flags: vk::MemoryPropertyFlags::from_raw(mt.property_flags),
                heap_index: mt.heap_index,
            };
        }
        mem_props.memory_heap_count = memory_heap_count;
        for (i, mh) in memory_heaps.iter().enumerate() {
            if i >= vk::MAX_MEMORY_HEAPS {
                break;
            }
            mem_props.memory_heaps[i] = vk::MemoryHeap {
                size: mh.size,
                flags: vk::MemoryHeapFlags::from_raw(mh.flags),
            };
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceMemoryProperties2(
    physical_device: vk::PhysicalDevice,
    p_memory_properties: *mut vk::PhysicalDeviceMemoryProperties2<'_>,
) {
    if p_memory_properties.is_null() {
        return;
    }
    vkGetPhysicalDeviceMemoryProperties(
        physical_device,
        &mut (*p_memory_properties).memory_properties,
    );
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceMemoryProperties2KHR(
    physical_device: vk::PhysicalDevice,
    p_memory_properties: *mut vk::PhysicalDeviceMemoryProperties2<'_>,
) {
    vkGetPhysicalDeviceMemoryProperties2(physical_device, p_memory_properties);
}

// ── vkGetPhysicalDeviceQueueFamilyProperties ───────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceQueueFamilyProperties(
    physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties,
) {
    if p_queue_family_property_count.is_null() {
        return;
    }

    let disp = physical_device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let pd_handle = match handle_store::get_physical_device(local_id) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetPhysicalDeviceQueueFamilyProperties {
        physical_device: pd_handle,
    };

    if let Ok(VulkanResponse::QueueFamilyProperties { families }) =
        send_vulkan_command(cmd)
    {
        if p_queue_family_properties.is_null() {
            *p_queue_family_property_count = families.len() as u32;
            return;
        }

        let requested = *p_queue_family_property_count as usize;
        let count = std::cmp::min(requested, families.len());

        for i in 0..count {
            let qf = &families[i];
            let dst = &mut *p_queue_family_properties.add(i);
            dst.queue_flags = vk::QueueFlags::from_raw(qf.queue_flags);
            dst.queue_count = qf.queue_count;
            dst.timestamp_valid_bits = qf.timestamp_valid_bits;
            dst.min_image_transfer_granularity = vk::Extent3D {
                width: qf.min_image_transfer_granularity[0],
                height: qf.min_image_transfer_granularity[1],
                depth: qf.min_image_transfer_granularity[2],
            };
        }
        *p_queue_family_property_count = count as u32;
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceQueueFamilyProperties2(
    physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties2<'_>,
) {
    if p_queue_family_property_count.is_null() {
        return;
    }

    // For count-only queries, delegate to the core function
    if p_queue_family_properties.is_null() {
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device,
            p_queue_family_property_count,
            std::ptr::null_mut(),
        );
        return;
    }

    // Get the count first
    let mut count = *p_queue_family_property_count;
    let mut core_props = vec![vk::QueueFamilyProperties::default(); count as usize];
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device,
        &mut count,
        core_props.as_mut_ptr(),
    );

    let fill = std::cmp::min(count as usize, *p_queue_family_property_count as usize);
    for i in 0..fill {
        (*p_queue_family_properties.add(i)).queue_family_properties = core_props[i];
    }
    *p_queue_family_property_count = count;
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceQueueFamilyProperties2KHR(
    physical_device: vk::PhysicalDevice,
    p_queue_family_property_count: *mut u32,
    p_queue_family_properties: *mut vk::QueueFamilyProperties2<'_>,
) {
    vkGetPhysicalDeviceQueueFamilyProperties2(
        physical_device,
        p_queue_family_property_count,
        p_queue_family_properties,
    );
}

// ── vkGetPhysicalDeviceFormatProperties ────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceFormatProperties(
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    p_format_properties: *mut vk::FormatProperties,
) {
    if p_format_properties.is_null() {
        return;
    }

    let disp = physical_device.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(disp);

    let pd_handle = match handle_store::get_physical_device(local_id) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetPhysicalDeviceFormatProperties {
        physical_device: pd_handle,
        format: format.as_raw(),
    };

    if let Ok(VulkanResponse::FormatProperties {
        linear_tiling_features,
        optimal_tiling_features,
        buffer_features,
    }) = send_vulkan_command(cmd)
    {
        let fp = &mut *p_format_properties;
        fp.linear_tiling_features = vk::FormatFeatureFlags::from_raw(linear_tiling_features);
        fp.optimal_tiling_features =
            vk::FormatFeatureFlags::from_raw(optimal_tiling_features);
        fp.buffer_features = vk::FormatFeatureFlags::from_raw(buffer_features);
    }
}

// ── vkGetPhysicalDeviceFormatProperties2 ───────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceFormatProperties2(
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    p_format_properties: *mut vk::FormatProperties2<'_>,
) {
    if p_format_properties.is_null() {
        return;
    }
    vkGetPhysicalDeviceFormatProperties(
        physical_device,
        format,
        &mut (*p_format_properties).format_properties,
    );
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceFormatProperties2KHR(
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    p_format_properties: *mut vk::FormatProperties2<'_>,
) {
    vkGetPhysicalDeviceFormatProperties2(physical_device, format, p_format_properties);
}

// ── Sparse image support stubs ─────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceSparseImageFormatProperties(
    _physical_device: vk::PhysicalDevice,
    _format: vk::Format,
    _type_: vk::ImageType,
    _samples: vk::SampleCountFlags,
    _usage: vk::ImageUsageFlags,
    _tiling: vk::ImageTiling,
    p_property_count: *mut u32,
    _p_properties: *mut c_void,
) {
    if !p_property_count.is_null() {
        *p_property_count = 0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceSparseImageFormatProperties2(
    _physical_device: vk::PhysicalDevice,
    _p_format_info: *const c_void,
    p_property_count: *mut u32,
    _p_properties: *mut c_void,
) {
    if !p_property_count.is_null() {
        *p_property_count = 0;
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkGetPhysicalDeviceSparseImageFormatProperties2KHR(
    physical_device: vk::PhysicalDevice,
    p_format_info: *const c_void,
    p_property_count: *mut u32,
    p_properties: *mut c_void,
) {
    vkGetPhysicalDeviceSparseImageFormatProperties2(
        physical_device,
        p_format_info,
        p_property_count,
        p_properties,
    );
}
