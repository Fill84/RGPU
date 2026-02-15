//! Descriptor pool, descriptor set allocation, and update functions.

use ash::vk;
use ash::vk::Handle;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{
    SerializedDescriptorBufferInfo, SerializedDescriptorPoolSize, SerializedWriteDescriptorSet,
    VulkanCommand, VulkanResponse,
};

// ── Descriptor Pool ─────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateDescriptorPool(
    device: vk::Device,
    p_create_info: *const vk::DescriptorPoolCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_descriptor_pool: *mut vk::DescriptorPool,
) -> vk::Result {
    if p_create_info.is_null() || p_descriptor_pool.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;
    let mut pool_sizes = Vec::new();
    if !ci.p_pool_sizes.is_null() {
        for i in 0..ci.pool_size_count as usize {
            let ps = &*ci.p_pool_sizes.add(i);
            pool_sizes.push(SerializedDescriptorPoolSize {
                descriptor_type: ps.ty.as_raw(),
                descriptor_count: ps.descriptor_count,
            });
        }
    }

    let cmd = VulkanCommand::CreateDescriptorPool {
        device: dev_handle,
        max_sets: ci.max_sets,
        pool_sizes,
        flags: ci.flags.as_raw(),
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::DescriptorPoolCreated { handle }) => {
            let local_id = handle_store::store_desc_pool(handle);
            *p_descriptor_pool = vk::DescriptorPool::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyDescriptorPool(
    device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if descriptor_pool == vk::DescriptorPool::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = descriptor_pool.as_raw();
    if let Some(handle) = handle_store::remove_desc_pool(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyDescriptorPool {
            device: dev_handle,
            pool: handle,
        });
    }
}

// ── Descriptor Set Allocation ───────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkAllocateDescriptorSets(
    device: vk::Device,
    p_allocate_info: *const vk::DescriptorSetAllocateInfo<'_>,
    p_descriptor_sets: *mut vk::DescriptorSet,
) -> vk::Result {
    if p_allocate_info.is_null() || p_descriptor_sets.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ai = &*p_allocate_info;

    // Resolve descriptor pool handle
    let pool_local_id = ai.descriptor_pool.as_raw();
    let pool_handle = match handle_store::get_desc_pool(pool_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    // Resolve set layout handles
    let mut set_layouts = Vec::new();
    if !ai.p_set_layouts.is_null() {
        for i in 0..ai.descriptor_set_count as usize {
            let sl = *ai.p_set_layouts.add(i);
            match handle_store::get_desc_set_layout(sl.as_raw()) {
                Some(h) => set_layouts.push(h),
                None => return vk::Result::ERROR_UNKNOWN,
            }
        }
    }

    let cmd = VulkanCommand::AllocateDescriptorSets {
        device: dev_handle,
        descriptor_pool: pool_handle,
        set_layouts,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::DescriptorSetsAllocated { handles }) => {
            for (i, h) in handles.iter().enumerate() {
                let local_id = handle_store::store_desc_set(*h);
                *p_descriptor_sets.add(i) = vk::DescriptorSet::from_raw(local_id);
            }
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkFreeDescriptorSets(
    device: vk::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
) -> vk::Result {
    if p_descriptor_sets.is_null() || descriptor_set_count == 0 {
        return vk::Result::SUCCESS;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let pool_handle = match handle_store::get_desc_pool(descriptor_pool.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let mut net_handles = Vec::new();
    for i in 0..descriptor_set_count as usize {
        let ds = *p_descriptor_sets.add(i);
        if let Some(h) = handle_store::remove_desc_set(ds.as_raw()) {
            net_handles.push(h);
        }
    }

    let cmd = VulkanCommand::FreeDescriptorSets {
        device: dev_handle,
        descriptor_pool: pool_handle,
        descriptor_sets: net_handles,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::SUCCESS,
    }
}

// ── Update Descriptor Sets ──────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkUpdateDescriptorSets(
    device: vk::Device,
    descriptor_write_count: u32,
    p_descriptor_writes: *const vk::WriteDescriptorSet<'_>,
    _descriptor_copy_count: u32,
    _p_descriptor_copies: *const vk::CopyDescriptorSet<'_>,
) {
    if p_descriptor_writes.is_null() || descriptor_write_count == 0 {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let mut writes = Vec::new();
    for i in 0..descriptor_write_count as usize {
        let w = &*p_descriptor_writes.add(i);

        // Resolve dest set handle
        let dst_set = match handle_store::get_desc_set(w.dst_set.as_raw()) {
            Some(h) => h,
            None => continue,
        };

        // Read buffer infos
        let mut buffer_infos = Vec::new();
        if !w.p_buffer_info.is_null() {
            for j in 0..w.descriptor_count as usize {
                let bi = &*w.p_buffer_info.add(j);
                let buf_handle = match handle_store::get_buffer(bi.buffer.as_raw()) {
                    Some(h) => h,
                    None => continue,
                };
                buffer_infos.push(SerializedDescriptorBufferInfo {
                    buffer: buf_handle,
                    offset: bi.offset,
                    range: bi.range,
                });
            }
        }

        writes.push(SerializedWriteDescriptorSet {
            dst_set,
            dst_binding: w.dst_binding,
            dst_array_element: w.dst_array_element,
            descriptor_type: w.descriptor_type.as_raw(),
            buffer_infos,
        });
    }

    let cmd = VulkanCommand::UpdateDescriptorSets {
        device: dev_handle,
        writes,
    };

    let _ = send_vulkan_command(cmd);
}
