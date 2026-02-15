//! Memory and buffer functions for the Vulkan ICD.

use ash::vk;
use ash::vk::Handle;
use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{MappedMemoryRange, VulkanCommand, VulkanResponse};

/// Shadow buffer info for a mapped memory region.
struct ShadowBuffer {
    ptr: *mut u8,
    size: usize,
    offset: u64,
    layout: std::alloc::Layout,
}

// Shadow buffers are only accessed under Mutex, so this is safe.
unsafe impl Send for ShadowBuffer {}

static SHADOW_BUFFERS: OnceLock<Mutex<HashMap<u64, ShadowBuffer>>> = OnceLock::new();

fn shadow_buffers() -> &'static Mutex<HashMap<u64, ShadowBuffer>> {
    SHADOW_BUFFERS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ── vkAllocateMemory ────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkAllocateMemory(
    device: vk::Device,
    p_allocate_info: *const vk::MemoryAllocateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_memory: *mut vk::DeviceMemory,
) -> vk::Result {
    if p_allocate_info.is_null() || p_memory.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ai = &*p_allocate_info;
    let cmd = VulkanCommand::AllocateMemory {
        device: dev_handle,
        alloc_size: ai.allocation_size,
        memory_type_index: ai.memory_type_index,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::MemoryAllocated { handle }) => {
            let local_id = handle_store::store_memory(handle);
            *p_memory = vk::DeviceMemory::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkFreeMemory(
    device: vk::Device,
    memory: vk::DeviceMemory,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if memory == vk::DeviceMemory::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = memory.as_raw();

    // Clean up shadow buffer if any
    if let Ok(mut bufs) = shadow_buffers().lock() {
        if let Some(sb) = bufs.remove(&local_id) {
            std::alloc::dealloc(sb.ptr, sb.layout);
        }
    }

    if let Some(handle) = handle_store::remove_memory(local_id) {
        let _ = send_vulkan_command(VulkanCommand::FreeMemory {
            device: dev_handle,
            memory: handle,
        });
    }
}

// ── vkMapMemory ─────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkMapMemory(
    device: vk::Device,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    flags: vk::MemoryMapFlags,
    pp_data: *mut *mut c_void,
) -> vk::Result {
    if pp_data.is_null() {
        return vk::Result::ERROR_MEMORY_MAP_FAILED;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let mem_local_id = memory.as_raw();
    let mem_handle = match handle_store::get_memory(mem_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_MEMORY_MAP_FAILED,
    };

    let cmd = VulkanCommand::MapMemory {
        device: dev_handle,
        memory: mem_handle,
        offset,
        size,
        flags: flags.as_raw(),
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::MemoryMapped { data }) => {
            let buf_size = data.len();
            let layout = match std::alloc::Layout::from_size_align(
                std::cmp::max(buf_size, 1),
                std::mem::align_of::<u64>(),
            ) {
                Ok(l) => l,
                Err(_) => return vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            };
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
            }

            // Copy server data into shadow buffer
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, buf_size);

            // Store shadow buffer
            if let Ok(mut bufs) = shadow_buffers().lock() {
                bufs.insert(
                    mem_local_id,
                    ShadowBuffer {
                        ptr,
                        size: buf_size,
                        offset,
                        layout,
                    },
                );
            }

            *pp_data = ptr as *mut c_void;
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_MEMORY_MAP_FAILED,
    }
}

// ── vkUnmapMemory ───────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkUnmapMemory(device: vk::Device, memory: vk::DeviceMemory) {
    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let mem_local_id = memory.as_raw();
    let mem_handle = match handle_store::get_memory(mem_local_id) {
        Some(h) => h,
        None => return,
    };

    // Read shadow buffer data and send to server
    let (written_data, offset) = if let Ok(mut bufs) = shadow_buffers().lock() {
        if let Some(sb) = bufs.remove(&mem_local_id) {
            let data = std::slice::from_raw_parts(sb.ptr, sb.size).to_vec();
            let offset = sb.offset;
            std::alloc::dealloc(sb.ptr, sb.layout);
            (Some(data), offset)
        } else {
            (None, 0)
        }
    } else {
        (None, 0)
    };

    let _ = send_vulkan_command(VulkanCommand::UnmapMemory {
        device: dev_handle,
        memory: mem_handle,
        written_data,
        offset,
    });
}

// ── vkFlushMappedMemoryRanges ───────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkFlushMappedMemoryRanges(
    device: vk::Device,
    memory_range_count: u32,
    p_memory_ranges: *const vk::MappedMemoryRange<'_>,
) -> vk::Result {
    if p_memory_ranges.is_null() || memory_range_count == 0 {
        return vk::Result::SUCCESS;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let mut ranges = Vec::new();
    let mut data_vec = Vec::new();

    for i in 0..memory_range_count as usize {
        let mr = &*p_memory_ranges.add(i);
        let mem_local_id = mr.memory.as_raw();

        let mem_handle = match handle_store::get_memory(mem_local_id) {
            Some(h) => h,
            None => continue,
        };

        // Read the flushed data from shadow buffer
        let range_data = if let Ok(bufs) = shadow_buffers().lock() {
            if let Some(sb) = bufs.get(&mem_local_id) {
                let flush_offset = (mr.offset - sb.offset) as usize;
                let flush_size = if mr.size == vk::WHOLE_SIZE {
                    sb.size - flush_offset
                } else {
                    mr.size as usize
                };
                let end = std::cmp::min(flush_offset + flush_size, sb.size);
                std::slice::from_raw_parts(sb.ptr.add(flush_offset), end - flush_offset)
                    .to_vec()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        ranges.push(MappedMemoryRange {
            memory: mem_handle,
            offset: mr.offset,
            size: mr.size,
        });
        data_vec.push(range_data);
    }

    let cmd = VulkanCommand::FlushMappedMemoryRanges {
        device: dev_handle,
        ranges,
        data: data_vec,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_MEMORY_MAP_FAILED,
    }
}

// ── vkInvalidateMappedMemoryRanges ──────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkInvalidateMappedMemoryRanges(
    device: vk::Device,
    memory_range_count: u32,
    p_memory_ranges: *const vk::MappedMemoryRange<'_>,
) -> vk::Result {
    if p_memory_ranges.is_null() || memory_range_count == 0 {
        return vk::Result::SUCCESS;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let mut ranges = Vec::new();
    for i in 0..memory_range_count as usize {
        let mr = &*p_memory_ranges.add(i);
        let mem_local_id = mr.memory.as_raw();
        let mem_handle = match handle_store::get_memory(mem_local_id) {
            Some(h) => h,
            None => continue,
        };
        ranges.push(MappedMemoryRange {
            memory: mem_handle,
            offset: mr.offset,
            size: mr.size,
        });
    }

    let cmd = VulkanCommand::InvalidateMappedMemoryRanges {
        device: dev_handle,
        ranges,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::InvalidatedData { range_data }) => {
            // Update shadow buffers with server data
            for (i, data) in range_data.iter().enumerate() {
                if i >= memory_range_count as usize {
                    break;
                }
                let mr = &*p_memory_ranges.add(i);
                let mem_local_id = mr.memory.as_raw();

                if let Ok(bufs) = shadow_buffers().lock() {
                    if let Some(sb) = bufs.get(&mem_local_id) {
                        let inv_offset = (mr.offset - sb.offset) as usize;
                        let copy_len = std::cmp::min(data.len(), sb.size - inv_offset);
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            sb.ptr.add(inv_offset),
                            copy_len,
                        );
                    }
                }
            }
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::SUCCESS,
    }
}

// ── Buffer ──────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateBuffer(
    device: vk::Device,
    p_create_info: *const vk::BufferCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_buffer: *mut vk::Buffer,
) -> vk::Result {
    if p_create_info.is_null() || p_buffer.is_null() {
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

    let cmd = VulkanCommand::CreateBuffer {
        device: dev_handle,
        size: ci.size,
        usage: ci.usage.as_raw(),
        sharing_mode: ci.sharing_mode.as_raw() as u32,
        queue_family_indices,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::BufferCreated { handle }) => {
            let local_id = handle_store::store_buffer(handle);
            *p_buffer = vk::Buffer::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyBuffer(
    device: vk::Device,
    buffer: vk::Buffer,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if buffer == vk::Buffer::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = buffer.as_raw();
    if let Some(handle) = handle_store::remove_buffer(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyBuffer {
            device: dev_handle,
            buffer: handle,
        });
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkBindBufferMemory(
    device: vk::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    memory_offset: vk::DeviceSize,
) -> vk::Result {
    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let buf_handle = match handle_store::get_buffer(buffer.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let mem_handle = match handle_store::get_memory(memory.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let cmd = VulkanCommand::BindBufferMemory {
        device: dev_handle,
        buffer: buf_handle,
        memory: mem_handle,
        memory_offset,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkGetBufferMemoryRequirements(
    device: vk::Device,
    buffer: vk::Buffer,
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

    let buf_handle = match handle_store::get_buffer(buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let cmd = VulkanCommand::GetBufferMemoryRequirements {
        device: dev_handle,
        buffer: buf_handle,
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
