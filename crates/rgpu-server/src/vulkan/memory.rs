use ash::vk;
use tracing::debug;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::{MappedMemoryInfo, VulkanExecutor};

impl VulkanExecutor {
    pub(crate) fn handle_allocate_memory(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, alloc_size, memory_type_index) = match cmd {
            VulkanCommand::AllocateMemory {
                device,
                alloc_size,
                memory_type_index,
            } => (device, alloc_size, memory_type_index),
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
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(alloc_size)
            .memory_type_index(memory_type_index);
        match unsafe { dev.allocate_memory(&alloc_info, None) } {
            Ok(memory) => {
                let handle = session.alloc_handle(ResourceType::VkDeviceMemory);
                self.memory_handles.insert(handle, memory);
                self.memory_to_device.insert(handle, device);
                debug!("allocated {} bytes of device memory: {:?}", alloc_size, handle);
                VulkanResponse::MemoryAllocated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_free_memory(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, memory) = match cmd {
            VulkanCommand::FreeMemory { device, memory } => (device, memory),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, mem)) = self.memory_handles.remove(&memory) {
            // Unmap if still mapped
            if let Some((_, info)) = self.memory_info.remove(&memory) {
                unsafe { dev.unmap_memory(mem) };
                let _ = info;
            }
            unsafe { dev.free_memory(mem, None) };
            self.memory_to_device.remove(&memory);
            session.remove_handle(&memory);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_map_memory(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, memory, offset, size) = match cmd {
            VulkanCommand::MapMemory {
                device,
                memory,
                offset,
                size,
                flags: _,
            } => (device, memory, offset, size),
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
        let mem = match self.memory_handles.get(&memory) {
            Some(m) => *m.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_MEMORY_MAP_FAILED.as_raw(),
                    message: "invalid memory handle".to_string(),
                }
            }
        };

        let map_size = if size == vk::WHOLE_SIZE { vk::WHOLE_SIZE } else { size };

        match unsafe {
            dev.map_memory(mem, offset, map_size, vk::MemoryMapFlags::empty())
        } {
            Ok(ptr) => {
                // Read current contents to send to client
                let actual_size = if map_size == vk::WHOLE_SIZE {
                    // We don't know the allocation size here, use a reasonable default
                    // The client should know the size from the allocation
                    0
                } else {
                    map_size as usize
                };

                let data = if actual_size > 0 {
                    unsafe {
                        std::slice::from_raw_parts(ptr as *const u8, actual_size).to_vec()
                    }
                } else {
                    Vec::new()
                };

                self.memory_info.insert(
                    memory,
                    MappedMemoryInfo {
                        offset,
                        _size: map_size,
                        ptr,
                    },
                );

                VulkanResponse::MemoryMapped { data }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_unmap_memory(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, memory, written_data, offset) = match cmd {
            VulkanCommand::UnmapMemory {
                device,
                memory,
                written_data,
                offset,
            } => (device, memory, written_data, offset),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        let mem = match self.memory_handles.get(&memory) {
            Some(m) => *m.value(),
            None => return VulkanResponse::Success,
        };

        // Write client data to mapped memory before unmapping
        if let Some(data) = written_data {
            if let Some(info) = self.memory_info.get(&memory) {
                let write_offset = offset.saturating_sub(info.offset) as usize;
                unsafe {
                    let dst = (info.ptr as *mut u8).add(write_offset);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                }
            }
        }

        unsafe { dev.unmap_memory(mem) };
        self.memory_info.remove(&memory);
        VulkanResponse::Success
    }

    pub(crate) fn handle_flush_mapped_memory_ranges(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, ranges, data) = match cmd {
            VulkanCommand::FlushMappedMemoryRanges {
                device,
                ranges,
                data,
            } => (device, ranges, data),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };

        // Write data to mapped regions
        for (i, range) in ranges.iter().enumerate() {
            if let Some(info) = self.memory_info.get(&range.memory) {
                if i < data.len() {
                    let write_offset = range.offset.saturating_sub(info.offset) as usize;
                    unsafe {
                        let dst = (info.ptr as *mut u8).add(write_offset);
                        std::ptr::copy_nonoverlapping(
                            data[i].as_ptr(),
                            dst,
                            data[i].len(),
                        );
                    }
                }
            }
        }

        // Build vk::MappedMemoryRange array
        let vk_ranges: Vec<vk::MappedMemoryRange> = ranges
            .iter()
            .filter_map(|r| {
                self.memory_handles.get(&r.memory).map(|m| {
                    vk::MappedMemoryRange::default()
                        .memory(*m.value())
                        .offset(r.offset)
                        .size(r.size)
                })
            })
            .collect();

        if !vk_ranges.is_empty() {
            match unsafe { dev.flush_mapped_memory_ranges(&vk_ranges) } {
                Ok(()) => {}
                Err(e) => return Self::vk_err(e),
            }
        }

        VulkanResponse::Success
    }

    pub(crate) fn handle_invalidate_mapped_memory_ranges(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, ranges) = match cmd {
            VulkanCommand::InvalidateMappedMemoryRanges { device, ranges } => (device, ranges),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };

        let vk_ranges: Vec<vk::MappedMemoryRange> = ranges
            .iter()
            .filter_map(|r| {
                self.memory_handles.get(&r.memory).map(|m| {
                    vk::MappedMemoryRange::default()
                        .memory(*m.value())
                        .offset(r.offset)
                        .size(r.size)
                })
            })
            .collect();

        if !vk_ranges.is_empty() {
            if let Err(e) = unsafe { dev.invalidate_mapped_memory_ranges(&vk_ranges) } {
                return Self::vk_err(e);
            }
        }

        // Read data from mapped regions
        let mut range_data = Vec::new();
        for range in &ranges {
            if let Some(info) = self.memory_info.get(&range.memory) {
                let read_offset = range.offset.saturating_sub(info.offset) as usize;
                let read_size = range.size as usize;
                let data = unsafe {
                    let src = (info.ptr as *const u8).add(read_offset);
                    std::slice::from_raw_parts(src, read_size).to_vec()
                };
                range_data.push(data);
            } else {
                range_data.push(Vec::new());
            }
        }

        VulkanResponse::InvalidatedData { range_data }
    }

    pub(crate) fn handle_get_buffer_memory_requirements(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, buffer) = match cmd {
            VulkanCommand::GetBufferMemoryRequirements { device, buffer } => (device, buffer),
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
        let reqs = unsafe { dev.get_buffer_memory_requirements(buf) };
        VulkanResponse::MemoryRequirements {
            size: reqs.size,
            alignment: reqs.alignment,
            memory_type_bits: reqs.memory_type_bits,
        }
    }

    pub(crate) fn handle_get_image_memory_requirements(
        &self,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, image) = match cmd {
            VulkanCommand::GetImageMemoryRequirements { device, image } => (device, image),
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
        let reqs = unsafe { dev.get_image_memory_requirements(img) };
        VulkanResponse::MemoryRequirements {
            size: reqs.size,
            alignment: reqs.alignment,
            memory_type_bits: reqs.memory_type_bits,
        }
    }
}
