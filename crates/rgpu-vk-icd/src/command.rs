//! Command pool, command buffer, and recording functions.
//! Command buffer recording is done client-side (batched) and sent at submit time.

use ash::vk;
use ash::vk::Handle;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{
    RecordedCommand, SerializedBufferCopy, SerializedBufferImageCopy, SerializedBufferMemoryBarrier,
    SerializedClearValue, SerializedImageMemoryBarrier, SerializedImageSubresourceLayers,
    SerializedImageSubresourceRange, SerializedMemoryBarrier, SerializedRect2D,
    SerializedViewport, VulkanCommand, VulkanResponse,
};

/// Per-command-buffer recording state.
struct CommandBufferState {
    recording: bool,
    commands: Vec<RecordedCommand>,
}

/// Map from local command buffer ID to its recording state.
static CMD_BUF_STATES: OnceLock<Mutex<HashMap<u64, CommandBufferState>>> = OnceLock::new();

fn cmd_buf_states() -> &'static Mutex<HashMap<u64, CommandBufferState>> {
    CMD_BUF_STATES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get recorded commands for a command buffer and clear them.
pub fn take_recorded_commands(local_id: u64) -> Vec<RecordedCommand> {
    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            return std::mem::take(&mut state.commands);
        }
    }
    Vec::new()
}

// ── Command Pool ────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateCommandPool(
    device: vk::Device,
    p_create_info: *const vk::CommandPoolCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_command_pool: *mut vk::CommandPool,
) -> vk::Result {
    if p_create_info.is_null() || p_command_pool.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;
    let cmd = VulkanCommand::CreateCommandPool {
        device: dev_handle,
        queue_family_index: ci.queue_family_index,
        flags: ci.flags.as_raw(),
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::CommandPoolCreated { handle }) => {
            let local_id = handle_store::store_cmd_pool(handle);
            *p_command_pool = vk::CommandPool::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyCommandPool(
    device: vk::Device,
    command_pool: vk::CommandPool,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if command_pool == vk::CommandPool::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = command_pool.as_raw();
    if let Some(handle) = handle_store::remove_cmd_pool(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyCommandPool {
            device: dev_handle,
            command_pool: handle,
        });
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkResetCommandPool(
    device: vk::Device,
    command_pool: vk::CommandPool,
    flags: vk::CommandPoolResetFlags,
) -> vk::Result {
    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let pool_handle = match handle_store::get_cmd_pool(command_pool.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let cmd = VulkanCommand::ResetCommandPool {
        device: dev_handle,
        command_pool: pool_handle,
        flags: flags.as_raw(),
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

// ── Command Buffer Allocation ───────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkAllocateCommandBuffers(
    device: vk::Device,
    p_allocate_info: *const vk::CommandBufferAllocateInfo<'_>,
    p_command_buffers: *mut vk::CommandBuffer,
) -> vk::Result {
    if p_allocate_info.is_null() || p_command_buffers.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ai = &*p_allocate_info;

    let pool_handle = match handle_store::get_cmd_pool(ai.command_pool.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let cmd = VulkanCommand::AllocateCommandBuffers {
        device: dev_handle,
        command_pool: pool_handle,
        level: ai.level.as_raw() as u32,
        count: ai.command_buffer_count,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::CommandBuffersAllocated { handles }) => {
            for (i, h) in handles.iter().enumerate() {
                let local_id = handle_store::store_cmd_buffer(*h);
                // Command buffers are dispatchable handles
                let cb_disp = DispatchableHandle::new(local_id);
                *p_command_buffers.add(i) = std::mem::transmute(cb_disp);

                // Initialize recording state
                if let Ok(mut states) = cmd_buf_states().lock() {
                    states.insert(
                        local_id,
                        CommandBufferState {
                            recording: false,
                            commands: Vec::new(),
                        },
                    );
                }
            }
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkFreeCommandBuffers(
    device: vk::Device,
    command_pool: vk::CommandPool,
    command_buffer_count: u32,
    p_command_buffers: *const vk::CommandBuffer,
) {
    if p_command_buffers.is_null() || command_buffer_count == 0 {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let pool_handle = match handle_store::get_cmd_pool(command_pool.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let mut net_handles = Vec::new();
    for i in 0..command_buffer_count as usize {
        let cb = *p_command_buffers.add(i);
        if cb == vk::CommandBuffer::null() {
            continue;
        }
        let cb_disp = cb.as_raw() as *mut DispatchableHandle;
        let local_id = DispatchableHandle::get_id(cb_disp);

        // Clean up recording state
        if let Ok(mut states) = cmd_buf_states().lock() {
            states.remove(&local_id);
        }

        if let Some(h) = handle_store::remove_cmd_buffer(local_id) {
            net_handles.push(h);
        }
        DispatchableHandle::destroy(cb_disp);
    }

    let cmd = VulkanCommand::FreeCommandBuffers {
        device: dev_handle,
        command_pool: pool_handle,
        command_buffers: net_handles,
    };

    let _ = send_vulkan_command(cmd);
}

// ── Command Buffer Recording (client-side batching) ─────────

#[no_mangle]
pub unsafe extern "C" fn vkBeginCommandBuffer(
    command_buffer: vk::CommandBuffer,
    _p_begin_info: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.recording = true;
            state.commands.clear();
        }
    }

    vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn vkEndCommandBuffer(
    command_buffer: vk::CommandBuffer,
) -> vk::Result {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.recording = false;
        }
    }

    vk::Result::SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn vkResetCommandBuffer(
    command_buffer: vk::CommandBuffer,
    _flags: vk::CommandBufferResetFlags,
) -> vk::Result {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.recording = false;
            state.commands.clear();
        }
    }

    vk::Result::SUCCESS
}

// ── vkCmd* recording functions ──────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCmdBindPipeline(
    command_buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    pipeline: vk::Pipeline,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let pipeline_handle = match handle_store::get_pipeline(pipeline.as_raw()) {
        Some(h) => h,
        None => return,
    };

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::BindPipeline {
                pipeline_bind_point: pipeline_bind_point.as_raw() as u32,
                pipeline: pipeline_handle,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdBindDescriptorSets(
    command_buffer: vk::CommandBuffer,
    pipeline_bind_point: vk::PipelineBindPoint,
    layout: vk::PipelineLayout,
    first_set: u32,
    descriptor_set_count: u32,
    p_descriptor_sets: *const vk::DescriptorSet,
    dynamic_offset_count: u32,
    p_dynamic_offsets: *const u32,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let layout_handle = match handle_store::get_pipeline_layout(layout.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let mut descriptor_sets = Vec::new();
    if !p_descriptor_sets.is_null() {
        for i in 0..descriptor_set_count as usize {
            let ds = *p_descriptor_sets.add(i);
            match handle_store::get_desc_set(ds.as_raw()) {
                Some(h) => descriptor_sets.push(h),
                None => return,
            }
        }
    }

    let dynamic_offsets = if !p_dynamic_offsets.is_null() && dynamic_offset_count > 0 {
        std::slice::from_raw_parts(p_dynamic_offsets, dynamic_offset_count as usize).to_vec()
    } else {
        Vec::new()
    };

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::BindDescriptorSets {
                pipeline_bind_point: pipeline_bind_point.as_raw() as u32,
                layout: layout_handle,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdDispatch(
    command_buffer: vk::CommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::Dispatch {
                group_count_x,
                group_count_y,
                group_count_z,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdPipelineBarrier(
    command_buffer: vk::CommandBuffer,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barrier_count: u32,
    p_memory_barriers: *const vk::MemoryBarrier<'_>,
    buffer_memory_barrier_count: u32,
    p_buffer_memory_barriers: *const vk::BufferMemoryBarrier<'_>,
    image_memory_barrier_count: u32,
    p_image_memory_barriers: *const vk::ImageMemoryBarrier<'_>,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let mut memory_barriers = Vec::new();
    if !p_memory_barriers.is_null() {
        for i in 0..memory_barrier_count as usize {
            let mb = &*p_memory_barriers.add(i);
            memory_barriers.push(SerializedMemoryBarrier {
                src_access_mask: mb.src_access_mask.as_raw(),
                dst_access_mask: mb.dst_access_mask.as_raw(),
            });
        }
    }

    let mut buffer_memory_barriers_vec = Vec::new();
    if !p_buffer_memory_barriers.is_null() {
        for i in 0..buffer_memory_barrier_count as usize {
            let bmb = &*p_buffer_memory_barriers.add(i);
            let buf_handle = match handle_store::get_buffer(bmb.buffer.as_raw()) {
                Some(h) => h,
                None => continue,
            };
            buffer_memory_barriers_vec.push(SerializedBufferMemoryBarrier {
                src_access_mask: bmb.src_access_mask.as_raw(),
                dst_access_mask: bmb.dst_access_mask.as_raw(),
                src_queue_family_index: bmb.src_queue_family_index,
                dst_queue_family_index: bmb.dst_queue_family_index,
                buffer: buf_handle,
                offset: bmb.offset,
                size: bmb.size,
            });
        }
    }

    let mut image_memory_barriers_vec = Vec::new();
    if !p_image_memory_barriers.is_null() {
        for i in 0..image_memory_barrier_count as usize {
            let imb = &*p_image_memory_barriers.add(i);
            let img_handle = match handle_store::get_image(imb.image.as_raw()) {
                Some(h) => h,
                None => continue,
            };
            image_memory_barriers_vec.push(SerializedImageMemoryBarrier {
                src_access_mask: imb.src_access_mask.as_raw(),
                dst_access_mask: imb.dst_access_mask.as_raw(),
                old_layout: imb.old_layout.as_raw(),
                new_layout: imb.new_layout.as_raw(),
                src_queue_family_index: imb.src_queue_family_index,
                dst_queue_family_index: imb.dst_queue_family_index,
                image: img_handle,
                subresource_range: SerializedImageSubresourceRange {
                    aspect_mask: imb.subresource_range.aspect_mask.as_raw(),
                    base_mip_level: imb.subresource_range.base_mip_level,
                    level_count: imb.subresource_range.level_count,
                    base_array_layer: imb.subresource_range.base_array_layer,
                    layer_count: imb.subresource_range.layer_count,
                },
            });
        }
    }

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::PipelineBarrier {
                src_stage_mask: src_stage_mask.as_raw(),
                dst_stage_mask: dst_stage_mask.as_raw(),
                dependency_flags: dependency_flags.as_raw(),
                memory_barriers,
                buffer_memory_barriers: buffer_memory_barriers_vec,
                image_memory_barriers: image_memory_barriers_vec,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdCopyBuffer(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_buffer: vk::Buffer,
    region_count: u32,
    p_regions: *const vk::BufferCopy,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let src_handle = match handle_store::get_buffer(src_buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };
    let dst_handle = match handle_store::get_buffer(dst_buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let mut regions = Vec::new();
    if !p_regions.is_null() {
        for i in 0..region_count as usize {
            let r = &*p_regions.add(i);
            regions.push(SerializedBufferCopy {
                src_offset: r.src_offset,
                dst_offset: r.dst_offset,
                size: r.size,
            });
        }
    }

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::CopyBuffer {
                src: src_handle,
                dst: dst_handle,
                regions,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdFillBuffer(
    command_buffer: vk::CommandBuffer,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
    data: u32,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let buf_handle = match handle_store::get_buffer(dst_buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::FillBuffer {
                buffer: buf_handle,
                offset: dst_offset,
                size,
                data,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdUpdateBuffer(
    command_buffer: vk::CommandBuffer,
    dst_buffer: vk::Buffer,
    dst_offset: vk::DeviceSize,
    data_size: vk::DeviceSize,
    p_data: *const std::os::raw::c_void,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let buf_handle = match handle_store::get_buffer(dst_buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let data = if !p_data.is_null() && data_size > 0 {
        std::slice::from_raw_parts(p_data as *const u8, data_size as usize).to_vec()
    } else {
        Vec::new()
    };

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::UpdateBuffer {
                buffer: buf_handle,
                offset: dst_offset,
                data,
            });
        }
    }
}

// ── Phase 5: Rendering recording functions ──────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCmdBeginRenderPass(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo<'_>,
    contents: vk::SubpassContents,
) {
    if p_render_pass_begin.is_null() {
        return;
    }

    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let rpi = &*p_render_pass_begin;

    let rp_handle = match handle_store::get_render_pass(rpi.render_pass.as_raw()) {
        Some(h) => h,
        None => return,
    };
    let fb_handle = match handle_store::get_framebuffer(rpi.framebuffer.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let clear_values: Vec<SerializedClearValue> = if !rpi.p_clear_values.is_null()
        && rpi.clear_value_count > 0
    {
        std::slice::from_raw_parts(rpi.p_clear_values, rpi.clear_value_count as usize)
            .iter()
            .map(|cv| SerializedClearValue {
                data: std::mem::transmute::<vk::ClearValue, [u8; 16]>(*cv),
            })
            .collect()
    } else {
        Vec::new()
    };

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::BeginRenderPass {
                render_pass: rp_handle,
                framebuffer: fb_handle,
                render_area: SerializedRect2D {
                    offset: [rpi.render_area.offset.x, rpi.render_area.offset.y],
                    extent: [rpi.render_area.extent.width, rpi.render_area.extent.height],
                },
                clear_values,
                contents: contents.as_raw() as u32,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdEndRenderPass(command_buffer: vk::CommandBuffer) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::EndRenderPass);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdDraw(
    command_buffer: vk::CommandBuffer,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdDrawIndexed(
    command_buffer: vk::CommandBuffer,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdBindVertexBuffers(
    command_buffer: vk::CommandBuffer,
    first_binding: u32,
    binding_count: u32,
    p_buffers: *const vk::Buffer,
    p_offsets: *const vk::DeviceSize,
) {
    if p_buffers.is_null() || p_offsets.is_null() || binding_count == 0 {
        return;
    }

    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let mut buffers = Vec::new();
    for i in 0..binding_count as usize {
        let buf = *p_buffers.add(i);
        match handle_store::get_buffer(buf.as_raw()) {
            Some(h) => buffers.push(h),
            None => return,
        }
    }

    let offsets =
        std::slice::from_raw_parts(p_offsets, binding_count as usize).to_vec();

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::BindVertexBuffers {
                first_binding,
                buffers,
                offsets,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdBindIndexBuffer(
    command_buffer: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    index_type: vk::IndexType,
) {
    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let buf_handle = match handle_store::get_buffer(buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::BindIndexBuffer {
                buffer: buf_handle,
                offset,
                index_type: index_type.as_raw() as u32,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdSetViewport(
    command_buffer: vk::CommandBuffer,
    first_viewport: u32,
    viewport_count: u32,
    p_viewports: *const vk::Viewport,
) {
    if p_viewports.is_null() || viewport_count == 0 {
        return;
    }

    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let viewports: Vec<SerializedViewport> =
        std::slice::from_raw_parts(p_viewports, viewport_count as usize)
            .iter()
            .map(|v| SerializedViewport {
                x: v.x,
                y: v.y,
                width: v.width,
                height: v.height,
                min_depth: v.min_depth,
                max_depth: v.max_depth,
            })
            .collect();

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::SetViewport {
                first_viewport,
                viewports,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdSetScissor(
    command_buffer: vk::CommandBuffer,
    first_scissor: u32,
    scissor_count: u32,
    p_scissors: *const vk::Rect2D,
) {
    if p_scissors.is_null() || scissor_count == 0 {
        return;
    }

    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let scissors: Vec<SerializedRect2D> =
        std::slice::from_raw_parts(p_scissors, scissor_count as usize)
            .iter()
            .map(|s| SerializedRect2D {
                offset: [s.offset.x, s.offset.y],
                extent: [s.extent.width, s.extent.height],
            })
            .collect();

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::SetScissor {
                first_scissor,
                scissors,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdCopyBufferToImage(
    command_buffer: vk::CommandBuffer,
    src_buffer: vk::Buffer,
    dst_image: vk::Image,
    dst_image_layout: vk::ImageLayout,
    region_count: u32,
    p_regions: *const vk::BufferImageCopy,
) {
    if p_regions.is_null() || region_count == 0 {
        return;
    }

    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let buf_handle = match handle_store::get_buffer(src_buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };
    let img_handle = match handle_store::get_image(dst_image.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let regions: Vec<SerializedBufferImageCopy> =
        std::slice::from_raw_parts(p_regions, region_count as usize)
            .iter()
            .map(|r| SerializedBufferImageCopy {
                buffer_offset: r.buffer_offset,
                buffer_row_length: r.buffer_row_length,
                buffer_image_height: r.buffer_image_height,
                image_subresource: SerializedImageSubresourceLayers {
                    aspect_mask: r.image_subresource.aspect_mask.as_raw(),
                    mip_level: r.image_subresource.mip_level,
                    base_array_layer: r.image_subresource.base_array_layer,
                    layer_count: r.image_subresource.layer_count,
                },
                image_offset: [r.image_offset.x, r.image_offset.y, r.image_offset.z],
                image_extent: [
                    r.image_extent.width,
                    r.image_extent.height,
                    r.image_extent.depth,
                ],
            })
            .collect();

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::CopyBufferToImage {
                src_buffer: buf_handle,
                dst_image: img_handle,
                dst_image_layout: dst_image_layout.as_raw(),
                regions,
            });
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkCmdCopyImageToBuffer(
    command_buffer: vk::CommandBuffer,
    src_image: vk::Image,
    src_image_layout: vk::ImageLayout,
    dst_buffer: vk::Buffer,
    region_count: u32,
    p_regions: *const vk::BufferImageCopy,
) {
    if p_regions.is_null() || region_count == 0 {
        return;
    }

    let cb_disp = command_buffer.as_raw() as *const DispatchableHandle;
    let local_id = DispatchableHandle::get_id(cb_disp);

    let img_handle = match handle_store::get_image(src_image.as_raw()) {
        Some(h) => h,
        None => return,
    };
    let buf_handle = match handle_store::get_buffer(dst_buffer.as_raw()) {
        Some(h) => h,
        None => return,
    };

    let regions: Vec<SerializedBufferImageCopy> =
        std::slice::from_raw_parts(p_regions, region_count as usize)
            .iter()
            .map(|r| SerializedBufferImageCopy {
                buffer_offset: r.buffer_offset,
                buffer_row_length: r.buffer_row_length,
                buffer_image_height: r.buffer_image_height,
                image_subresource: SerializedImageSubresourceLayers {
                    aspect_mask: r.image_subresource.aspect_mask.as_raw(),
                    mip_level: r.image_subresource.mip_level,
                    base_array_layer: r.image_subresource.base_array_layer,
                    layer_count: r.image_subresource.layer_count,
                },
                image_offset: [r.image_offset.x, r.image_offset.y, r.image_offset.z],
                image_extent: [
                    r.image_extent.width,
                    r.image_extent.height,
                    r.image_extent.depth,
                ],
            })
            .collect();

    if let Ok(mut states) = cmd_buf_states().lock() {
        if let Some(state) = states.get_mut(&local_id) {
            state.commands.push(RecordedCommand::CopyImageToBuffer {
                src_image: img_handle,
                src_image_layout: src_image_layout.as_raw(),
                dst_buffer: buf_handle,
                regions,
            });
        }
    }
}
