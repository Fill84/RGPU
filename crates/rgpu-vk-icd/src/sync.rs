//! Fence, semaphore, and queue submission functions for the Vulkan ICD.

use ash::vk;
use ash::vk::Handle;

use crate::command;
use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{SerializedSubmitInfo, VulkanCommand, VulkanResponse};

// ── Fence ───────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateFence(
    device: vk::Device,
    p_create_info: *const vk::FenceCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_fence: *mut vk::Fence,
) -> vk::Result {
    if p_create_info.is_null() || p_fence.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;
    let signaled = ci.flags.contains(vk::FenceCreateFlags::SIGNALED);

    let cmd = VulkanCommand::CreateFence {
        device: dev_handle,
        signaled,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::FenceCreated { handle }) => {
            let local_id = handle_store::store_fence(handle);
            *p_fence = vk::Fence::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroyFence(
    device: vk::Device,
    fence: vk::Fence,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if fence == vk::Fence::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = fence.as_raw();
    if let Some(handle) = handle_store::remove_fence(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyFence {
            device: dev_handle,
            fence: handle,
        });
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkWaitForFences(
    device: vk::Device,
    fence_count: u32,
    p_fences: *const vk::Fence,
    wait_all: vk::Bool32,
    timeout: u64,
) -> vk::Result {
    if p_fences.is_null() || fence_count == 0 {
        return vk::Result::SUCCESS;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let mut fence_handles = Vec::new();
    for i in 0..fence_count as usize {
        let f = *p_fences.add(i);
        match handle_store::get_fence(f.as_raw()) {
            Some(h) => fence_handles.push(h),
            None => return vk::Result::ERROR_UNKNOWN,
        }
    }

    let cmd = VulkanCommand::WaitForFences {
        device: dev_handle,
        fences: fence_handles,
        wait_all: wait_all != 0,
        timeout_ns: timeout,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::FenceWaitResult { result }) => vk::Result::from_raw(result),
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkResetFences(
    device: vk::Device,
    fence_count: u32,
    p_fences: *const vk::Fence,
) -> vk::Result {
    if p_fences.is_null() || fence_count == 0 {
        return vk::Result::SUCCESS;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let mut fence_handles = Vec::new();
    for i in 0..fence_count as usize {
        let f = *p_fences.add(i);
        match handle_store::get_fence(f.as_raw()) {
            Some(h) => fence_handles.push(h),
            None => return vk::Result::ERROR_UNKNOWN,
        }
    }

    let cmd = VulkanCommand::ResetFences {
        device: dev_handle,
        fences: fence_handles,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkGetFenceStatus(
    device: vk::Device,
    fence: vk::Fence,
) -> vk::Result {
    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let fence_handle = match handle_store::get_fence(fence.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let cmd = VulkanCommand::GetFenceStatus {
        device: dev_handle,
        fence: fence_handle,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::FenceStatus { signaled }) => {
            if signaled {
                vk::Result::SUCCESS
            } else {
                vk::Result::NOT_READY
            }
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

// ── Semaphore ──────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateSemaphore(
    device: vk::Device,
    _p_create_info: *const vk::SemaphoreCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_semaphore: *mut vk::Semaphore,
) -> vk::Result {
    if p_semaphore.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let cmd = VulkanCommand::CreateSemaphore {
        device: dev_handle,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::SemaphoreCreated { handle }) => {
            let local_id = handle_store::store_semaphore(handle);
            *p_semaphore = vk::Semaphore::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkDestroySemaphore(
    device: vk::Device,
    semaphore: vk::Semaphore,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if semaphore == vk::Semaphore::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = semaphore.as_raw();
    if let Some(handle) = handle_store::remove_semaphore(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroySemaphore {
            device: dev_handle,
            semaphore: handle,
        });
    }
}

// ── Queue Submit ────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkQueueSubmit(
    queue: vk::Queue,
    submit_count: u32,
    p_submits: *const vk::SubmitInfo<'_>,
    fence: vk::Fence,
) -> vk::Result {
    let q_disp = queue.as_raw() as *const DispatchableHandle;
    let q_local_id = DispatchableHandle::get_id(q_disp);

    let queue_handle = match handle_store::get_queue(q_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    // First, send recorded commands for all command buffers in all submits
    if !p_submits.is_null() {
        for i in 0..submit_count as usize {
            let si = &*p_submits.add(i);
            if !si.p_command_buffers.is_null() {
                for j in 0..si.command_buffer_count as usize {
                    let cb = *si.p_command_buffers.add(j);
                    let cb_disp = cb.as_raw() as *const DispatchableHandle;
                    let cb_local_id = DispatchableHandle::get_id(cb_disp);

                    let cb_handle = match handle_store::get_cmd_buffer(cb_local_id) {
                        Some(h) => h,
                        None => return vk::Result::ERROR_UNKNOWN,
                    };

                    // Take recorded commands and send them
                    let commands = command::take_recorded_commands(cb_local_id);
                    if !commands.is_empty() {
                        let cmd = VulkanCommand::SubmitRecordedCommands {
                            command_buffer: cb_handle,
                            commands,
                        };

                        match send_vulkan_command(cmd) {
                            Ok(VulkanResponse::Success) => {}
                            Ok(VulkanResponse::Error { code, .. }) => {
                                return vk::Result::from_raw(code);
                            }
                            _ => return vk::Result::ERROR_UNKNOWN,
                        }
                    }
                }
            }
        }
    }

    // Now do the actual queue submit
    let mut submits = Vec::new();
    if !p_submits.is_null() {
        for i in 0..submit_count as usize {
            let si = &*p_submits.add(i);

            let mut command_buffers = Vec::new();
            if !si.p_command_buffers.is_null() {
                for j in 0..si.command_buffer_count as usize {
                    let cb = *si.p_command_buffers.add(j);
                    let cb_disp = cb.as_raw() as *const DispatchableHandle;
                    let cb_local_id = DispatchableHandle::get_id(cb_disp);
                    match handle_store::get_cmd_buffer(cb_local_id) {
                        Some(h) => command_buffers.push(h),
                        None => return vk::Result::ERROR_UNKNOWN,
                    }
                }
            }

            // Resolve wait semaphores
            let mut wait_semaphores = Vec::new();
            let mut wait_dst_stage_masks = Vec::new();
            if !si.p_wait_semaphores.is_null() && si.wait_semaphore_count > 0 {
                for k in 0..si.wait_semaphore_count as usize {
                    let sem = *si.p_wait_semaphores.add(k);
                    match handle_store::get_semaphore(sem.as_raw()) {
                        Some(h) => wait_semaphores.push(h),
                        None => return vk::Result::ERROR_UNKNOWN,
                    }
                }
                if !si.p_wait_dst_stage_mask.is_null() {
                    for k in 0..si.wait_semaphore_count as usize {
                        let mask = *si.p_wait_dst_stage_mask.add(k);
                        wait_dst_stage_masks.push(mask.as_raw());
                    }
                }
            }

            // Resolve signal semaphores
            let mut signal_semaphores = Vec::new();
            if !si.p_signal_semaphores.is_null() && si.signal_semaphore_count > 0 {
                for k in 0..si.signal_semaphore_count as usize {
                    let sem = *si.p_signal_semaphores.add(k);
                    match handle_store::get_semaphore(sem.as_raw()) {
                        Some(h) => signal_semaphores.push(h),
                        None => return vk::Result::ERROR_UNKNOWN,
                    }
                }
            }

            submits.push(SerializedSubmitInfo {
                wait_semaphores,
                wait_dst_stage_masks,
                command_buffers,
                signal_semaphores,
            });
        }
    }

    let fence_handle = if fence != vk::Fence::null() {
        handle_store::get_fence(fence.as_raw())
    } else {
        None
    };

    let cmd = VulkanCommand::QueueSubmit {
        queue: queue_handle,
        submits,
        fence: fence_handle,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

#[no_mangle]
pub unsafe extern "C" fn vkQueueWaitIdle(queue: vk::Queue) -> vk::Result {
    let q_disp = queue.as_raw() as *const DispatchableHandle;
    let q_local_id = DispatchableHandle::get_id(q_disp);

    let queue_handle = match handle_store::get_queue(q_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let cmd = VulkanCommand::QueueWaitIdle {
        queue: queue_handle,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::Success) => vk::Result::SUCCESS,
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_DEVICE_LOST,
    }
}
