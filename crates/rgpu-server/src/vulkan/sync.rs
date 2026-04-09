use ash::vk;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_fence(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, signaled) = match cmd {
            VulkanCommand::CreateFence { device, signaled } => (device, signaled),
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

        let mut create_info = vk::FenceCreateInfo::default();
        if signaled {
            create_info = create_info.flags(vk::FenceCreateFlags::SIGNALED);
        }

        match unsafe { dev.create_fence(&create_info, None) } {
            Ok(fence) => {
                let handle = session.alloc_handle(ResourceType::VkFence);
                self.fence_handles.insert(handle, fence);
                self.fence_to_device.insert(handle, device);
                VulkanResponse::FenceCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_fence(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, fence) = match cmd {
            VulkanCommand::DestroyFence { device, fence } => (device, fence),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, f)) = self.fence_handles.remove(&fence) {
            unsafe { dev.destroy_fence(f, None) };
            self.fence_to_device.remove(&fence);
            session.remove_handle(&fence);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_wait_for_fences(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, fences, wait_all, timeout_ns) = match cmd {
            VulkanCommand::WaitForFences {
                device,
                fences,
                wait_all,
                timeout_ns,
            } => (device, fences, wait_all, timeout_ns),
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

        let vk_fences: Vec<vk::Fence> = fences
            .iter()
            .filter_map(|h| self.fence_handles.get(h).map(|v| *v.value()))
            .collect();

        match unsafe { dev.wait_for_fences(&vk_fences, wait_all, timeout_ns) } {
            Ok(()) => VulkanResponse::FenceWaitResult {
                result: vk::Result::SUCCESS.as_raw(),
            },
            Err(vk::Result::TIMEOUT) => VulkanResponse::FenceWaitResult {
                result: vk::Result::TIMEOUT.as_raw(),
            },
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_reset_fences(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, fences) = match cmd {
            VulkanCommand::ResetFences { device, fences } => (device, fences),
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

        let vk_fences: Vec<vk::Fence> = fences
            .iter()
            .filter_map(|h| self.fence_handles.get(h).map(|v| *v.value()))
            .collect();

        match unsafe { dev.reset_fences(&vk_fences) } {
            Ok(()) => VulkanResponse::Success,
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_get_fence_status(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (device, fence) = match cmd {
            VulkanCommand::GetFenceStatus { device, fence } => (device, fence),
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
        let f = match self.fence_handles.get(&fence) {
            Some(f) => *f.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid fence handle".to_string(),
                }
            }
        };
        match unsafe { dev.get_fence_status(f) } {
            Ok(true) => VulkanResponse::FenceStatus { signaled: true },
            Ok(false) => VulkanResponse::FenceStatus { signaled: false },
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_create_semaphore(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let device = match cmd {
            VulkanCommand::CreateSemaphore { device } => device,
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

        let ci = vk::SemaphoreCreateInfo::default();
        match unsafe { dev.create_semaphore(&ci, None) } {
            Ok(sem) => {
                let handle = session.alloc_handle(ResourceType::VkSemaphore);
                self.semaphore_handles.insert(handle, sem);
                self.semaphore_to_device.insert(handle, device);
                VulkanResponse::SemaphoreCreated { handle }
            }
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_destroy_semaphore(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, semaphore) = match cmd {
            VulkanCommand::DestroySemaphore { device, semaphore } => (device, semaphore),
            _ => unreachable!(),
        };

        let dev = match self.device_wrappers.get(&device) {
            Some(d) => d,
            None => return VulkanResponse::Success,
        };
        if let Some((_, sem)) = self.semaphore_handles.remove(&semaphore) {
            unsafe { dev.destroy_semaphore(sem, None) };
            self.semaphore_to_device.remove(&semaphore);
            session.remove_handle(&semaphore);
        }
        VulkanResponse::Success
    }

    pub(crate) fn handle_queue_submit(&self, cmd: VulkanCommand) -> VulkanResponse {
        let (queue, submits, fence) = match cmd {
            VulkanCommand::QueueSubmit {
                queue,
                submits,
                fence,
            } => (queue, submits, fence),
            _ => unreachable!(),
        };

        let q = match self.queue_handles.get(&queue) {
            Some(q) => *q.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid queue handle".to_string(),
                }
            }
        };

        // We need to find the device wrapper for this queue.
        // Use any device wrapper we have (in Phase 3, single device assumption).
        let dev = match self.device_wrappers.iter().next() {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "no device available".to_string(),
                }
            }
        };

        // Resolve command buffer handles
        let mut resolved_submits = Vec::new();
        let mut cmd_buf_vecs: Vec<Vec<vk::CommandBuffer>> = Vec::new();
        let mut wait_sem_vecs: Vec<Vec<vk::Semaphore>> = Vec::new();
        let mut sig_sem_vecs: Vec<Vec<vk::Semaphore>> = Vec::new();
        let mut stage_mask_vecs: Vec<Vec<vk::PipelineStageFlags>> = Vec::new();

        for submit in &submits {
            let cmd_bufs: Vec<vk::CommandBuffer> = submit
                .command_buffers
                .iter()
                .filter_map(|h| self.command_buffer_handles.get(h).map(|v| *v.value()))
                .collect();
            cmd_buf_vecs.push(cmd_bufs);

            let wait_sems: Vec<vk::Semaphore> = submit
                .wait_semaphores
                .iter()
                .filter_map(|h| self.semaphore_handles.get(h).map(|v| *v.value()))
                .collect();
            wait_sem_vecs.push(wait_sems);

            let sig_sems: Vec<vk::Semaphore> = submit
                .signal_semaphores
                .iter()
                .filter_map(|h| self.semaphore_handles.get(h).map(|v| *v.value()))
                .collect();
            sig_sem_vecs.push(sig_sems);

            let stage_masks: Vec<vk::PipelineStageFlags> = submit
                .wait_dst_stage_masks
                .iter()
                .map(|m| vk::PipelineStageFlags::from_raw(*m))
                .collect();
            stage_mask_vecs.push(stage_masks);
        }

        for i in 0..submits.len() {
            let mut submit_info = vk::SubmitInfo::default()
                .command_buffers(&cmd_buf_vecs[i]);
            if !wait_sem_vecs[i].is_empty() {
                submit_info = submit_info
                    .wait_semaphores(&wait_sem_vecs[i])
                    .wait_dst_stage_mask(&stage_mask_vecs[i]);
            }
            if !sig_sem_vecs[i].is_empty() {
                submit_info = submit_info.signal_semaphores(&sig_sem_vecs[i]);
            }
            resolved_submits.push(submit_info);
        }

        let vk_fence = fence
            .and_then(|fh| self.fence_handles.get(&fh).map(|v| *v.value()))
            .unwrap_or(vk::Fence::null());

        match unsafe { dev.queue_submit(q, &resolved_submits, vk_fence) } {
            Ok(()) => VulkanResponse::Success,
            Err(e) => Self::vk_err(e),
        }
    }

    pub(crate) fn handle_queue_wait_idle(&self, cmd: VulkanCommand) -> VulkanResponse {
        let queue = match cmd {
            VulkanCommand::QueueWaitIdle { queue } => queue,
            _ => unreachable!(),
        };

        let q = match self.queue_handles.get(&queue) {
            Some(q) => *q.value(),
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "invalid queue handle".to_string(),
                }
            }
        };
        let dev = match self.device_wrappers.iter().next() {
            Some(d) => d,
            None => {
                return VulkanResponse::Error {
                    code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                    message: "no device available".to_string(),
                }
            }
        };
        match unsafe { dev.queue_wait_idle(q) } {
            Ok(()) => VulkanResponse::Success,
            Err(e) => Self::vk_err(e),
        }
    }
}
