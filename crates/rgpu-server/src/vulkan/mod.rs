mod instance;
mod physical_device;
mod device;
mod memory;
mod buffer;
mod image;
mod pipeline;
mod graphics_pipeline;
mod descriptor;
mod command;
mod sync;
mod renderpass;

use std::sync::Arc;

use ash::vk;
use dashmap::DashMap;
use tracing::{info, warn};

use rgpu_protocol::handle::{NetworkHandle, ResourceType};
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;

/// Server-side Vulkan command executor.
/// Executes Vulkan commands on real GPU hardware via `ash`.
pub struct VulkanExecutor {
    /// The ash Entry (loaded once). None if Vulkan is not available on this system.
    pub(crate) entry: Option<Arc<ash::Entry>>,

    // ── Handle Maps ─────────────────────────────────────────
    pub(crate) instance_handles: DashMap<NetworkHandle, vk::Instance>,
    pub(crate) instance_wrappers: DashMap<NetworkHandle, ash::Instance>,
    /// (physical_device, parent_instance_handle)
    pub(crate) physical_device_handles: DashMap<NetworkHandle, (vk::PhysicalDevice, NetworkHandle)>,
    pub(crate) device_handles: DashMap<NetworkHandle, vk::Device>,
    pub(crate) device_wrappers: DashMap<NetworkHandle, ash::Device>,
    pub(crate) device_to_instance: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) queue_handles: DashMap<NetworkHandle, vk::Queue>,
    pub(crate) memory_handles: DashMap<NetworkHandle, vk::DeviceMemory>,
    pub(crate) memory_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) memory_info: DashMap<NetworkHandle, MappedMemoryInfo>,
    pub(crate) buffer_handles: DashMap<NetworkHandle, vk::Buffer>,
    pub(crate) buffer_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) shader_module_handles: DashMap<NetworkHandle, vk::ShaderModule>,
    pub(crate) shader_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) desc_set_layout_handles: DashMap<NetworkHandle, vk::DescriptorSetLayout>,
    pub(crate) desc_set_layout_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) pipeline_layout_handles: DashMap<NetworkHandle, vk::PipelineLayout>,
    pub(crate) pipeline_layout_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) pipeline_handles: DashMap<NetworkHandle, vk::Pipeline>,
    pub(crate) pipeline_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) desc_pool_handles: DashMap<NetworkHandle, vk::DescriptorPool>,
    pub(crate) desc_pool_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) desc_set_handles: DashMap<NetworkHandle, vk::DescriptorSet>,
    pub(crate) command_pool_handles: DashMap<NetworkHandle, vk::CommandPool>,
    pub(crate) command_pool_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) command_buffer_handles: DashMap<NetworkHandle, vk::CommandBuffer>,
    pub(crate) command_buffer_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) fence_handles: DashMap<NetworkHandle, vk::Fence>,
    pub(crate) fence_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) image_handles: DashMap<NetworkHandle, vk::Image>,
    pub(crate) image_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) image_view_handles: DashMap<NetworkHandle, vk::ImageView>,
    pub(crate) image_view_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) render_pass_handles: DashMap<NetworkHandle, vk::RenderPass>,
    pub(crate) render_pass_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) framebuffer_handles: DashMap<NetworkHandle, vk::Framebuffer>,
    pub(crate) framebuffer_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pub(crate) semaphore_handles: DashMap<NetworkHandle, vk::Semaphore>,
    pub(crate) semaphore_to_device: DashMap<NetworkHandle, NetworkHandle>,
}

pub(crate) struct MappedMemoryInfo {
    pub(crate) offset: u64,
    pub(crate) _size: u64,
    pub(crate) ptr: *mut std::ffi::c_void,
}

// SAFETY: Vulkan handles are valid across threads with proper external synchronization
unsafe impl Send for VulkanExecutor {}
unsafe impl Sync for VulkanExecutor {}

impl VulkanExecutor {
    pub fn new() -> Self {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => {
                info!("Vulkan entry loaded successfully");
                Some(Arc::new(e))
            }
            Err(e) => {
                warn!("Vulkan not available: {}. Vulkan commands will return errors.", e);
                None
            }
        };

        Self {
            entry,
            instance_handles: DashMap::new(),
            instance_wrappers: DashMap::new(),
            physical_device_handles: DashMap::new(),
            device_handles: DashMap::new(),
            device_wrappers: DashMap::new(),
            device_to_instance: DashMap::new(),
            queue_handles: DashMap::new(),
            memory_handles: DashMap::new(),
            memory_to_device: DashMap::new(),
            memory_info: DashMap::new(),
            buffer_handles: DashMap::new(),
            buffer_to_device: DashMap::new(),
            shader_module_handles: DashMap::new(),
            shader_to_device: DashMap::new(),
            desc_set_layout_handles: DashMap::new(),
            desc_set_layout_to_device: DashMap::new(),
            pipeline_layout_handles: DashMap::new(),
            pipeline_layout_to_device: DashMap::new(),
            pipeline_handles: DashMap::new(),
            pipeline_to_device: DashMap::new(),
            desc_pool_handles: DashMap::new(),
            desc_pool_to_device: DashMap::new(),
            desc_set_handles: DashMap::new(),
            command_pool_handles: DashMap::new(),
            command_pool_to_device: DashMap::new(),
            command_buffer_handles: DashMap::new(),
            command_buffer_to_device: DashMap::new(),
            fence_handles: DashMap::new(),
            fence_to_device: DashMap::new(),
            image_handles: DashMap::new(),
            image_to_device: DashMap::new(),
            image_view_handles: DashMap::new(),
            image_view_to_device: DashMap::new(),
            render_pass_handles: DashMap::new(),
            render_pass_to_device: DashMap::new(),
            framebuffer_handles: DashMap::new(),
            framebuffer_to_device: DashMap::new(),
            semaphore_handles: DashMap::new(),
            semaphore_to_device: DashMap::new(),
        }
    }

    pub(crate) fn vk_err(result: vk::Result) -> VulkanResponse {
        VulkanResponse::Error {
            code: result.as_raw(),
            message: format!("{:?}", result),
        }
    }

    /// Check if Vulkan is available on this system.
    pub fn is_available(&self) -> bool {
        self.entry.is_some()
    }

    /// Execute a Vulkan command and return the response.
    pub fn execute(&self, session: &Session, cmd: VulkanCommand) -> VulkanResponse {
        let entry = match &self.entry {
            Some(e) => e,
            None => {
                return VulkanResponse::Error {
                    code: -1,
                    message: "Vulkan is not available on this system".to_string(),
                };
            }
        };

        match cmd {
            // ── Instance ────────────────────────────────────────
            VulkanCommand::CreateInstance { .. } => self.handle_create_instance(session, entry, cmd),
            VulkanCommand::DestroyInstance { .. } => self.handle_destroy_instance(session, cmd),
            VulkanCommand::EnumeratePhysicalDevices { .. } => self.handle_enumerate_physical_devices(session, cmd),
            VulkanCommand::EnumerateInstanceExtensionProperties { .. } => self.handle_enumerate_instance_extension_properties(cmd),
            VulkanCommand::EnumerateInstanceLayerProperties => self.handle_enumerate_instance_layer_properties(),
            VulkanCommand::EnumerateDeviceExtensionProperties { .. } => self.handle_enumerate_device_extension_properties(cmd),

            // ── Physical Device Queries ─────────────────────────
            VulkanCommand::GetPhysicalDeviceProperties { .. }
            | VulkanCommand::GetPhysicalDeviceProperties2 { .. } => self.handle_get_physical_device_properties(cmd),
            VulkanCommand::GetPhysicalDeviceFeatures { .. }
            | VulkanCommand::GetPhysicalDeviceFeatures2 { .. } => self.handle_get_physical_device_features(cmd),
            VulkanCommand::GetPhysicalDeviceMemoryProperties { .. }
            | VulkanCommand::GetPhysicalDeviceMemoryProperties2 { .. } => self.handle_get_physical_device_memory_properties(cmd),
            VulkanCommand::GetPhysicalDeviceQueueFamilyProperties { .. }
            | VulkanCommand::GetPhysicalDeviceQueueFamilyProperties2 { .. } => self.handle_get_physical_device_queue_family_properties(cmd),
            VulkanCommand::GetPhysicalDeviceFormatProperties { .. } => self.handle_get_physical_device_format_properties(cmd),

            // ── Logical Device ──────────────────────────────────
            VulkanCommand::CreateDevice { .. } => self.handle_create_device(session, cmd),
            VulkanCommand::DestroyDevice { .. } => self.handle_destroy_device(session, cmd),
            VulkanCommand::DeviceWaitIdle { .. } => self.handle_device_wait_idle(cmd),
            VulkanCommand::GetDeviceQueue { .. } => self.handle_get_device_queue(session, cmd),

            // ── Memory ──────────────────────────────────────────
            VulkanCommand::AllocateMemory { .. } => self.handle_allocate_memory(session, cmd),
            VulkanCommand::FreeMemory { .. } => self.handle_free_memory(session, cmd),
            VulkanCommand::MapMemory { .. } => self.handle_map_memory(cmd),
            VulkanCommand::UnmapMemory { .. } => self.handle_unmap_memory(cmd),
            VulkanCommand::FlushMappedMemoryRanges { .. } => self.handle_flush_mapped_memory_ranges(cmd),
            VulkanCommand::InvalidateMappedMemoryRanges { .. } => self.handle_invalidate_mapped_memory_ranges(cmd),
            VulkanCommand::GetBufferMemoryRequirements { .. } => self.handle_get_buffer_memory_requirements(cmd),
            VulkanCommand::GetImageMemoryRequirements { .. } => self.handle_get_image_memory_requirements(cmd),

            // ── Buffer ──────────────────────────────────────────
            VulkanCommand::CreateBuffer { .. } => self.handle_create_buffer(session, cmd),
            VulkanCommand::DestroyBuffer { .. } => self.handle_destroy_buffer(session, cmd),
            VulkanCommand::BindBufferMemory { .. } => self.handle_bind_buffer_memory(cmd),

            // ── Image ───────────────────────────────────────────
            VulkanCommand::CreateImage { .. } => self.handle_create_image(session, cmd),
            VulkanCommand::DestroyImage { .. } => self.handle_destroy_image(session, cmd),
            VulkanCommand::BindImageMemory { .. } => self.handle_bind_image_memory(cmd),
            VulkanCommand::CreateImageView { .. } => self.handle_create_image_view(session, cmd),
            VulkanCommand::DestroyImageView { .. } => self.handle_destroy_image_view(session, cmd),

            // ── Pipeline ────────────────────────────────────────
            VulkanCommand::CreateShaderModule { .. } => self.handle_create_shader_module(session, cmd),
            VulkanCommand::DestroyShaderModule { .. } => self.handle_destroy_shader_module(session, cmd),
            VulkanCommand::CreateDescriptorSetLayout { .. } => self.handle_create_descriptor_set_layout(session, cmd),
            VulkanCommand::DestroyDescriptorSetLayout { .. } => self.handle_destroy_descriptor_set_layout(session, cmd),
            VulkanCommand::CreatePipelineLayout { .. } => self.handle_create_pipeline_layout(session, cmd),
            VulkanCommand::DestroyPipelineLayout { .. } => self.handle_destroy_pipeline_layout(session, cmd),
            VulkanCommand::CreateComputePipelines { .. } => self.handle_create_compute_pipelines(session, cmd),
            VulkanCommand::DestroyPipeline { .. } => self.handle_destroy_pipeline(session, cmd),

            // ── Graphics Pipeline ───────────────────────────────
            VulkanCommand::CreateGraphicsPipelines { .. } => self.handle_create_graphics_pipelines(session, cmd),

            // ── Descriptor ──────────────────────────────────────
            VulkanCommand::CreateDescriptorPool { .. } => self.handle_create_descriptor_pool(session, cmd),
            VulkanCommand::DestroyDescriptorPool { .. } => self.handle_destroy_descriptor_pool(session, cmd),
            VulkanCommand::AllocateDescriptorSets { .. } => self.handle_allocate_descriptor_sets(session, cmd),
            VulkanCommand::FreeDescriptorSets { .. } => self.handle_free_descriptor_sets(session, cmd),
            VulkanCommand::UpdateDescriptorSets { .. } => self.handle_update_descriptor_sets(cmd),

            // ── Command ─────────────────────────────────────────
            VulkanCommand::CreateCommandPool { .. } => self.handle_create_command_pool(session, cmd),
            VulkanCommand::DestroyCommandPool { .. } => self.handle_destroy_command_pool(session, cmd),
            VulkanCommand::ResetCommandPool { .. } => self.handle_reset_command_pool(cmd),
            VulkanCommand::AllocateCommandBuffers { .. } => self.handle_allocate_command_buffers(session, cmd),
            VulkanCommand::FreeCommandBuffers { .. } => self.handle_free_command_buffers(session, cmd),
            VulkanCommand::SubmitRecordedCommands { .. } => self.handle_submit_recorded_commands(cmd),

            // ── Sync ────────────────────────────────────────────
            VulkanCommand::CreateFence { .. } => self.handle_create_fence(session, cmd),
            VulkanCommand::DestroyFence { .. } => self.handle_destroy_fence(session, cmd),
            VulkanCommand::WaitForFences { .. } => self.handle_wait_for_fences(cmd),
            VulkanCommand::ResetFences { .. } => self.handle_reset_fences(cmd),
            VulkanCommand::GetFenceStatus { .. } => self.handle_get_fence_status(cmd),
            VulkanCommand::CreateSemaphore { .. } => self.handle_create_semaphore(session, cmd),
            VulkanCommand::DestroySemaphore { .. } => self.handle_destroy_semaphore(session, cmd),
            VulkanCommand::QueueSubmit { .. } => self.handle_queue_submit(cmd),
            VulkanCommand::QueueWaitIdle { .. } => self.handle_queue_wait_idle(cmd),

            // ── Render Pass ─────────────────────────────────────
            VulkanCommand::CreateRenderPass { .. } => self.handle_create_render_pass(session, cmd),
            VulkanCommand::DestroyRenderPass { .. } => self.handle_destroy_render_pass(session, cmd),
            VulkanCommand::CreateFramebuffer { .. } => self.handle_create_framebuffer(session, cmd),
            VulkanCommand::DestroyFramebuffer { .. } => self.handle_destroy_framebuffer(session, cmd),
        }
    }

    /// Clean up all Vulkan resources owned by a disconnecting session.
    /// Destroys in reverse-dependency order to avoid use-after-free.
    pub fn cleanup_session(&self, session: &Session) {
        let handles = session.all_handles();
        if handles.is_empty() {
            return;
        }

        let mut cleaned = 0u32;

        // Helper: get ash::Device for a handle via *_to_device mapping
        macro_rules! cleanup_vk {
            ($handles:expr, $to_device:expr, $resource_type:expr, $destroy_fn:ident) => {
                for h in handles.iter().filter(|h| h.resource_type == $resource_type) {
                    if let Some((_, raw)) = $handles.remove(h) {
                        if let Some((_, dev_handle)) = $to_device.remove(h) {
                            if let Some(dev) = self.device_wrappers.get(&dev_handle) {
                                unsafe { dev.$destroy_fn(raw, None); }
                            }
                        }
                        cleaned += 1;
                    }
                }
            };
        }

        // Pass 1: Framebuffers
        cleanup_vk!(self.framebuffer_handles, self.framebuffer_to_device, ResourceType::VkFramebuffer, destroy_framebuffer);

        // Pass 2: ImageViews
        cleanup_vk!(self.image_view_handles, self.image_view_to_device, ResourceType::VkImageView, destroy_image_view);

        // Pass 3: RenderPasses
        cleanup_vk!(self.render_pass_handles, self.render_pass_to_device, ResourceType::VkRenderPass, destroy_render_pass);

        // Pass 4: Pipelines
        cleanup_vk!(self.pipeline_handles, self.pipeline_to_device, ResourceType::VkPipeline, destroy_pipeline);

        // Pass 5: PipelineLayouts
        cleanup_vk!(self.pipeline_layout_handles, self.pipeline_layout_to_device, ResourceType::VkPipelineLayout, destroy_pipeline_layout);

        // Pass 6: DescriptorPools (implicitly frees descriptor sets)
        cleanup_vk!(self.desc_pool_handles, self.desc_pool_to_device, ResourceType::VkDescriptorPool, destroy_descriptor_pool);
        // Remove any remaining descriptor sets and layouts
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::VkDescriptorSet) {
            self.desc_set_handles.remove(h);
        }
        cleanup_vk!(self.desc_set_layout_handles, self.desc_set_layout_to_device, ResourceType::VkDescriptorSetLayout, destroy_descriptor_set_layout);

        // Pass 7: ShaderModules
        cleanup_vk!(self.shader_module_handles, self.shader_to_device, ResourceType::VkShaderModule, destroy_shader_module);

        // Pass 8: CommandPools (implicitly frees command buffers)
        cleanup_vk!(self.command_pool_handles, self.command_pool_to_device, ResourceType::VkCommandPool, destroy_command_pool);
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::VkCommandBuffer) {
            self.command_buffer_handles.remove(h);
            self.command_buffer_to_device.remove(h);
        }

        // Pass 9: Fences, Semaphores, Events
        cleanup_vk!(self.fence_handles, self.fence_to_device, ResourceType::VkFence, destroy_fence);
        cleanup_vk!(self.semaphore_handles, self.semaphore_to_device, ResourceType::VkSemaphore, destroy_semaphore);

        // Pass 10: Images
        cleanup_vk!(self.image_handles, self.image_to_device, ResourceType::VkImage, destroy_image);

        // Pass 11: Buffers
        cleanup_vk!(self.buffer_handles, self.buffer_to_device, ResourceType::VkBuffer, destroy_buffer);

        // Pass 12: DeviceMemory
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::VkDeviceMemory) {
            if let Some((_, mem)) = self.memory_handles.remove(h) {
                if let Some((_, dev_handle)) = self.memory_to_device.remove(h) {
                    if let Some(dev) = self.device_wrappers.get(&dev_handle) {
                        unsafe { dev.free_memory(mem, None); }
                    }
                }
                self.memory_info.remove(h);
                cleaned += 1;
            }
        }

        // Pass 13: Queues (no destroy, just remove tracking)
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::VkQueue) {
            self.queue_handles.remove(h);
        }

        // Pass 14: Devices
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::VkDevice) {
            if let Some((_, _raw)) = self.device_handles.remove(h) {
                if let Some((_, dev_wrapper)) = self.device_wrappers.remove(h) {
                    unsafe { dev_wrapper.destroy_device(None); }
                }
                self.device_to_instance.remove(h);
                cleaned += 1;
            }
        }

        // Pass 15: PhysicalDevices (no destroy, just remove)
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::VkPhysicalDevice) {
            self.physical_device_handles.remove(h);
        }

        // Pass 16: Instances
        for h in handles.iter().filter(|h| h.resource_type == ResourceType::VkInstance) {
            if let Some((_, _raw)) = self.instance_handles.remove(h) {
                if let Some((_, inst_wrapper)) = self.instance_wrappers.remove(h) {
                    unsafe { inst_wrapper.destroy_instance(None); }
                }
                cleaned += 1;
            }
        }

        if cleaned > 0 {
            info!(
                session_id = session.session_id,
                "cleaned up {} Vulkan resource(s)", cleaned
            );
        }
    }
}
