use std::ffi::CStr;
use std::sync::Arc;

use ash::vk;
use dashmap::DashMap;
use tracing::{debug, error, info};

use rgpu_protocol::handle::{NetworkHandle, ResourceType};
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;

/// Server-side Vulkan command executor.
/// Executes Vulkan commands on real GPU hardware via `ash`.
pub struct VulkanExecutor {
    /// The ash Entry (loaded once)
    entry: Arc<ash::Entry>,

    // ── Handle Maps ─────────────────────────────────────────
    instance_handles: DashMap<NetworkHandle, vk::Instance>,
    instance_wrappers: DashMap<NetworkHandle, ash::Instance>,
    /// (physical_device, parent_instance_handle)
    physical_device_handles: DashMap<NetworkHandle, (vk::PhysicalDevice, NetworkHandle)>,
    device_handles: DashMap<NetworkHandle, vk::Device>,
    device_wrappers: DashMap<NetworkHandle, ash::Device>,
    device_to_instance: DashMap<NetworkHandle, NetworkHandle>,
    queue_handles: DashMap<NetworkHandle, vk::Queue>,
    memory_handles: DashMap<NetworkHandle, vk::DeviceMemory>,
    memory_to_device: DashMap<NetworkHandle, NetworkHandle>,
    memory_info: DashMap<NetworkHandle, MappedMemoryInfo>,
    buffer_handles: DashMap<NetworkHandle, vk::Buffer>,
    buffer_to_device: DashMap<NetworkHandle, NetworkHandle>,
    shader_module_handles: DashMap<NetworkHandle, vk::ShaderModule>,
    shader_to_device: DashMap<NetworkHandle, NetworkHandle>,
    desc_set_layout_handles: DashMap<NetworkHandle, vk::DescriptorSetLayout>,
    desc_set_layout_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pipeline_layout_handles: DashMap<NetworkHandle, vk::PipelineLayout>,
    pipeline_layout_to_device: DashMap<NetworkHandle, NetworkHandle>,
    pipeline_handles: DashMap<NetworkHandle, vk::Pipeline>,
    pipeline_to_device: DashMap<NetworkHandle, NetworkHandle>,
    desc_pool_handles: DashMap<NetworkHandle, vk::DescriptorPool>,
    desc_pool_to_device: DashMap<NetworkHandle, NetworkHandle>,
    desc_set_handles: DashMap<NetworkHandle, vk::DescriptorSet>,
    command_pool_handles: DashMap<NetworkHandle, vk::CommandPool>,
    command_pool_to_device: DashMap<NetworkHandle, NetworkHandle>,
    command_buffer_handles: DashMap<NetworkHandle, vk::CommandBuffer>,
    command_buffer_to_device: DashMap<NetworkHandle, NetworkHandle>,
    fence_handles: DashMap<NetworkHandle, vk::Fence>,
    fence_to_device: DashMap<NetworkHandle, NetworkHandle>,
    image_handles: DashMap<NetworkHandle, vk::Image>,
    image_to_device: DashMap<NetworkHandle, NetworkHandle>,
    image_view_handles: DashMap<NetworkHandle, vk::ImageView>,
    image_view_to_device: DashMap<NetworkHandle, NetworkHandle>,
    render_pass_handles: DashMap<NetworkHandle, vk::RenderPass>,
    render_pass_to_device: DashMap<NetworkHandle, NetworkHandle>,
    framebuffer_handles: DashMap<NetworkHandle, vk::Framebuffer>,
    framebuffer_to_device: DashMap<NetworkHandle, NetworkHandle>,
    semaphore_handles: DashMap<NetworkHandle, vk::Semaphore>,
    semaphore_to_device: DashMap<NetworkHandle, NetworkHandle>,
}

struct MappedMemoryInfo {
    offset: u64,
    _size: u64,
    ptr: *mut std::ffi::c_void,
}

// SAFETY: Vulkan handles are valid across threads with proper external synchronization
unsafe impl Send for VulkanExecutor {}
unsafe impl Sync for VulkanExecutor {}

impl VulkanExecutor {
    pub fn new() -> Self {
        let entry = match unsafe { ash::Entry::load() } {
            Ok(e) => {
                info!("Vulkan entry loaded successfully");
                Arc::new(e)
            }
            Err(e) => {
                error!("Failed to load Vulkan entry: {}", e);
                panic!("Vulkan is required for VulkanExecutor");
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

    fn vk_err(result: vk::Result) -> VulkanResponse {
        VulkanResponse::Error {
            code: result.as_raw(),
            message: format!("{:?}", result),
        }
    }

    /// Execute a Vulkan command and return the response.
    pub fn execute(&self, session: &Session, cmd: VulkanCommand) -> VulkanResponse {
        match cmd {
            // ── Instance ────────────────────────────────────────
            VulkanCommand::CreateInstance {
                app_name,
                app_version,
                engine_name,
                engine_version,
                api_version,
                enabled_extensions: _,
                enabled_layers: _,
            } => {
                let app_name_c = app_name
                    .as_deref()
                    .map(|s| std::ffi::CString::new(s).unwrap_or_default());
                let engine_name_c = engine_name
                    .as_deref()
                    .map(|s| std::ffi::CString::new(s).unwrap_or_default());

                let mut app_info = vk::ApplicationInfo::default()
                    .application_version(app_version)
                    .engine_version(engine_version)
                    .api_version(if api_version == 0 {
                        vk::make_api_version(0, 1, 3, 0)
                    } else {
                        api_version
                    });

                if let Some(ref name) = app_name_c {
                    app_info = app_info.application_name(name.as_c_str());
                }
                if let Some(ref name) = engine_name_c {
                    app_info = app_info.engine_name(name.as_c_str());
                }

                let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

                match unsafe { self.entry.create_instance(&create_info, None) } {
                    Ok(instance) => {
                        let handle = session.alloc_handle(ResourceType::VkInstance);
                        let raw = instance.handle();
                        self.instance_handles.insert(handle, raw);
                        self.instance_wrappers.insert(handle, instance);
                        info!("created Vulkan instance: {:?}", handle);
                        VulkanResponse::InstanceCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyInstance { instance } => {
                if let Some((_, inst)) = self.instance_handles.remove(&instance) {
                    if let Some((_, wrapper)) = self.instance_wrappers.remove(&instance) {
                        unsafe { wrapper.destroy_instance(None) };
                    }
                    // Clean up physical devices belonging to this instance
                    let pd_keys: Vec<NetworkHandle> = self
                        .physical_device_handles
                        .iter()
                        .filter(|e| e.value().1 == instance)
                        .map(|e| *e.key())
                        .collect();
                    for key in pd_keys {
                        self.physical_device_handles.remove(&key);
                    }
                    let _ = inst;
                    session.remove_handle(&instance);
                    debug!("destroyed Vulkan instance: {:?}", instance);
                }
                VulkanResponse::Success
            }

            // ── Enumeration ─────────────────────────────────────
            VulkanCommand::EnumeratePhysicalDevices { instance } => {
                let wrapper = match self.instance_wrappers.get(&instance) {
                    Some(w) => w,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: "invalid instance handle".to_string(),
                        }
                    }
                };

                match unsafe { wrapper.enumerate_physical_devices() } {
                    Ok(physical_devices) => {
                        let mut handles = Vec::new();
                        for pd in physical_devices {
                            let handle = session.alloc_handle(ResourceType::VkPhysicalDevice);
                            self.physical_device_handles
                                .insert(handle, (pd, instance));
                            handles.push(handle);
                        }
                        debug!("enumerated {} physical devices", handles.len());
                        VulkanResponse::PhysicalDevices { handles }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::EnumerateInstanceExtensionProperties { layer_name: _ } => {
                // For Phase 3, report a minimal set of extensions
                let extensions = vec![
                    SerializedExtensionProperties {
                        extension_name: "VK_KHR_get_physical_device_properties2".to_string(),
                        spec_version: 2,
                    },
                ];
                VulkanResponse::ExtensionProperties { extensions }
            }

            VulkanCommand::EnumerateInstanceLayerProperties => {
                VulkanResponse::LayerProperties { layers: vec![] }
            }

            VulkanCommand::EnumerateDeviceExtensionProperties {
                physical_device: _,
                layer_name: _,
            } => {
                // Minimal device extensions for Phase 3
                VulkanResponse::ExtensionProperties {
                    extensions: vec![],
                }
            }

            // ── Physical Device Queries ─────────────────────────
            VulkanCommand::GetPhysicalDeviceProperties { physical_device }
            | VulkanCommand::GetPhysicalDeviceProperties2 { physical_device } => {
                let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
                    Some(e) => *e.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid physical device handle".to_string(),
                        }
                    }
                };
                let wrapper = match self.instance_wrappers.get(&inst_handle) {
                    Some(w) => w,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: "instance wrapper not found".to_string(),
                        }
                    }
                };

                let props = unsafe { wrapper.get_physical_device_properties(pd) };

                let device_name = unsafe {
                    CStr::from_ptr(props.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };

                // Serialize limits as raw bytes
                let limits_raw = unsafe {
                    let ptr = &props.limits as *const vk::PhysicalDeviceLimits as *const u8;
                    std::slice::from_raw_parts(
                        ptr,
                        std::mem::size_of::<vk::PhysicalDeviceLimits>(),
                    )
                    .to_vec()
                };

                let sparse_properties_raw = unsafe {
                    let ptr = &props.sparse_properties
                        as *const vk::PhysicalDeviceSparseProperties
                        as *const u8;
                    std::slice::from_raw_parts(
                        ptr,
                        std::mem::size_of::<vk::PhysicalDeviceSparseProperties>(),
                    )
                    .to_vec()
                };

                VulkanResponse::PhysicalDeviceProperties {
                    api_version: props.api_version,
                    driver_version: props.driver_version,
                    vendor_id: props.vendor_id,
                    device_id: props.device_id,
                    device_type: props.device_type.as_raw() as u32,
                    device_name,
                    pipeline_cache_uuid: props.pipeline_cache_uuid,
                    limits_raw,
                    sparse_properties_raw,
                }
            }

            VulkanCommand::GetPhysicalDeviceFeatures { physical_device }
            | VulkanCommand::GetPhysicalDeviceFeatures2 { physical_device } => {
                let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
                    Some(e) => *e.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid physical device handle".to_string(),
                        }
                    }
                };
                let wrapper = match self.instance_wrappers.get(&inst_handle) {
                    Some(w) => w,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: "instance wrapper not found".to_string(),
                        }
                    }
                };

                let features = unsafe { wrapper.get_physical_device_features(pd) };
                let features_raw = unsafe {
                    let ptr = &features as *const vk::PhysicalDeviceFeatures as *const u8;
                    std::slice::from_raw_parts(
                        ptr,
                        std::mem::size_of::<vk::PhysicalDeviceFeatures>(),
                    )
                    .to_vec()
                };

                VulkanResponse::PhysicalDeviceFeatures { features_raw }
            }

            VulkanCommand::GetPhysicalDeviceMemoryProperties { physical_device }
            | VulkanCommand::GetPhysicalDeviceMemoryProperties2 { physical_device } => {
                let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
                    Some(e) => *e.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid physical device handle".to_string(),
                        }
                    }
                };
                let wrapper = match self.instance_wrappers.get(&inst_handle) {
                    Some(w) => w,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: "instance wrapper not found".to_string(),
                        }
                    }
                };

                let mem_props =
                    unsafe { wrapper.get_physical_device_memory_properties(pd) };

                let memory_types: Vec<SerializedMemoryType> =
                    (0..mem_props.memory_type_count as usize)
                        .map(|i| SerializedMemoryType {
                            property_flags: mem_props.memory_types[i].property_flags.as_raw(),
                            heap_index: mem_props.memory_types[i].heap_index,
                        })
                        .collect();

                let memory_heaps: Vec<SerializedMemoryHeap> =
                    (0..mem_props.memory_heap_count as usize)
                        .map(|i| SerializedMemoryHeap {
                            size: mem_props.memory_heaps[i].size,
                            flags: mem_props.memory_heaps[i].flags.as_raw(),
                        })
                        .collect();

                VulkanResponse::PhysicalDeviceMemoryProperties {
                    memory_type_count: mem_props.memory_type_count,
                    memory_types,
                    memory_heap_count: mem_props.memory_heap_count,
                    memory_heaps,
                }
            }

            VulkanCommand::GetPhysicalDeviceQueueFamilyProperties { physical_device }
            | VulkanCommand::GetPhysicalDeviceQueueFamilyProperties2 { physical_device } => {
                let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
                    Some(e) => *e.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid physical device handle".to_string(),
                        }
                    }
                };
                let wrapper = match self.instance_wrappers.get(&inst_handle) {
                    Some(w) => w,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: "instance wrapper not found".to_string(),
                        }
                    }
                };

                let queue_families =
                    unsafe { wrapper.get_physical_device_queue_family_properties(pd) };

                let families: Vec<SerializedQueueFamilyProperties> = queue_families
                    .iter()
                    .map(|qf| SerializedQueueFamilyProperties {
                        queue_flags: qf.queue_flags.as_raw(),
                        queue_count: qf.queue_count,
                        timestamp_valid_bits: qf.timestamp_valid_bits,
                        min_image_transfer_granularity: [
                            qf.min_image_transfer_granularity.width,
                            qf.min_image_transfer_granularity.height,
                            qf.min_image_transfer_granularity.depth,
                        ],
                    })
                    .collect();

                VulkanResponse::QueueFamilyProperties { families }
            }

            VulkanCommand::GetPhysicalDeviceFormatProperties {
                physical_device,
                format,
            } => {
                let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
                    Some(e) => *e.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid physical device handle".to_string(),
                        }
                    }
                };
                let wrapper = match self.instance_wrappers.get(&inst_handle) {
                    Some(w) => w,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: "instance wrapper not found".to_string(),
                        }
                    }
                };

                let vk_format = vk::Format::from_raw(format);
                let format_props =
                    unsafe { wrapper.get_physical_device_format_properties(pd, vk_format) };

                VulkanResponse::FormatProperties {
                    linear_tiling_features: format_props.linear_tiling_features.as_raw(),
                    optimal_tiling_features: format_props.optimal_tiling_features.as_raw(),
                    buffer_features: format_props.buffer_features.as_raw(),
                }
            }

            // ── Logical Device ──────────────────────────────────
            VulkanCommand::CreateDevice {
                physical_device,
                queue_create_infos,
                enabled_extensions: _,
                enabled_features,
            } => {
                let (pd, inst_handle) = match self.physical_device_handles.get(&physical_device) {
                    Some(e) => *e.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid physical device handle".to_string(),
                        }
                    }
                };
                let wrapper = match self.instance_wrappers.get(&inst_handle) {
                    Some(w) => w,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                            message: "instance wrapper not found".to_string(),
                        }
                    }
                };

                // Build queue create infos
                let queue_priorities: Vec<Vec<f32>> = queue_create_infos
                    .iter()
                    .map(|qi| qi.queue_priorities.clone())
                    .collect();

                let vk_queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = queue_create_infos
                    .iter()
                    .enumerate()
                    .map(|(i, qi)| {
                        vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(qi.queue_family_index)
                            .queue_priorities(&queue_priorities[i])
                    })
                    .collect();

                // Optionally set features
                let features = enabled_features.and_then(|raw| {
                    if raw.len() == std::mem::size_of::<vk::PhysicalDeviceFeatures>() {
                        Some(unsafe {
                            std::ptr::read(raw.as_ptr() as *const vk::PhysicalDeviceFeatures)
                        })
                    } else {
                        None
                    }
                });

                let mut device_create_info = vk::DeviceCreateInfo::default()
                    .queue_create_infos(&vk_queue_create_infos);

                if let Some(ref f) = features {
                    device_create_info = device_create_info.enabled_features(f);
                }

                match unsafe { wrapper.create_device(pd, &device_create_info, None) } {
                    Ok(device) => {
                        let handle = session.alloc_handle(ResourceType::VkDevice);
                        let raw = device.handle();
                        self.device_handles.insert(handle, raw);
                        self.device_wrappers.insert(handle, device);
                        self.device_to_instance.insert(handle, inst_handle);
                        info!("created Vulkan device: {:?}", handle);
                        VulkanResponse::DeviceCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyDevice { device } => {
                if let Some((_, dev)) = self.device_wrappers.remove(&device) {
                    unsafe { dev.destroy_device(None) };
                    self.device_handles.remove(&device);
                    self.device_to_instance.remove(&device);
                    session.remove_handle(&device);
                    debug!("destroyed Vulkan device: {:?}", device);
                }
                VulkanResponse::Success
            }

            VulkanCommand::DeviceWaitIdle { device } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };
                match unsafe { dev.device_wait_idle() } {
                    Ok(()) => VulkanResponse::Success,
                    Err(e) => Self::vk_err(e),
                }
            }

            // ── Queue ───────────────────────────────────────────
            VulkanCommand::GetDeviceQueue {
                device,
                queue_family_index,
                queue_index,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };
                let queue =
                    unsafe { dev.get_device_queue(queue_family_index, queue_index) };
                let handle = session.alloc_handle(ResourceType::VkQueue);
                self.queue_handles.insert(handle, queue);
                debug!(
                    "got queue family={} index={}: {:?}",
                    queue_family_index, queue_index, handle
                );
                VulkanResponse::QueueRetrieved { handle }
            }

            VulkanCommand::QueueSubmit {
                queue,
                submits,
                fence,
            } => {
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

            VulkanCommand::QueueWaitIdle { queue } => {
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

            // ── Memory ──────────────────────────────────────────
            VulkanCommand::AllocateMemory {
                device,
                alloc_size,
                memory_type_index,
            } => {
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

            VulkanCommand::FreeMemory { device, memory } => {
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

            VulkanCommand::MapMemory {
                device,
                memory,
                offset,
                size,
                flags: _,
            } => {
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

            VulkanCommand::UnmapMemory {
                device,
                memory,
                written_data,
                offset,
            } => {
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

            VulkanCommand::FlushMappedMemoryRanges {
                device,
                ranges,
                data,
            } => {
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

            VulkanCommand::InvalidateMappedMemoryRanges { device, ranges } => {
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

            // ── Buffer ──────────────────────────────────────────
            VulkanCommand::CreateBuffer {
                device,
                size,
                usage,
                sharing_mode,
                queue_family_indices,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let mut create_info = vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::from_raw(usage))
                    .sharing_mode(vk::SharingMode::from_raw(sharing_mode as i32));

                if !queue_family_indices.is_empty() {
                    create_info =
                        create_info.queue_family_indices(&queue_family_indices);
                }

                match unsafe { dev.create_buffer(&create_info, None) } {
                    Ok(buffer) => {
                        let handle = session.alloc_handle(ResourceType::VkBuffer);
                        self.buffer_handles.insert(handle, buffer);
                        self.buffer_to_device.insert(handle, device);
                        debug!("created buffer ({} bytes): {:?}", size, handle);
                        VulkanResponse::BufferCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyBuffer { device, buffer } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, buf)) = self.buffer_handles.remove(&buffer) {
                    unsafe { dev.destroy_buffer(buf, None) };
                    self.buffer_to_device.remove(&buffer);
                    session.remove_handle(&buffer);
                }
                VulkanResponse::Success
            }

            VulkanCommand::BindBufferMemory {
                device,
                buffer,
                memory,
                memory_offset,
            } => {
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
                let mem = match self.memory_handles.get(&memory) {
                    Some(m) => *m.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };
                match unsafe { dev.bind_buffer_memory(buf, mem, memory_offset) } {
                    Ok(()) => VulkanResponse::Success,
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::GetBufferMemoryRequirements { device, buffer } => {
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

            // ── Shader Module ───────────────────────────────────
            VulkanCommand::CreateShaderModule { device, code } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                // SPIR-V code must be aligned to 4 bytes and size must be multiple of 4
                if code.len() % 4 != 0 {
                    return VulkanResponse::Error {
                        code: vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                        message: "SPIR-V code size must be multiple of 4".to_string(),
                    };
                }

                let code_u32: &[u32] = unsafe {
                    std::slice::from_raw_parts(
                        code.as_ptr() as *const u32,
                        code.len() / 4,
                    )
                };

                let create_info = vk::ShaderModuleCreateInfo::default().code(code_u32);
                match unsafe { dev.create_shader_module(&create_info, None) } {
                    Ok(module) => {
                        let handle = session.alloc_handle(ResourceType::VkShaderModule);
                        self.shader_module_handles.insert(handle, module);
                        self.shader_to_device.insert(handle, device);
                        debug!("created shader module: {:?}", handle);
                        VulkanResponse::ShaderModuleCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyShaderModule {
                device,
                shader_module,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, module)) = self.shader_module_handles.remove(&shader_module) {
                    unsafe { dev.destroy_shader_module(module, None) };
                    self.shader_to_device.remove(&shader_module);
                    session.remove_handle(&shader_module);
                }
                VulkanResponse::Success
            }

            // ── Descriptor Set Layout ───────────────────────────
            VulkanCommand::CreateDescriptorSetLayout { device, bindings } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
                    .iter()
                    .map(|b| {
                        vk::DescriptorSetLayoutBinding::default()
                            .binding(b.binding)
                            .descriptor_type(vk::DescriptorType::from_raw(b.descriptor_type))
                            .descriptor_count(b.descriptor_count)
                            .stage_flags(vk::ShaderStageFlags::from_raw(b.stage_flags))
                    })
                    .collect();

                let create_info =
                    vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);

                match unsafe { dev.create_descriptor_set_layout(&create_info, None) } {
                    Ok(layout) => {
                        let handle = session.alloc_handle(ResourceType::VkDescriptorSetLayout);
                        self.desc_set_layout_handles.insert(handle, layout);
                        self.desc_set_layout_to_device.insert(handle, device);
                        VulkanResponse::DescriptorSetLayoutCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyDescriptorSetLayout { device, layout } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, l)) = self.desc_set_layout_handles.remove(&layout) {
                    unsafe { dev.destroy_descriptor_set_layout(l, None) };
                    self.desc_set_layout_to_device.remove(&layout);
                    session.remove_handle(&layout);
                }
                VulkanResponse::Success
            }

            // ── Pipeline Layout ─────────────────────────────────
            VulkanCommand::CreatePipelineLayout {
                device,
                set_layouts,
                push_constant_ranges,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let vk_layouts: Vec<vk::DescriptorSetLayout> = set_layouts
                    .iter()
                    .filter_map(|h| {
                        self.desc_set_layout_handles
                            .get(h)
                            .map(|v| *v.value())
                    })
                    .collect();

                let vk_push_ranges: Vec<vk::PushConstantRange> = push_constant_ranges
                    .iter()
                    .map(|p| {
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::from_raw(p.stage_flags))
                            .offset(p.offset)
                            .size(p.size)
                    })
                    .collect();

                let create_info = vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&vk_layouts)
                    .push_constant_ranges(&vk_push_ranges);

                match unsafe { dev.create_pipeline_layout(&create_info, None) } {
                    Ok(layout) => {
                        let handle = session.alloc_handle(ResourceType::VkPipelineLayout);
                        self.pipeline_layout_handles.insert(handle, layout);
                        self.pipeline_layout_to_device.insert(handle, device);
                        VulkanResponse::PipelineLayoutCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyPipelineLayout { device, layout } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, l)) = self.pipeline_layout_handles.remove(&layout) {
                    unsafe { dev.destroy_pipeline_layout(l, None) };
                    self.pipeline_layout_to_device.remove(&layout);
                    session.remove_handle(&layout);
                }
                VulkanResponse::Success
            }

            // ── Compute Pipeline ────────────────────────────────
            VulkanCommand::CreateComputePipelines {
                device,
                create_infos,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let entry_points: Vec<std::ffi::CString> = create_infos
                    .iter()
                    .map(|ci| {
                        std::ffi::CString::new(ci.stage.entry_point.as_str()).unwrap_or_default()
                    })
                    .collect();

                let vk_create_infos: Vec<vk::ComputePipelineCreateInfo> = create_infos
                    .iter()
                    .enumerate()
                    .map(|(i, ci)| {
                        let module = self
                            .shader_module_handles
                            .get(&ci.stage.module)
                            .map(|v| *v.value())
                            .unwrap_or(vk::ShaderModule::null());
                        let layout = self
                            .pipeline_layout_handles
                            .get(&ci.layout)
                            .map(|v| *v.value())
                            .unwrap_or(vk::PipelineLayout::null());

                        let stage = vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::from_raw(ci.stage.stage))
                            .module(module)
                            .name(entry_points[i].as_c_str());

                        vk::ComputePipelineCreateInfo::default()
                            .stage(stage)
                            .layout(layout)
                            .flags(vk::PipelineCreateFlags::from_raw(ci.flags))
                    })
                    .collect();

                match unsafe {
                    dev.create_compute_pipelines(
                        vk::PipelineCache::null(),
                        &vk_create_infos,
                        None,
                    )
                } {
                    Ok(pipelines) => {
                        let mut handles = Vec::new();
                        for pipeline in pipelines {
                            let handle = session.alloc_handle(ResourceType::VkPipeline);
                            self.pipeline_handles.insert(handle, pipeline);
                            self.pipeline_to_device.insert(handle, device);
                            handles.push(handle);
                        }
                        VulkanResponse::PipelinesCreated { handles }
                    }
                    Err((_, e)) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyPipeline { device, pipeline } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, p)) = self.pipeline_handles.remove(&pipeline) {
                    unsafe { dev.destroy_pipeline(p, None) };
                    self.pipeline_to_device.remove(&pipeline);
                    session.remove_handle(&pipeline);
                }
                VulkanResponse::Success
            }

            // ── Descriptor Pool ─────────────────────────────────
            VulkanCommand::CreateDescriptorPool {
                device,
                max_sets,
                pool_sizes,
                flags,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let vk_pool_sizes: Vec<vk::DescriptorPoolSize> = pool_sizes
                    .iter()
                    .map(|ps| {
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::from_raw(ps.descriptor_type))
                            .descriptor_count(ps.descriptor_count)
                    })
                    .collect();

                let create_info = vk::DescriptorPoolCreateInfo::default()
                    .max_sets(max_sets)
                    .pool_sizes(&vk_pool_sizes)
                    .flags(vk::DescriptorPoolCreateFlags::from_raw(flags));

                match unsafe { dev.create_descriptor_pool(&create_info, None) } {
                    Ok(pool) => {
                        let handle = session.alloc_handle(ResourceType::VkDescriptorPool);
                        self.desc_pool_handles.insert(handle, pool);
                        self.desc_pool_to_device.insert(handle, device);
                        VulkanResponse::DescriptorPoolCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyDescriptorPool { device, pool } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, p)) = self.desc_pool_handles.remove(&pool) {
                    unsafe { dev.destroy_descriptor_pool(p, None) };
                    self.desc_pool_to_device.remove(&pool);
                    session.remove_handle(&pool);
                }
                VulkanResponse::Success
            }

            // ── Descriptor Set ──────────────────────────────────
            VulkanCommand::AllocateDescriptorSets {
                device,
                descriptor_pool,
                set_layouts,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };
                let pool = match self.desc_pool_handles.get(&descriptor_pool) {
                    Some(p) => *p.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid descriptor pool handle".to_string(),
                        }
                    }
                };

                let vk_layouts: Vec<vk::DescriptorSetLayout> = set_layouts
                    .iter()
                    .filter_map(|h| {
                        self.desc_set_layout_handles
                            .get(h)
                            .map(|v| *v.value())
                    })
                    .collect();

                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&vk_layouts);

                match unsafe { dev.allocate_descriptor_sets(&alloc_info) } {
                    Ok(sets) => {
                        let mut handles = Vec::new();
                        for set in sets {
                            let handle = session.alloc_handle(ResourceType::VkDescriptorSet);
                            self.desc_set_handles.insert(handle, set);
                            handles.push(handle);
                        }
                        VulkanResponse::DescriptorSetsAllocated { handles }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::FreeDescriptorSets {
                device,
                descriptor_pool,
                descriptor_sets,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                let pool = match self.desc_pool_handles.get(&descriptor_pool) {
                    Some(p) => *p.value(),
                    None => return VulkanResponse::Success,
                };

                let vk_sets: Vec<vk::DescriptorSet> = descriptor_sets
                    .iter()
                    .filter_map(|h| {
                        self.desc_set_handles
                            .remove(h)
                            .map(|(_, s)| {
                                session.remove_handle(h);
                                s
                            })
                    })
                    .collect();

                if !vk_sets.is_empty() {
                    let _ = unsafe { dev.free_descriptor_sets(pool, &vk_sets) };
                }
                VulkanResponse::Success
            }

            VulkanCommand::UpdateDescriptorSets { device, writes } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                // Build buffer info arrays (must live long enough)
                let buffer_info_vecs: Vec<Vec<vk::DescriptorBufferInfo>> = writes
                    .iter()
                    .map(|w| {
                        w.buffer_infos
                            .iter()
                            .map(|bi| {
                                let buffer = self
                                    .buffer_handles
                                    .get(&bi.buffer)
                                    .map(|v| *v.value())
                                    .unwrap_or(vk::Buffer::null());
                                vk::DescriptorBufferInfo::default()
                                    .buffer(buffer)
                                    .offset(bi.offset)
                                    .range(bi.range)
                            })
                            .collect()
                    })
                    .collect();

                let vk_writes: Vec<vk::WriteDescriptorSet> = writes
                    .iter()
                    .enumerate()
                    .map(|(i, w)| {
                        let dst_set = self
                            .desc_set_handles
                            .get(&w.dst_set)
                            .map(|v| *v.value())
                            .unwrap_or(vk::DescriptorSet::null());

                        vk::WriteDescriptorSet::default()
                            .dst_set(dst_set)
                            .dst_binding(w.dst_binding)
                            .dst_array_element(w.dst_array_element)
                            .descriptor_type(vk::DescriptorType::from_raw(w.descriptor_type))
                            .buffer_info(&buffer_info_vecs[i])
                    })
                    .collect();

                unsafe { dev.update_descriptor_sets(&vk_writes, &[]) };
                VulkanResponse::Success
            }

            // ── Command Pool ────────────────────────────────────
            VulkanCommand::CreateCommandPool {
                device,
                queue_family_index,
                flags,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let create_info = vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::from_raw(flags));

                match unsafe { dev.create_command_pool(&create_info, None) } {
                    Ok(pool) => {
                        let handle = session.alloc_handle(ResourceType::VkCommandPool);
                        self.command_pool_handles.insert(handle, pool);
                        self.command_pool_to_device.insert(handle, device);
                        VulkanResponse::CommandPoolCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyCommandPool {
                device,
                command_pool,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, pool)) = self.command_pool_handles.remove(&command_pool) {
                    unsafe { dev.destroy_command_pool(pool, None) };
                    self.command_pool_to_device.remove(&command_pool);
                    session.remove_handle(&command_pool);
                }
                VulkanResponse::Success
            }

            VulkanCommand::ResetCommandPool {
                device,
                command_pool,
                flags,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };
                let pool = match self.command_pool_handles.get(&command_pool) {
                    Some(p) => *p.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid command pool handle".to_string(),
                        }
                    }
                };
                match unsafe {
                    dev.reset_command_pool(pool, vk::CommandPoolResetFlags::from_raw(flags))
                } {
                    Ok(()) => VulkanResponse::Success,
                    Err(e) => Self::vk_err(e),
                }
            }

            // ── Command Buffer ──────────────────────────────────
            VulkanCommand::AllocateCommandBuffers {
                device,
                command_pool,
                level,
                count,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };
                let pool = match self.command_pool_handles.get(&command_pool) {
                    Some(p) => *p.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid command pool handle".to_string(),
                        }
                    }
                };

                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::from_raw(level as i32))
                    .command_buffer_count(count);

                match unsafe { dev.allocate_command_buffers(&alloc_info) } {
                    Ok(cmd_bufs) => {
                        let mut handles = Vec::new();
                        for cb in cmd_bufs {
                            let handle = session.alloc_handle(ResourceType::VkCommandBuffer);
                            self.command_buffer_handles.insert(handle, cb);
                            self.command_buffer_to_device.insert(handle, device);
                            handles.push(handle);
                        }
                        VulkanResponse::CommandBuffersAllocated { handles }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::FreeCommandBuffers {
                device,
                command_pool,
                command_buffers,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                let pool = match self.command_pool_handles.get(&command_pool) {
                    Some(p) => *p.value(),
                    None => return VulkanResponse::Success,
                };

                let vk_bufs: Vec<vk::CommandBuffer> = command_buffers
                    .iter()
                    .filter_map(|h| {
                        self.command_buffer_handles
                            .remove(h)
                            .map(|(_, cb)| {
                                self.command_buffer_to_device.remove(h);
                                session.remove_handle(h);
                                cb
                            })
                    })
                    .collect();

                if !vk_bufs.is_empty() {
                    unsafe { dev.free_command_buffers(pool, &vk_bufs) };
                }
                VulkanResponse::Success
            }

            // ── Command Buffer Recording (Batched) ──────────────
            VulkanCommand::SubmitRecordedCommands {
                command_buffer,
                commands,
            } => {
                let cb = match self.command_buffer_handles.get(&command_buffer) {
                    Some(c) => *c.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid command buffer handle".to_string(),
                        }
                    }
                };

                // Find the device for this command buffer
                let dev_handle = match self.command_buffer_to_device.get(&command_buffer) {
                    Some(d) => *d.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "command buffer has no associated device".to_string(),
                        }
                    }
                };
                let dev = match self.device_wrappers.get(&dev_handle) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "device not found".to_string(),
                        }
                    }
                };

                // Begin command buffer
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

                if let Err(e) = unsafe { dev.begin_command_buffer(cb, &begin_info) } {
                    return Self::vk_err(e);
                }

                // Replay all recorded commands
                for cmd in &commands {
                    match cmd {
                        RecordedCommand::BindPipeline {
                            pipeline_bind_point,
                            pipeline,
                        } => {
                            let p = match self.pipeline_handles.get(pipeline) {
                                Some(p) => *p.value(),
                                None => continue,
                            };
                            unsafe {
                                dev.cmd_bind_pipeline(
                                    cb,
                                    vk::PipelineBindPoint::from_raw(*pipeline_bind_point as i32),
                                    p,
                                );
                            }
                        }

                        RecordedCommand::BindDescriptorSets {
                            pipeline_bind_point,
                            layout,
                            first_set,
                            descriptor_sets,
                            dynamic_offsets,
                        } => {
                            let pl = match self.pipeline_layout_handles.get(layout) {
                                Some(l) => *l.value(),
                                None => continue,
                            };
                            let sets: Vec<vk::DescriptorSet> = descriptor_sets
                                .iter()
                                .filter_map(|h| {
                                    self.desc_set_handles.get(h).map(|v| *v.value())
                                })
                                .collect();
                            unsafe {
                                dev.cmd_bind_descriptor_sets(
                                    cb,
                                    vk::PipelineBindPoint::from_raw(
                                        *pipeline_bind_point as i32,
                                    ),
                                    pl,
                                    *first_set,
                                    &sets,
                                    dynamic_offsets,
                                );
                            }
                        }

                        RecordedCommand::Dispatch {
                            group_count_x,
                            group_count_y,
                            group_count_z,
                        } => unsafe {
                            dev.cmd_dispatch(
                                cb,
                                *group_count_x,
                                *group_count_y,
                                *group_count_z,
                            );
                        },

                        RecordedCommand::PipelineBarrier {
                            src_stage_mask,
                            dst_stage_mask,
                            dependency_flags,
                            memory_barriers,
                            buffer_memory_barriers,
                            image_memory_barriers,
                        } => {
                            let vk_mem_barriers: Vec<vk::MemoryBarrier> = memory_barriers
                                .iter()
                                .map(|mb| {
                                    vk::MemoryBarrier::default()
                                        .src_access_mask(vk::AccessFlags::from_raw(
                                            mb.src_access_mask,
                                        ))
                                        .dst_access_mask(vk::AccessFlags::from_raw(
                                            mb.dst_access_mask,
                                        ))
                                })
                                .collect();

                            let vk_buf_barriers: Vec<vk::BufferMemoryBarrier> =
                                buffer_memory_barriers
                                    .iter()
                                    .map(|bmb| {
                                        let buffer = self
                                            .buffer_handles
                                            .get(&bmb.buffer)
                                            .map(|v| *v.value())
                                            .unwrap_or(vk::Buffer::null());
                                        vk::BufferMemoryBarrier::default()
                                            .src_access_mask(vk::AccessFlags::from_raw(
                                                bmb.src_access_mask,
                                            ))
                                            .dst_access_mask(vk::AccessFlags::from_raw(
                                                bmb.dst_access_mask,
                                            ))
                                            .src_queue_family_index(bmb.src_queue_family_index)
                                            .dst_queue_family_index(bmb.dst_queue_family_index)
                                            .buffer(buffer)
                                            .offset(bmb.offset)
                                            .size(bmb.size)
                                    })
                                    .collect();

                            let vk_img_barriers: Vec<vk::ImageMemoryBarrier> =
                                image_memory_barriers
                                    .iter()
                                    .map(|imb| {
                                        let image = self
                                            .image_handles
                                            .get(&imb.image)
                                            .map(|v| *v.value())
                                            .unwrap_or(vk::Image::null());
                                        vk::ImageMemoryBarrier::default()
                                            .src_access_mask(vk::AccessFlags::from_raw(
                                                imb.src_access_mask,
                                            ))
                                            .dst_access_mask(vk::AccessFlags::from_raw(
                                                imb.dst_access_mask,
                                            ))
                                            .old_layout(vk::ImageLayout::from_raw(imb.old_layout))
                                            .new_layout(vk::ImageLayout::from_raw(imb.new_layout))
                                            .src_queue_family_index(imb.src_queue_family_index)
                                            .dst_queue_family_index(imb.dst_queue_family_index)
                                            .image(image)
                                            .subresource_range(vk::ImageSubresourceRange {
                                                aspect_mask: vk::ImageAspectFlags::from_raw(
                                                    imb.subresource_range.aspect_mask,
                                                ),
                                                base_mip_level: imb.subresource_range.base_mip_level,
                                                level_count: imb.subresource_range.level_count,
                                                base_array_layer: imb
                                                    .subresource_range
                                                    .base_array_layer,
                                                layer_count: imb.subresource_range.layer_count,
                                            })
                                    })
                                    .collect();

                            unsafe {
                                dev.cmd_pipeline_barrier(
                                    cb,
                                    vk::PipelineStageFlags::from_raw(*src_stage_mask),
                                    vk::PipelineStageFlags::from_raw(*dst_stage_mask),
                                    vk::DependencyFlags::from_raw(*dependency_flags),
                                    &vk_mem_barriers,
                                    &vk_buf_barriers,
                                    &vk_img_barriers,
                                );
                            }
                        }

                        RecordedCommand::CopyBuffer { src, dst, regions } => {
                            let src_buf = match self.buffer_handles.get(src) {
                                Some(b) => *b.value(),
                                None => continue,
                            };
                            let dst_buf = match self.buffer_handles.get(dst) {
                                Some(b) => *b.value(),
                                None => continue,
                            };
                            let vk_regions: Vec<vk::BufferCopy> = regions
                                .iter()
                                .map(|r| {
                                    vk::BufferCopy::default()
                                        .src_offset(r.src_offset)
                                        .dst_offset(r.dst_offset)
                                        .size(r.size)
                                })
                                .collect();
                            unsafe { dev.cmd_copy_buffer(cb, src_buf, dst_buf, &vk_regions) };
                        }

                        RecordedCommand::FillBuffer {
                            buffer,
                            offset,
                            size,
                            data,
                        } => {
                            let buf = match self.buffer_handles.get(buffer) {
                                Some(b) => *b.value(),
                                None => continue,
                            };
                            unsafe { dev.cmd_fill_buffer(cb, buf, *offset, *size, *data) };
                        }

                        RecordedCommand::UpdateBuffer {
                            buffer,
                            offset,
                            data,
                        } => {
                            let buf = match self.buffer_handles.get(buffer) {
                                Some(b) => *b.value(),
                                None => continue,
                            };
                            unsafe { dev.cmd_update_buffer(cb, buf, *offset, data) };
                        }

                        RecordedCommand::BeginRenderPass {
                            render_pass,
                            framebuffer,
                            render_area,
                            clear_values,
                            contents,
                        } => {
                            let rp = match self.render_pass_handles.get(render_pass) {
                                Some(r) => *r.value(),
                                None => continue,
                            };
                            let fb = match self.framebuffer_handles.get(framebuffer) {
                                Some(f) => *f.value(),
                                None => continue,
                            };
                            let vk_clear_values: Vec<vk::ClearValue> = clear_values
                                .iter()
                                .map(|cv| unsafe {
                                    std::mem::transmute::<[u8; 16], vk::ClearValue>(cv.data)
                                })
                                .collect();
                            let begin_info = vk::RenderPassBeginInfo::default()
                                .render_pass(rp)
                                .framebuffer(fb)
                                .render_area(vk::Rect2D {
                                    offset: vk::Offset2D {
                                        x: render_area.offset[0],
                                        y: render_area.offset[1],
                                    },
                                    extent: vk::Extent2D {
                                        width: render_area.extent[0],
                                        height: render_area.extent[1],
                                    },
                                })
                                .clear_values(&vk_clear_values);
                            unsafe {
                                dev.cmd_begin_render_pass(
                                    cb,
                                    &begin_info,
                                    vk::SubpassContents::from_raw(*contents as i32),
                                );
                            }
                        }

                        RecordedCommand::EndRenderPass => unsafe {
                            dev.cmd_end_render_pass(cb);
                        },

                        RecordedCommand::Draw {
                            vertex_count,
                            instance_count,
                            first_vertex,
                            first_instance,
                        } => unsafe {
                            dev.cmd_draw(
                                cb,
                                *vertex_count,
                                *instance_count,
                                *first_vertex,
                                *first_instance,
                            );
                        },

                        RecordedCommand::DrawIndexed {
                            index_count,
                            instance_count,
                            first_index,
                            vertex_offset,
                            first_instance,
                        } => unsafe {
                            dev.cmd_draw_indexed(
                                cb,
                                *index_count,
                                *instance_count,
                                *first_index,
                                *vertex_offset,
                                *first_instance,
                            );
                        },

                        RecordedCommand::BindVertexBuffers {
                            first_binding,
                            buffers,
                            offsets,
                        } => {
                            let vk_bufs: Vec<vk::Buffer> = buffers
                                .iter()
                                .filter_map(|h| self.buffer_handles.get(h).map(|v| *v.value()))
                                .collect();
                            unsafe {
                                dev.cmd_bind_vertex_buffers(cb, *first_binding, &vk_bufs, offsets);
                            }
                        }

                        RecordedCommand::BindIndexBuffer {
                            buffer,
                            offset,
                            index_type,
                        } => {
                            let buf = match self.buffer_handles.get(buffer) {
                                Some(b) => *b.value(),
                                None => continue,
                            };
                            unsafe {
                                dev.cmd_bind_index_buffer(
                                    cb,
                                    buf,
                                    *offset,
                                    vk::IndexType::from_raw(*index_type as i32),
                                );
                            }
                        }

                        RecordedCommand::SetViewport {
                            first_viewport,
                            viewports,
                        } => {
                            let vk_viewports: Vec<vk::Viewport> = viewports
                                .iter()
                                .map(|vp| vk::Viewport {
                                    x: vp.x,
                                    y: vp.y,
                                    width: vp.width,
                                    height: vp.height,
                                    min_depth: vp.min_depth,
                                    max_depth: vp.max_depth,
                                })
                                .collect();
                            unsafe {
                                dev.cmd_set_viewport(cb, *first_viewport, &vk_viewports);
                            }
                        }

                        RecordedCommand::SetScissor {
                            first_scissor,
                            scissors,
                        } => {
                            let vk_scissors: Vec<vk::Rect2D> = scissors
                                .iter()
                                .map(|s| vk::Rect2D {
                                    offset: vk::Offset2D {
                                        x: s.offset[0],
                                        y: s.offset[1],
                                    },
                                    extent: vk::Extent2D {
                                        width: s.extent[0],
                                        height: s.extent[1],
                                    },
                                })
                                .collect();
                            unsafe {
                                dev.cmd_set_scissor(cb, *first_scissor, &vk_scissors);
                            }
                        }

                        RecordedCommand::CopyBufferToImage {
                            src_buffer,
                            dst_image,
                            dst_image_layout,
                            regions,
                        } => {
                            let buf = match self.buffer_handles.get(src_buffer) {
                                Some(b) => *b.value(),
                                None => continue,
                            };
                            let img = match self.image_handles.get(dst_image) {
                                Some(i) => *i.value(),
                                None => continue,
                            };
                            let vk_regions: Vec<vk::BufferImageCopy> = regions
                                .iter()
                                .map(|r| vk::BufferImageCopy {
                                    buffer_offset: r.buffer_offset,
                                    buffer_row_length: r.buffer_row_length,
                                    buffer_image_height: r.buffer_image_height,
                                    image_subresource: vk::ImageSubresourceLayers {
                                        aspect_mask: vk::ImageAspectFlags::from_raw(
                                            r.image_subresource.aspect_mask,
                                        ),
                                        mip_level: r.image_subresource.mip_level,
                                        base_array_layer: r.image_subresource.base_array_layer,
                                        layer_count: r.image_subresource.layer_count,
                                    },
                                    image_offset: vk::Offset3D {
                                        x: r.image_offset[0],
                                        y: r.image_offset[1],
                                        z: r.image_offset[2],
                                    },
                                    image_extent: vk::Extent3D {
                                        width: r.image_extent[0],
                                        height: r.image_extent[1],
                                        depth: r.image_extent[2],
                                    },
                                })
                                .collect();
                            unsafe {
                                dev.cmd_copy_buffer_to_image(
                                    cb,
                                    buf,
                                    img,
                                    vk::ImageLayout::from_raw(*dst_image_layout),
                                    &vk_regions,
                                );
                            }
                        }

                        RecordedCommand::CopyImageToBuffer {
                            src_image,
                            src_image_layout,
                            dst_buffer,
                            regions,
                        } => {
                            let img = match self.image_handles.get(src_image) {
                                Some(i) => *i.value(),
                                None => continue,
                            };
                            let buf = match self.buffer_handles.get(dst_buffer) {
                                Some(b) => *b.value(),
                                None => continue,
                            };
                            let vk_regions: Vec<vk::BufferImageCopy> = regions
                                .iter()
                                .map(|r| vk::BufferImageCopy {
                                    buffer_offset: r.buffer_offset,
                                    buffer_row_length: r.buffer_row_length,
                                    buffer_image_height: r.buffer_image_height,
                                    image_subresource: vk::ImageSubresourceLayers {
                                        aspect_mask: vk::ImageAspectFlags::from_raw(
                                            r.image_subresource.aspect_mask,
                                        ),
                                        mip_level: r.image_subresource.mip_level,
                                        base_array_layer: r.image_subresource.base_array_layer,
                                        layer_count: r.image_subresource.layer_count,
                                    },
                                    image_offset: vk::Offset3D {
                                        x: r.image_offset[0],
                                        y: r.image_offset[1],
                                        z: r.image_offset[2],
                                    },
                                    image_extent: vk::Extent3D {
                                        width: r.image_extent[0],
                                        height: r.image_extent[1],
                                        depth: r.image_extent[2],
                                    },
                                })
                                .collect();
                            unsafe {
                                dev.cmd_copy_image_to_buffer(
                                    cb,
                                    img,
                                    vk::ImageLayout::from_raw(*src_image_layout),
                                    buf,
                                    &vk_regions,
                                );
                            }
                        }
                    }
                }

                // End command buffer
                if let Err(e) = unsafe { dev.end_command_buffer(cb) } {
                    return Self::vk_err(e);
                }

                VulkanResponse::Success
            }

            // ── Fence ───────────────────────────────────────────
            VulkanCommand::CreateFence { device, signaled } => {
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

            VulkanCommand::DestroyFence { device, fence } => {
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

            VulkanCommand::WaitForFences {
                device,
                fences,
                wait_all,
                timeout_ns,
            } => {
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

            VulkanCommand::ResetFences { device, fences } => {
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

            VulkanCommand::GetFenceStatus { device, fence } => {
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

            // ── Image ──────────────────────────────────────────────
            VulkanCommand::CreateImage { device, create_info } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let ci = &create_info;
                let qfi = &ci.queue_family_indices;
                let mut image_ci = vk::ImageCreateInfo::default()
                    .flags(vk::ImageCreateFlags::from_raw(ci.flags))
                    .image_type(vk::ImageType::from_raw(ci.image_type))
                    .format(vk::Format::from_raw(ci.format))
                    .extent(vk::Extent3D {
                        width: ci.extent[0],
                        height: ci.extent[1],
                        depth: ci.extent[2],
                    })
                    .mip_levels(ci.mip_levels)
                    .array_layers(ci.array_layers)
                    .samples(vk::SampleCountFlags::from_raw(ci.samples))
                    .tiling(vk::ImageTiling::from_raw(ci.tiling))
                    .usage(vk::ImageUsageFlags::from_raw(ci.usage))
                    .sharing_mode(vk::SharingMode::from_raw(ci.sharing_mode))
                    .initial_layout(vk::ImageLayout::from_raw(ci.initial_layout));
                if !qfi.is_empty() {
                    image_ci = image_ci.queue_family_indices(qfi);
                }

                match unsafe { dev.create_image(&image_ci, None) } {
                    Ok(image) => {
                        let handle = session.alloc_handle(ResourceType::VkImage);
                        self.image_handles.insert(handle, image);
                        self.image_to_device.insert(handle, device);
                        debug!("created image: {:?}", handle);
                        VulkanResponse::ImageCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyImage { device, image } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, img)) = self.image_handles.remove(&image) {
                    unsafe { dev.destroy_image(img, None) };
                    self.image_to_device.remove(&image);
                    session.remove_handle(&image);
                }
                VulkanResponse::Success
            }

            VulkanCommand::GetImageMemoryRequirements { device, image } => {
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

            VulkanCommand::BindImageMemory {
                device,
                image,
                memory,
                memory_offset,
            } => {
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
                let mem = match self.memory_handles.get(&memory) {
                    Some(m) => *m.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid memory handle".to_string(),
                        }
                    }
                };
                match unsafe { dev.bind_image_memory(img, mem, memory_offset) } {
                    Ok(()) => VulkanResponse::Success,
                    Err(e) => Self::vk_err(e),
                }
            }

            // ── Image View ─────────────────────────────────────────
            VulkanCommand::CreateImageView {
                device,
                image,
                view_type,
                format,
                components,
                subresource_range,
            } => {
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
                let ci = vk::ImageViewCreateInfo::default()
                    .image(img)
                    .view_type(vk::ImageViewType::from_raw(view_type))
                    .format(vk::Format::from_raw(format))
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::from_raw(components.r),
                        g: vk::ComponentSwizzle::from_raw(components.g),
                        b: vk::ComponentSwizzle::from_raw(components.b),
                        a: vk::ComponentSwizzle::from_raw(components.a),
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::from_raw(subresource_range.aspect_mask),
                        base_mip_level: subresource_range.base_mip_level,
                        level_count: subresource_range.level_count,
                        base_array_layer: subresource_range.base_array_layer,
                        layer_count: subresource_range.layer_count,
                    });

                match unsafe { dev.create_image_view(&ci, None) } {
                    Ok(view) => {
                        let handle = session.alloc_handle(ResourceType::VkImageView);
                        self.image_view_handles.insert(handle, view);
                        self.image_view_to_device.insert(handle, device);
                        debug!("created image view: {:?}", handle);
                        VulkanResponse::ImageViewCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyImageView { device, image_view } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, view)) = self.image_view_handles.remove(&image_view) {
                    unsafe { dev.destroy_image_view(view, None) };
                    self.image_view_to_device.remove(&image_view);
                    session.remove_handle(&image_view);
                }
                VulkanResponse::Success
            }

            // ── Render Pass ────────────────────────────────────────
            VulkanCommand::CreateRenderPass {
                device,
                attachments,
                subpasses,
                dependencies,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                let vk_attachments: Vec<vk::AttachmentDescription> = attachments
                    .iter()
                    .map(|a| {
                        vk::AttachmentDescription::default()
                            .flags(vk::AttachmentDescriptionFlags::from_raw(a.flags))
                            .format(vk::Format::from_raw(a.format))
                            .samples(vk::SampleCountFlags::from_raw(a.samples))
                            .load_op(vk::AttachmentLoadOp::from_raw(a.load_op))
                            .store_op(vk::AttachmentStoreOp::from_raw(a.store_op))
                            .stencil_load_op(vk::AttachmentLoadOp::from_raw(a.stencil_load_op))
                            .stencil_store_op(vk::AttachmentStoreOp::from_raw(a.stencil_store_op))
                            .initial_layout(vk::ImageLayout::from_raw(a.initial_layout))
                            .final_layout(vk::ImageLayout::from_raw(a.final_layout))
                    })
                    .collect();

                // Build subpass reference arrays - must be kept alive
                let mut input_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();
                let mut color_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();
                let mut resolve_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();
                let mut ds_refs: Vec<Option<vk::AttachmentReference>> = Vec::new();

                for sp in &subpasses {
                    input_refs.push(
                        sp.input_attachments
                            .iter()
                            .map(|r| vk::AttachmentReference {
                                attachment: r.attachment,
                                layout: vk::ImageLayout::from_raw(r.layout),
                            })
                            .collect(),
                    );
                    color_refs.push(
                        sp.color_attachments
                            .iter()
                            .map(|r| vk::AttachmentReference {
                                attachment: r.attachment,
                                layout: vk::ImageLayout::from_raw(r.layout),
                            })
                            .collect(),
                    );
                    resolve_refs.push(
                        sp.resolve_attachments
                            .iter()
                            .map(|r| vk::AttachmentReference {
                                attachment: r.attachment,
                                layout: vk::ImageLayout::from_raw(r.layout),
                            })
                            .collect(),
                    );
                    ds_refs.push(sp.depth_stencil_attachment.as_ref().map(|r| {
                        vk::AttachmentReference {
                            attachment: r.attachment,
                            layout: vk::ImageLayout::from_raw(r.layout),
                        }
                    }));
                }

                let mut vk_subpasses: Vec<vk::SubpassDescription> = Vec::new();
                for (i, sp) in subpasses.iter().enumerate() {
                    let mut desc = vk::SubpassDescription::default()
                        .flags(vk::SubpassDescriptionFlags::from_raw(sp.flags))
                        .pipeline_bind_point(vk::PipelineBindPoint::from_raw(
                            sp.pipeline_bind_point,
                        ))
                        .input_attachments(&input_refs[i])
                        .color_attachments(&color_refs[i])
                        .preserve_attachments(&sp.preserve_attachments);
                    if !resolve_refs[i].is_empty() {
                        desc = desc.resolve_attachments(&resolve_refs[i]);
                    }
                    if let Some(ref ds) = ds_refs[i] {
                        desc = desc.depth_stencil_attachment(ds);
                    }
                    vk_subpasses.push(desc);
                }

                let vk_dependencies: Vec<vk::SubpassDependency> = dependencies
                    .iter()
                    .map(|d| {
                        vk::SubpassDependency::default()
                            .src_subpass(d.src_subpass)
                            .dst_subpass(d.dst_subpass)
                            .src_stage_mask(vk::PipelineStageFlags::from_raw(d.src_stage_mask))
                            .dst_stage_mask(vk::PipelineStageFlags::from_raw(d.dst_stage_mask))
                            .src_access_mask(vk::AccessFlags::from_raw(d.src_access_mask))
                            .dst_access_mask(vk::AccessFlags::from_raw(d.dst_access_mask))
                            .dependency_flags(vk::DependencyFlags::from_raw(d.dependency_flags))
                    })
                    .collect();

                let rp_ci = vk::RenderPassCreateInfo::default()
                    .attachments(&vk_attachments)
                    .subpasses(&vk_subpasses)
                    .dependencies(&vk_dependencies);

                match unsafe { dev.create_render_pass(&rp_ci, None) } {
                    Ok(rp) => {
                        let handle = session.alloc_handle(ResourceType::VkRenderPass);
                        self.render_pass_handles.insert(handle, rp);
                        self.render_pass_to_device.insert(handle, device);
                        debug!("created render pass: {:?}", handle);
                        VulkanResponse::RenderPassCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyRenderPass { device, render_pass } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, rp)) = self.render_pass_handles.remove(&render_pass) {
                    unsafe { dev.destroy_render_pass(rp, None) };
                    self.render_pass_to_device.remove(&render_pass);
                    session.remove_handle(&render_pass);
                }
                VulkanResponse::Success
            }

            // ── Framebuffer ────────────────────────────────────────
            VulkanCommand::CreateFramebuffer {
                device,
                render_pass,
                attachments: attachment_handles,
                width,
                height,
                layers,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };
                let rp = match self.render_pass_handles.get(&render_pass) {
                    Some(r) => *r.value(),
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid render pass handle".to_string(),
                        }
                    }
                };

                let vk_attachments: Vec<vk::ImageView> = attachment_handles
                    .iter()
                    .filter_map(|h| self.image_view_handles.get(h).map(|v| *v.value()))
                    .collect();

                let fb_ci = vk::FramebufferCreateInfo::default()
                    .render_pass(rp)
                    .attachments(&vk_attachments)
                    .width(width)
                    .height(height)
                    .layers(layers);

                match unsafe { dev.create_framebuffer(&fb_ci, None) } {
                    Ok(fb) => {
                        let handle = session.alloc_handle(ResourceType::VkFramebuffer);
                        self.framebuffer_handles.insert(handle, fb);
                        self.framebuffer_to_device.insert(handle, device);
                        debug!("created framebuffer: {:?}", handle);
                        VulkanResponse::FramebufferCreated { handle }
                    }
                    Err(e) => Self::vk_err(e),
                }
            }

            VulkanCommand::DestroyFramebuffer {
                device,
                framebuffer,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => return VulkanResponse::Success,
                };
                if let Some((_, fb)) = self.framebuffer_handles.remove(&framebuffer) {
                    unsafe { dev.destroy_framebuffer(fb, None) };
                    self.framebuffer_to_device.remove(&framebuffer);
                    session.remove_handle(&framebuffer);
                }
                VulkanResponse::Success
            }

            // ── Graphics Pipeline ──────────────────────────────────
            VulkanCommand::CreateGraphicsPipelines {
                device,
                create_infos,
            } => {
                let dev = match self.device_wrappers.get(&device) {
                    Some(d) => d,
                    None => {
                        return VulkanResponse::Error {
                            code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                            message: "invalid device handle".to_string(),
                        }
                    }
                };

                // Build all pipeline create infos - careful with lifetimes
                let mut vk_create_infos: Vec<vk::GraphicsPipelineCreateInfo> = Vec::new();

                // Keep all intermediate data alive
                let mut all_stages: Vec<Vec<vk::PipelineShaderStageCreateInfo>> = Vec::new();
                let mut all_entry_points: Vec<Vec<std::ffi::CString>> = Vec::new();
                let mut all_vi_bindings: Vec<Vec<vk::VertexInputBindingDescription>> = Vec::new();
                let mut all_vi_attrs: Vec<Vec<vk::VertexInputAttributeDescription>> = Vec::new();
                let mut all_vi_states: Vec<vk::PipelineVertexInputStateCreateInfo> = Vec::new();
                let mut all_ia_states: Vec<vk::PipelineInputAssemblyStateCreateInfo> = Vec::new();
                let mut all_viewports: Vec<Vec<vk::Viewport>> = Vec::new();
                let mut all_scissors: Vec<Vec<vk::Rect2D>> = Vec::new();
                let mut all_vp_states: Vec<vk::PipelineViewportStateCreateInfo> = Vec::new();
                let mut all_rs_states: Vec<vk::PipelineRasterizationStateCreateInfo> = Vec::new();
                let mut all_ms_states: Vec<vk::PipelineMultisampleStateCreateInfo> = Vec::new();
                let mut all_ds_states: Vec<vk::PipelineDepthStencilStateCreateInfo> = Vec::new();
                let mut all_cb_attachments: Vec<Vec<vk::PipelineColorBlendAttachmentState>> =
                    Vec::new();
                let mut all_cb_states: Vec<vk::PipelineColorBlendStateCreateInfo> = Vec::new();
                let mut all_dyn_states_raw: Vec<Vec<vk::DynamicState>> = Vec::new();
                let mut all_dyn_states: Vec<vk::PipelineDynamicStateCreateInfo> = Vec::new();

                // Phase 1: Collect all raw data into Vecs (no references created yet)
                for ci in &create_infos {
                    // Shader stages - collect entry points
                    let mut entry_points = Vec::new();
                    for stage in &ci.stages {
                        if self.shader_module_handles.get(&stage.module).is_none() {
                            return VulkanResponse::Error {
                                code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                                message: "invalid shader module handle".to_string(),
                            };
                        }
                        let ep =
                            std::ffi::CString::new(stage.entry_point.as_str()).unwrap_or_default();
                        entry_points.push(ep);
                    }
                    all_entry_points.push(entry_points);

                    // Vertex input
                    let vi_bindings: Vec<vk::VertexInputBindingDescription> = ci
                        .vertex_input_state
                        .vertex_binding_descriptions
                        .iter()
                        .map(|b| vk::VertexInputBindingDescription {
                            binding: b.binding,
                            stride: b.stride,
                            input_rate: vk::VertexInputRate::from_raw(b.input_rate),
                        })
                        .collect();
                    let vi_attrs: Vec<vk::VertexInputAttributeDescription> = ci
                        .vertex_input_state
                        .vertex_attribute_descriptions
                        .iter()
                        .map(|a| vk::VertexInputAttributeDescription {
                            location: a.location,
                            binding: a.binding,
                            format: vk::Format::from_raw(a.format),
                            offset: a.offset,
                        })
                        .collect();
                    all_vi_bindings.push(vi_bindings);
                    all_vi_attrs.push(vi_attrs);

                    // Viewport/scissor data
                    if let Some(ref vps) = ci.viewport_state {
                        let viewports: Vec<vk::Viewport> = vps
                            .viewports
                            .iter()
                            .map(|v| vk::Viewport {
                                x: v.x,
                                y: v.y,
                                width: v.width,
                                height: v.height,
                                min_depth: v.min_depth,
                                max_depth: v.max_depth,
                            })
                            .collect();
                        let scissors: Vec<vk::Rect2D> = vps
                            .scissors
                            .iter()
                            .map(|s| vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: s.offset[0],
                                    y: s.offset[1],
                                },
                                extent: vk::Extent2D {
                                    width: s.extent[0],
                                    height: s.extent[1],
                                },
                            })
                            .collect();
                        all_viewports.push(viewports);
                        all_scissors.push(scissors);
                    } else {
                        all_viewports.push(Vec::new());
                        all_scissors.push(Vec::new());
                    }

                    // Color blend attachments
                    if let Some(ref cbs) = ci.color_blend_state {
                        let blend_attachments: Vec<vk::PipelineColorBlendAttachmentState> = cbs
                            .attachments
                            .iter()
                            .map(|a| {
                                vk::PipelineColorBlendAttachmentState::default()
                                    .blend_enable(a.blend_enable)
                                    .src_color_blend_factor(vk::BlendFactor::from_raw(
                                        a.src_color_blend_factor,
                                    ))
                                    .dst_color_blend_factor(vk::BlendFactor::from_raw(
                                        a.dst_color_blend_factor,
                                    ))
                                    .color_blend_op(vk::BlendOp::from_raw(a.color_blend_op))
                                    .src_alpha_blend_factor(vk::BlendFactor::from_raw(
                                        a.src_alpha_blend_factor,
                                    ))
                                    .dst_alpha_blend_factor(vk::BlendFactor::from_raw(
                                        a.dst_alpha_blend_factor,
                                    ))
                                    .alpha_blend_op(vk::BlendOp::from_raw(a.alpha_blend_op))
                                    .color_write_mask(vk::ColorComponentFlags::from_raw(
                                        a.color_write_mask,
                                    ))
                            })
                            .collect();
                        all_cb_attachments.push(blend_attachments);
                    } else {
                        all_cb_attachments.push(Vec::new());
                    }

                    // Dynamic states
                    if let Some(ref dyn_s) = ci.dynamic_state {
                        let dyn_states: Vec<vk::DynamicState> = dyn_s
                            .dynamic_states
                            .iter()
                            .map(|d| vk::DynamicState::from_raw(*d))
                            .collect();
                        all_dyn_states_raw.push(dyn_states);
                    } else {
                        all_dyn_states_raw.push(Vec::new());
                    }
                }

                // Phase 2: Build all CreateInfo structs (now all Vecs are stable, no more pushes)
                for (i, ci) in create_infos.iter().enumerate() {
                    // Shader stages
                    let mut stages = Vec::new();
                    for (j, stage) in ci.stages.iter().enumerate() {
                        let module = match self.shader_module_handles.get(&stage.module) {
                            Some(m) => *m.value(),
                            None => {
                                return VulkanResponse::Error {
                                    code: ash::vk::Result::ERROR_INITIALIZATION_FAILED.as_raw(),
                                    message: format!("shader module handle {:?} not found", stage.module),
                                };
                            }
                        };
                        stages.push(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::from_raw(stage.stage))
                                .module(module)
                                .name(all_entry_points[i][j].as_c_str()),
                        );
                    }
                    all_stages.push(stages);

                    // Vertex input state
                    let vi_state = vk::PipelineVertexInputStateCreateInfo::default()
                        .vertex_binding_descriptions(&all_vi_bindings[i])
                        .vertex_attribute_descriptions(&all_vi_attrs[i]);
                    all_vi_states.push(vi_state);

                    // Input assembly
                    let ia_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                        .topology(vk::PrimitiveTopology::from_raw(
                            ci.input_assembly_state.topology,
                        ))
                        .primitive_restart_enable(ci.input_assembly_state.primitive_restart_enable);
                    all_ia_states.push(ia_state);

                    // Viewport state
                    if ci.viewport_state.is_some() {
                        let vp_state = vk::PipelineViewportStateCreateInfo::default()
                            .viewports(&all_viewports[i])
                            .scissors(&all_scissors[i]);
                        all_vp_states.push(vp_state);
                    } else {
                        all_vp_states.push(vk::PipelineViewportStateCreateInfo::default());
                    }

                    // Rasterization state
                    let rs = &ci.rasterization_state;
                    let rs_state = vk::PipelineRasterizationStateCreateInfo::default()
                        .depth_clamp_enable(rs.depth_clamp_enable)
                        .rasterizer_discard_enable(rs.rasterizer_discard_enable)
                        .polygon_mode(vk::PolygonMode::from_raw(rs.polygon_mode))
                        .cull_mode(vk::CullModeFlags::from_raw(rs.cull_mode))
                        .front_face(vk::FrontFace::from_raw(rs.front_face))
                        .depth_bias_enable(rs.depth_bias_enable)
                        .depth_bias_constant_factor(rs.depth_bias_constant_factor)
                        .depth_bias_clamp(rs.depth_bias_clamp)
                        .depth_bias_slope_factor(rs.depth_bias_slope_factor)
                        .line_width(rs.line_width);
                    all_rs_states.push(rs_state);

                    // Multisample state
                    if let Some(ref ms) = ci.multisample_state {
                        let ms_state = vk::PipelineMultisampleStateCreateInfo::default()
                            .rasterization_samples(vk::SampleCountFlags::from_raw(
                                ms.rasterization_samples,
                            ))
                            .sample_shading_enable(ms.sample_shading_enable)
                            .min_sample_shading(ms.min_sample_shading)
                            .alpha_to_coverage_enable(ms.alpha_to_coverage_enable)
                            .alpha_to_one_enable(ms.alpha_to_one_enable);
                        all_ms_states.push(ms_state);
                    } else {
                        all_ms_states.push(
                            vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        );
                    }

                    // Depth stencil state
                    if let Some(ref ds) = ci.depth_stencil_state {
                        let ds_state = vk::PipelineDepthStencilStateCreateInfo::default()
                            .depth_test_enable(ds.depth_test_enable)
                            .depth_write_enable(ds.depth_write_enable)
                            .depth_compare_op(vk::CompareOp::from_raw(ds.depth_compare_op))
                            .depth_bounds_test_enable(ds.depth_bounds_test_enable)
                            .stencil_test_enable(ds.stencil_test_enable)
                            .front(vk::StencilOpState {
                                fail_op: vk::StencilOp::from_raw(ds.front.fail_op),
                                pass_op: vk::StencilOp::from_raw(ds.front.pass_op),
                                depth_fail_op: vk::StencilOp::from_raw(ds.front.depth_fail_op),
                                compare_op: vk::CompareOp::from_raw(ds.front.compare_op),
                                compare_mask: ds.front.compare_mask,
                                write_mask: ds.front.write_mask,
                                reference: ds.front.reference,
                            })
                            .back(vk::StencilOpState {
                                fail_op: vk::StencilOp::from_raw(ds.back.fail_op),
                                pass_op: vk::StencilOp::from_raw(ds.back.pass_op),
                                depth_fail_op: vk::StencilOp::from_raw(ds.back.depth_fail_op),
                                compare_op: vk::CompareOp::from_raw(ds.back.compare_op),
                                compare_mask: ds.back.compare_mask,
                                write_mask: ds.back.write_mask,
                                reference: ds.back.reference,
                            })
                            .min_depth_bounds(ds.min_depth_bounds)
                            .max_depth_bounds(ds.max_depth_bounds);
                        all_ds_states.push(ds_state);
                    } else {
                        all_ds_states.push(vk::PipelineDepthStencilStateCreateInfo::default());
                    }

                    // Color blend state
                    if let Some(ref cbs) = ci.color_blend_state {
                        let cb_state = vk::PipelineColorBlendStateCreateInfo::default()
                            .logic_op_enable(cbs.logic_op_enable)
                            .logic_op(vk::LogicOp::from_raw(cbs.logic_op))
                            .attachments(&all_cb_attachments[i])
                            .blend_constants(cbs.blend_constants);
                        all_cb_states.push(cb_state);
                    } else {
                        all_cb_states.push(vk::PipelineColorBlendStateCreateInfo::default());
                    }

                    // Dynamic state
                    if ci.dynamic_state.is_some() && !all_dyn_states_raw[i].is_empty() {
                        let dyn_state = vk::PipelineDynamicStateCreateInfo::default()
                            .dynamic_states(&all_dyn_states_raw[i]);
                        all_dyn_states.push(dyn_state);
                    } else {
                        all_dyn_states.push(vk::PipelineDynamicStateCreateInfo::default());
                    }
                }

                // Phase 3: Assemble final pipeline create infos
                for (i, ci) in create_infos.iter().enumerate() {
                    let layout = match self.pipeline_layout_handles.get(&ci.layout) {
                        Some(l) => *l.value(),
                        None => {
                            return VulkanResponse::Error {
                                code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                                message: "invalid pipeline layout handle".to_string(),
                            }
                        }
                    };
                    let rp = match self.render_pass_handles.get(&ci.render_pass) {
                        Some(r) => *r.value(),
                        None => {
                            return VulkanResponse::Error {
                                code: vk::Result::ERROR_DEVICE_LOST.as_raw(),
                                message: "invalid render pass handle".to_string(),
                            }
                        }
                    };

                    let mut pipeline_ci = vk::GraphicsPipelineCreateInfo::default()
                        .flags(vk::PipelineCreateFlags::from_raw(ci.flags))
                        .stages(&all_stages[i])
                        .vertex_input_state(&all_vi_states[i])
                        .input_assembly_state(&all_ia_states[i])
                        .viewport_state(&all_vp_states[i])
                        .rasterization_state(&all_rs_states[i])
                        .multisample_state(&all_ms_states[i])
                        .depth_stencil_state(&all_ds_states[i])
                        .color_blend_state(&all_cb_states[i])
                        .layout(layout)
                        .render_pass(rp)
                        .subpass(ci.subpass);

                    if ci.dynamic_state.is_some()
                        && !all_dyn_states_raw[i].is_empty()
                    {
                        pipeline_ci = pipeline_ci.dynamic_state(&all_dyn_states[i]);
                    }

                    vk_create_infos.push(pipeline_ci);
                }

                match unsafe {
                    dev.create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &vk_create_infos,
                        None,
                    )
                } {
                    Ok(pipelines) => {
                        let mut handles = Vec::new();
                        for p in pipelines {
                            let handle = session.alloc_handle(ResourceType::VkPipeline);
                            self.pipeline_handles.insert(handle, p);
                            self.pipeline_to_device.insert(handle, device);
                            handles.push(handle);
                        }
                        debug!("created {} graphics pipeline(s)", handles.len());
                        VulkanResponse::PipelinesCreated { handles }
                    }
                    Err((pipelines, e)) => {
                        // Some pipelines may have succeeded
                        for p in pipelines {
                            if p != vk::Pipeline::null() {
                                unsafe { dev.destroy_pipeline(p, None) };
                            }
                        }
                        Self::vk_err(e)
                    }
                }
            }

            // ── Semaphore ──────────────────────────────────────────
            VulkanCommand::CreateSemaphore { device } => {
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

            VulkanCommand::DestroySemaphore { device, semaphore } => {
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
