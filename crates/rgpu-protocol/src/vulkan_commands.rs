use serde::{Deserialize, Serialize};

use crate::handle::NetworkHandle;

// ============================================================================
// Serialized Vulkan types for network transport
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct DeviceQueueCreateInfo {
    pub queue_family_index: u32,
    pub queue_priorities: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MappedMemoryRange {
    pub memory: NetworkHandle,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: i32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPushConstantRange {
    pub stage_flags: u32,
    pub offset: u32,
    pub size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedComputePipelineCreateInfo {
    pub stage: SerializedPipelineShaderStageCreateInfo,
    pub layout: NetworkHandle,
    pub flags: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineShaderStageCreateInfo {
    pub module: NetworkHandle,
    pub entry_point: String,
    pub stage: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedDescriptorPoolSize {
    pub descriptor_type: i32,
    pub descriptor_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedWriteDescriptorSet {
    pub dst_set: NetworkHandle,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_type: i32,
    pub buffer_infos: Vec<SerializedDescriptorBufferInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedDescriptorBufferInfo {
    pub buffer: NetworkHandle,
    pub offset: u64,
    pub range: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedSubmitInfo {
    pub wait_semaphores: Vec<NetworkHandle>,
    pub wait_dst_stage_masks: Vec<u32>,
    pub command_buffers: Vec<NetworkHandle>,
    pub signal_semaphores: Vec<NetworkHandle>,
}

/// Recorded command buffer commands, batched client-side and sent at submit time.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum RecordedCommand {
    BindPipeline {
        pipeline_bind_point: u32,
        pipeline: NetworkHandle,
    },
    BindDescriptorSets {
        pipeline_bind_point: u32,
        layout: NetworkHandle,
        first_set: u32,
        descriptor_sets: Vec<NetworkHandle>,
        dynamic_offsets: Vec<u32>,
    },
    Dispatch {
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
    PipelineBarrier {
        src_stage_mask: u32,
        dst_stage_mask: u32,
        dependency_flags: u32,
        memory_barriers: Vec<SerializedMemoryBarrier>,
        buffer_memory_barriers: Vec<SerializedBufferMemoryBarrier>,
        image_memory_barriers: Vec<SerializedImageMemoryBarrier>,
    },
    CopyBuffer {
        src: NetworkHandle,
        dst: NetworkHandle,
        regions: Vec<SerializedBufferCopy>,
    },
    FillBuffer {
        buffer: NetworkHandle,
        offset: u64,
        size: u64,
        data: u32,
    },
    UpdateBuffer {
        buffer: NetworkHandle,
        offset: u64,
        data: Vec<u8>,
    },

    // ── Phase 5: Rendering commands ─────────────────────────
    BeginRenderPass {
        render_pass: NetworkHandle,
        framebuffer: NetworkHandle,
        render_area: SerializedRect2D,
        clear_values: Vec<SerializedClearValue>,
        contents: u32,
    },
    EndRenderPass,
    Draw {
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    BindVertexBuffers {
        first_binding: u32,
        buffers: Vec<NetworkHandle>,
        offsets: Vec<u64>,
    },
    BindIndexBuffer {
        buffer: NetworkHandle,
        offset: u64,
        index_type: u32,
    },
    SetViewport {
        first_viewport: u32,
        viewports: Vec<SerializedViewport>,
    },
    SetScissor {
        first_scissor: u32,
        scissors: Vec<SerializedRect2D>,
    },
    CopyBufferToImage {
        src_buffer: NetworkHandle,
        dst_image: NetworkHandle,
        dst_image_layout: i32,
        regions: Vec<SerializedBufferImageCopy>,
    },
    CopyImageToBuffer {
        src_image: NetworkHandle,
        src_image_layout: i32,
        dst_buffer: NetworkHandle,
        regions: Vec<SerializedBufferImageCopy>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedMemoryBarrier {
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedBufferMemoryBarrier {
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: NetworkHandle,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedBufferCopy {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

// ============================================================================
// Phase 5: Image, RenderPass, Graphics Pipeline serialization types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedImageCreateInfo {
    pub flags: u32,
    pub image_type: i32,
    pub format: i32,
    pub extent: [u32; 3],
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: u32,
    pub tiling: i32,
    pub usage: u32,
    pub sharing_mode: i32,
    pub queue_family_indices: Vec<u32>,
    pub initial_layout: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedComponentMapping {
    pub r: i32,
    pub g: i32,
    pub b: i32,
    pub a: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedImageSubresourceRange {
    pub aspect_mask: u32,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedAttachmentDescription {
    pub flags: u32,
    pub format: i32,
    pub samples: u32,
    pub load_op: i32,
    pub store_op: i32,
    pub stencil_load_op: i32,
    pub stencil_store_op: i32,
    pub initial_layout: i32,
    pub final_layout: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedAttachmentReference {
    pub attachment: u32,
    pub layout: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedSubpassDescription {
    pub flags: u32,
    pub pipeline_bind_point: i32,
    pub input_attachments: Vec<SerializedAttachmentReference>,
    pub color_attachments: Vec<SerializedAttachmentReference>,
    pub resolve_attachments: Vec<SerializedAttachmentReference>,
    pub depth_stencil_attachment: Option<SerializedAttachmentReference>,
    pub preserve_attachments: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedSubpassDependency {
    pub src_subpass: u32,
    pub dst_subpass: u32,
    pub src_stage_mask: u32,
    pub dst_stage_mask: u32,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub dependency_flags: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedVertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedVertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: i32,
    pub offset: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineVertexInputStateCreateInfo {
    pub vertex_binding_descriptions: Vec<SerializedVertexInputBindingDescription>,
    pub vertex_attribute_descriptions: Vec<SerializedVertexInputAttributeDescription>,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineInputAssemblyStateCreateInfo {
    pub topology: i32,
    pub primitive_restart_enable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedRect2D {
    pub offset: [i32; 2],
    pub extent: [u32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineViewportStateCreateInfo {
    pub viewports: Vec<SerializedViewport>,
    pub scissors: Vec<SerializedRect2D>,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineRasterizationStateCreateInfo {
    pub depth_clamp_enable: bool,
    pub rasterizer_discard_enable: bool,
    pub polygon_mode: i32,
    pub cull_mode: u32,
    pub front_face: i32,
    pub depth_bias_enable: bool,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineMultisampleStateCreateInfo {
    pub rasterization_samples: u32,
    pub sample_shading_enable: bool,
    pub min_sample_shading: f32,
    pub alpha_to_coverage_enable: bool,
    pub alpha_to_one_enable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedStencilOpState {
    pub fail_op: i32,
    pub pass_op: i32,
    pub depth_fail_op: i32,
    pub compare_op: i32,
    pub compare_mask: u32,
    pub write_mask: u32,
    pub reference: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineDepthStencilStateCreateInfo {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: i32,
    pub depth_bounds_test_enable: bool,
    pub stencil_test_enable: bool,
    pub front: SerializedStencilOpState,
    pub back: SerializedStencilOpState,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineColorBlendAttachmentState {
    pub blend_enable: bool,
    pub src_color_blend_factor: i32,
    pub dst_color_blend_factor: i32,
    pub color_blend_op: i32,
    pub src_alpha_blend_factor: i32,
    pub dst_alpha_blend_factor: i32,
    pub alpha_blend_op: i32,
    pub color_write_mask: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineColorBlendStateCreateInfo {
    pub logic_op_enable: bool,
    pub logic_op: i32,
    pub attachments: Vec<SerializedPipelineColorBlendAttachmentState>,
    pub blend_constants: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedPipelineDynamicStateCreateInfo {
    pub dynamic_states: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedGraphicsPipelineCreateInfo {
    pub flags: u32,
    pub stages: Vec<SerializedPipelineShaderStageCreateInfo>,
    pub vertex_input_state: SerializedPipelineVertexInputStateCreateInfo,
    pub input_assembly_state: SerializedPipelineInputAssemblyStateCreateInfo,
    pub viewport_state: Option<SerializedPipelineViewportStateCreateInfo>,
    pub rasterization_state: SerializedPipelineRasterizationStateCreateInfo,
    pub multisample_state: Option<SerializedPipelineMultisampleStateCreateInfo>,
    pub depth_stencil_state: Option<SerializedPipelineDepthStencilStateCreateInfo>,
    pub color_blend_state: Option<SerializedPipelineColorBlendStateCreateInfo>,
    pub dynamic_state: Option<SerializedPipelineDynamicStateCreateInfo>,
    pub layout: NetworkHandle,
    pub render_pass: NetworkHandle,
    pub subpass: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedImageMemoryBarrier {
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub old_layout: i32,
    pub new_layout: i32,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: NetworkHandle,
    pub subresource_range: SerializedImageSubresourceRange,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedClearValue {
    pub data: [u8; 16],
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedImageSubresourceLayers {
    pub aspect_mask: u32,
    pub mip_level: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedBufferImageCopy {
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: SerializedImageSubresourceLayers,
    pub image_offset: [i32; 3],
    pub image_extent: [u32; 3],
}

// ============================================================================
// Response serialized types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedMemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedMemoryHeap {
    pub size: u64,
    pub flags: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedQueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: [u32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedExtensionProperties {
    pub extension_name: String,
    pub spec_version: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct SerializedLayerProperties {
    pub layer_name: String,
    pub spec_version: u32,
    pub implementation_version: u32,
    pub description: String,
}

// ============================================================================
// Vulkan Commands (client → server)
// ============================================================================

/// Vulkan API commands sent from client to server.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum VulkanCommand {
    // ── Instance ────────────────────────────────────────────
    CreateInstance {
        app_name: Option<String>,
        app_version: u32,
        engine_name: Option<String>,
        engine_version: u32,
        api_version: u32,
        enabled_extensions: Vec<String>,
        enabled_layers: Vec<String>,
    },
    DestroyInstance {
        instance: NetworkHandle,
    },

    // ── Enumeration ─────────────────────────────────────────
    EnumeratePhysicalDevices {
        instance: NetworkHandle,
    },
    EnumerateInstanceExtensionProperties {
        layer_name: Option<String>,
    },
    EnumerateInstanceLayerProperties,
    EnumerateDeviceExtensionProperties {
        physical_device: NetworkHandle,
        layer_name: Option<String>,
    },

    // ── Physical Device Queries ─────────────────────────────
    GetPhysicalDeviceProperties {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceProperties2 {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceFeatures {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceFeatures2 {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceMemoryProperties {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceMemoryProperties2 {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceQueueFamilyProperties {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceQueueFamilyProperties2 {
        physical_device: NetworkHandle,
    },
    GetPhysicalDeviceFormatProperties {
        physical_device: NetworkHandle,
        format: i32,
    },

    // ── Logical Device ──────────────────────────────────────
    CreateDevice {
        physical_device: NetworkHandle,
        queue_create_infos: Vec<DeviceQueueCreateInfo>,
        enabled_extensions: Vec<String>,
        /// Physical device features serialized as raw bytes (VkPhysicalDeviceFeatures)
        enabled_features: Option<Vec<u8>>,
    },
    DestroyDevice {
        device: NetworkHandle,
    },
    DeviceWaitIdle {
        device: NetworkHandle,
    },

    // ── Queue ───────────────────────────────────────────────
    GetDeviceQueue {
        device: NetworkHandle,
        queue_family_index: u32,
        queue_index: u32,
    },
    QueueSubmit {
        queue: NetworkHandle,
        submits: Vec<SerializedSubmitInfo>,
        fence: Option<NetworkHandle>,
    },
    QueueWaitIdle {
        queue: NetworkHandle,
    },

    // ── Memory ──────────────────────────────────────────────
    AllocateMemory {
        device: NetworkHandle,
        alloc_size: u64,
        memory_type_index: u32,
    },
    FreeMemory {
        device: NetworkHandle,
        memory: NetworkHandle,
    },
    MapMemory {
        device: NetworkHandle,
        memory: NetworkHandle,
        offset: u64,
        size: u64,
        flags: u32,
    },
    UnmapMemory {
        device: NetworkHandle,
        memory: NetworkHandle,
        /// Data written by the client to the mapped region
        written_data: Option<Vec<u8>>,
        offset: u64,
    },
    FlushMappedMemoryRanges {
        device: NetworkHandle,
        ranges: Vec<MappedMemoryRange>,
        /// Data for flushed ranges
        data: Vec<Vec<u8>>,
    },
    InvalidateMappedMemoryRanges {
        device: NetworkHandle,
        ranges: Vec<MappedMemoryRange>,
    },

    // ── Buffer ──────────────────────────────────────────────
    CreateBuffer {
        device: NetworkHandle,
        size: u64,
        usage: u32,
        sharing_mode: u32,
        queue_family_indices: Vec<u32>,
    },
    DestroyBuffer {
        device: NetworkHandle,
        buffer: NetworkHandle,
    },
    BindBufferMemory {
        device: NetworkHandle,
        buffer: NetworkHandle,
        memory: NetworkHandle,
        memory_offset: u64,
    },
    GetBufferMemoryRequirements {
        device: NetworkHandle,
        buffer: NetworkHandle,
    },

    // ── Shader Module ───────────────────────────────────────
    CreateShaderModule {
        device: NetworkHandle,
        code: Vec<u8>,
    },
    DestroyShaderModule {
        device: NetworkHandle,
        shader_module: NetworkHandle,
    },

    // ── Descriptor Set Layout ───────────────────────────────
    CreateDescriptorSetLayout {
        device: NetworkHandle,
        bindings: Vec<SerializedDescriptorSetLayoutBinding>,
    },
    DestroyDescriptorSetLayout {
        device: NetworkHandle,
        layout: NetworkHandle,
    },

    // ── Pipeline Layout ─────────────────────────────────────
    CreatePipelineLayout {
        device: NetworkHandle,
        set_layouts: Vec<NetworkHandle>,
        push_constant_ranges: Vec<SerializedPushConstantRange>,
    },
    DestroyPipelineLayout {
        device: NetworkHandle,
        layout: NetworkHandle,
    },

    // ── Compute Pipeline ────────────────────────────────────
    CreateComputePipelines {
        device: NetworkHandle,
        create_infos: Vec<SerializedComputePipelineCreateInfo>,
    },
    DestroyPipeline {
        device: NetworkHandle,
        pipeline: NetworkHandle,
    },

    // ── Descriptor Pool ─────────────────────────────────────
    CreateDescriptorPool {
        device: NetworkHandle,
        max_sets: u32,
        pool_sizes: Vec<SerializedDescriptorPoolSize>,
        flags: u32,
    },
    DestroyDescriptorPool {
        device: NetworkHandle,
        pool: NetworkHandle,
    },

    // ── Descriptor Set ──────────────────────────────────────
    AllocateDescriptorSets {
        device: NetworkHandle,
        descriptor_pool: NetworkHandle,
        set_layouts: Vec<NetworkHandle>,
    },
    FreeDescriptorSets {
        device: NetworkHandle,
        descriptor_pool: NetworkHandle,
        descriptor_sets: Vec<NetworkHandle>,
    },
    UpdateDescriptorSets {
        device: NetworkHandle,
        writes: Vec<SerializedWriteDescriptorSet>,
    },

    // ── Command Pool ────────────────────────────────────────
    CreateCommandPool {
        device: NetworkHandle,
        queue_family_index: u32,
        flags: u32,
    },
    DestroyCommandPool {
        device: NetworkHandle,
        command_pool: NetworkHandle,
    },
    ResetCommandPool {
        device: NetworkHandle,
        command_pool: NetworkHandle,
        flags: u32,
    },

    // ── Command Buffer ──────────────────────────────────────
    AllocateCommandBuffers {
        device: NetworkHandle,
        command_pool: NetworkHandle,
        level: u32,
        count: u32,
    },
    FreeCommandBuffers {
        device: NetworkHandle,
        command_pool: NetworkHandle,
        command_buffers: Vec<NetworkHandle>,
    },

    // ── Command Buffer Recording (Batched) ──────────────────
    SubmitRecordedCommands {
        command_buffer: NetworkHandle,
        commands: Vec<RecordedCommand>,
    },

    // ── Fence ───────────────────────────────────────────────
    CreateFence {
        device: NetworkHandle,
        signaled: bool,
    },
    DestroyFence {
        device: NetworkHandle,
        fence: NetworkHandle,
    },
    WaitForFences {
        device: NetworkHandle,
        fences: Vec<NetworkHandle>,
        wait_all: bool,
        timeout_ns: u64,
    },
    ResetFences {
        device: NetworkHandle,
        fences: Vec<NetworkHandle>,
    },
    GetFenceStatus {
        device: NetworkHandle,
        fence: NetworkHandle,
    },

    // ── Image ──────────────────────────────────────────────
    CreateImage {
        device: NetworkHandle,
        create_info: SerializedImageCreateInfo,
    },
    DestroyImage {
        device: NetworkHandle,
        image: NetworkHandle,
    },
    GetImageMemoryRequirements {
        device: NetworkHandle,
        image: NetworkHandle,
    },
    BindImageMemory {
        device: NetworkHandle,
        image: NetworkHandle,
        memory: NetworkHandle,
        memory_offset: u64,
    },

    // ── Image View ─────────────────────────────────────────
    CreateImageView {
        device: NetworkHandle,
        image: NetworkHandle,
        view_type: i32,
        format: i32,
        components: SerializedComponentMapping,
        subresource_range: SerializedImageSubresourceRange,
    },
    DestroyImageView {
        device: NetworkHandle,
        image_view: NetworkHandle,
    },

    // ── Render Pass ────────────────────────────────────────
    CreateRenderPass {
        device: NetworkHandle,
        attachments: Vec<SerializedAttachmentDescription>,
        subpasses: Vec<SerializedSubpassDescription>,
        dependencies: Vec<SerializedSubpassDependency>,
    },
    DestroyRenderPass {
        device: NetworkHandle,
        render_pass: NetworkHandle,
    },

    // ── Framebuffer ────────────────────────────────────────
    CreateFramebuffer {
        device: NetworkHandle,
        render_pass: NetworkHandle,
        attachments: Vec<NetworkHandle>,
        width: u32,
        height: u32,
        layers: u32,
    },
    DestroyFramebuffer {
        device: NetworkHandle,
        framebuffer: NetworkHandle,
    },

    // ── Graphics Pipeline ──────────────────────────────────
    CreateGraphicsPipelines {
        device: NetworkHandle,
        create_infos: Vec<SerializedGraphicsPipelineCreateInfo>,
    },

    // ── Semaphore ──────────────────────────────────────────
    CreateSemaphore {
        device: NetworkHandle,
    },
    DestroySemaphore {
        device: NetworkHandle,
        semaphore: NetworkHandle,
    },
}

// ============================================================================
// Vulkan Responses (server → client)
// ============================================================================

/// Vulkan API responses sent from server to client.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum VulkanResponse {
    Success,
    Error { code: i32, message: String },

    // ── Instance ────────────────────────────────────────────
    InstanceCreated { handle: NetworkHandle },

    // ── Physical Device ─────────────────────────────────────
    PhysicalDevices { handles: Vec<NetworkHandle> },
    PhysicalDeviceProperties {
        api_version: u32,
        driver_version: u32,
        vendor_id: u32,
        device_id: u32,
        device_type: u32,
        device_name: String,
        pipeline_cache_uuid: [u8; 16],
        /// VkPhysicalDeviceLimits serialized as raw bytes
        limits_raw: Vec<u8>,
        /// VkPhysicalDeviceSparseProperties serialized as raw bytes
        sparse_properties_raw: Vec<u8>,
    },
    PhysicalDeviceFeatures {
        /// VkPhysicalDeviceFeatures serialized as raw bytes
        features_raw: Vec<u8>,
    },
    PhysicalDeviceMemoryProperties {
        memory_type_count: u32,
        memory_types: Vec<SerializedMemoryType>,
        memory_heap_count: u32,
        memory_heaps: Vec<SerializedMemoryHeap>,
    },
    QueueFamilyProperties {
        families: Vec<SerializedQueueFamilyProperties>,
    },
    FormatProperties {
        linear_tiling_features: u32,
        optimal_tiling_features: u32,
        buffer_features: u32,
    },

    // ── Enumeration ─────────────────────────────────────────
    ExtensionProperties {
        extensions: Vec<SerializedExtensionProperties>,
    },
    LayerProperties {
        layers: Vec<SerializedLayerProperties>,
    },

    // ── Device ──────────────────────────────────────────────
    DeviceCreated { handle: NetworkHandle },
    QueueRetrieved { handle: NetworkHandle },

    // ── Memory ──────────────────────────────────────────────
    MemoryAllocated { handle: NetworkHandle },
    MemoryMapped { data: Vec<u8> },
    InvalidatedData { range_data: Vec<Vec<u8>> },

    // ── Buffer ──────────────────────────────────────────────
    BufferCreated { handle: NetworkHandle },
    MemoryRequirements {
        size: u64,
        alignment: u64,
        memory_type_bits: u32,
    },

    // ── Shader/Pipeline ─────────────────────────────────────
    ShaderModuleCreated { handle: NetworkHandle },
    DescriptorSetLayoutCreated { handle: NetworkHandle },
    PipelineLayoutCreated { handle: NetworkHandle },
    PipelinesCreated { handles: Vec<NetworkHandle> },
    DescriptorPoolCreated { handle: NetworkHandle },
    DescriptorSetsAllocated { handles: Vec<NetworkHandle> },

    // ── Command Pool/Buffer ─────────────────────────────────
    CommandPoolCreated { handle: NetworkHandle },
    CommandBuffersAllocated { handles: Vec<NetworkHandle> },

    // ── Fence ───────────────────────────────────────────────
    FenceCreated { handle: NetworkHandle },
    FenceStatus { signaled: bool },
    FenceWaitResult { result: i32 },

    // ── Image ───────────────────────────────────────────────
    ImageCreated { handle: NetworkHandle },
    ImageViewCreated { handle: NetworkHandle },

    // ── Render Pass / Framebuffer ───────────────────────────
    RenderPassCreated { handle: NetworkHandle },
    FramebufferCreated { handle: NetworkHandle },

    // ── Semaphore ───────────────────────────────────────────
    SemaphoreCreated { handle: NetworkHandle },
}
