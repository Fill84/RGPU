use serde::{Deserialize, Serialize};

/// A network-safe handle that uniquely identifies a GPU resource.
/// Opaque to the client -- the server assigns these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct NetworkHandle {
    /// Which server owns this resource
    pub server_id: u16,
    /// Which client session created it
    pub session_id: u32,
    /// Unique resource identifier within the session
    pub resource_id: u64,
    /// Type tag for debugging and validation
    pub resource_type: ResourceType,
}

impl NetworkHandle {
    /// Create a null/invalid handle.
    pub fn null() -> Self {
        Self {
            server_id: 0,
            session_id: 0,
            resource_id: 0,
            resource_type: ResourceType::None,
        }
    }

    /// Create a handle representing the CUDA default (null) stream.
    pub fn null_stream() -> Self {
        Self {
            server_id: 0,
            session_id: 0,
            resource_id: 0,
            resource_type: ResourceType::CuStream,
        }
    }

    pub fn is_null(&self) -> bool {
        self.resource_type == ResourceType::None && self.resource_id == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ResourceType {
    None,

    // Vulkan resources
    VkInstance,
    VkPhysicalDevice,
    VkDevice,
    VkQueue,
    VkCommandPool,
    VkCommandBuffer,
    VkDeviceMemory,
    VkBuffer,
    VkImage,
    VkImageView,
    VkSampler,
    VkPipeline,
    VkPipelineLayout,
    VkDescriptorSetLayout,
    VkDescriptorPool,
    VkDescriptorSet,
    VkShaderModule,
    VkRenderPass,
    VkFramebuffer,
    VkFence,
    VkSemaphore,
    VkEvent,
    VkSwapchain,

    // CUDA resources
    CuDevice,
    CuContext,
    CuModule,
    CuFunction,
    CuDevicePtr,
    CuStream,
    CuEvent,
    CuHostPtr,
    CuMemPool,
    CuLinker,
}
