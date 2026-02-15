use serde::{Deserialize, Serialize};

/// Comprehensive GPU capability information, shared between server and client.
#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct GpuInfo {
    /// Human-readable device name (e.g., "NVIDIA GeForce RTX 4090")
    pub device_name: String,
    /// PCI vendor ID
    pub vendor_id: u32,
    /// PCI device ID
    pub device_id: u32,
    /// Device type classification
    pub device_type: GpuDeviceType,
    /// Total device memory in bytes
    pub total_memory: u64,
    /// Whether this GPU supports Vulkan
    pub supports_vulkan: bool,
    /// Whether this GPU supports CUDA
    pub supports_cuda: bool,
    /// Vulkan API version (if supported), encoded as per Vulkan spec
    pub vulkan_api_version: Option<u32>,
    /// Vulkan driver version
    pub vulkan_driver_version: Option<u32>,
    /// CUDA compute capability (major, minor) if supported
    pub cuda_compute_capability: Option<(i32, i32)>,
    /// Number of Vulkan queue families
    pub queue_family_count: u32,
    /// Memory heap information
    pub memory_heaps: Vec<MemoryHeapInfo>,
    /// Server-side device index
    pub server_device_index: u32,
    /// Which server this GPU belongs to
    #[serde(default)]
    pub server_id: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum GpuDeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize,
         rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct MemoryHeapInfo {
    pub size: u64,
    pub is_device_local: bool,
}
