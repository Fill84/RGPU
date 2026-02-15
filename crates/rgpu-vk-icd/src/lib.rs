//! RGPU Vulkan ICD (Installable Client Driver)
//!
//! This cdylib implements a Vulkan ICD that presents remote GPUs as local
//! physical devices. Applications using Vulkan will automatically see remote
//! GPUs when this ICD is registered with the Vulkan loader.

use std::ffi::{c_char, CStr};
use std::sync::OnceLock;

use rgpu_protocol::vulkan_commands::{VulkanCommand, VulkanResponse};

pub mod command;
pub mod descriptor;
pub mod device;
pub mod dispatch;
pub mod graphics_pipeline;
pub mod handle_store;
pub mod image;
pub mod instance;
pub mod ipc_client;
pub mod memory;
pub mod physical_device;
pub mod pipeline;
pub mod renderpass;
pub mod sync;

// ── IPC Client singleton ────────────────────────────────────

static IPC_CLIENT: OnceLock<ipc_client::IpcClient> = OnceLock::new();

fn get_ipc_client() -> &'static ipc_client::IpcClient {
    IPC_CLIENT.get_or_init(|| {
        let path = rgpu_common::platform::default_ipc_path();
        ipc_client::IpcClient::new(&path)
    })
}

/// Send a Vulkan command to the daemon via IPC.
pub fn send_vulkan_command(cmd: VulkanCommand) -> Result<VulkanResponse, String> {
    get_ipc_client().send_command(cmd)
}

// ── ICD Negotiation ─────────────────────────────────────────

/// Negotiate the ICD interface version with the Vulkan loader.
#[no_mangle]
pub unsafe extern "C" fn vk_icdNegotiateLoaderICDInterfaceVersion(
    supported_version: *mut u32,
) -> i32 {
    if supported_version.is_null() {
        return -3; // VK_ERROR_INITIALIZATION_FAILED
    }
    let requested = *supported_version;
    *supported_version = std::cmp::min(requested, 5);
    0 // VK_SUCCESS
}

/// Returns function pointers for Vulkan functions.
/// The Vulkan loader calls this to resolve all Vulkan entry points.
#[no_mangle]
pub unsafe extern "C" fn vk_icdGetInstanceProcAddr(
    _instance: usize,
    p_name: *const c_char,
) -> Option<unsafe extern "C" fn()> {
    if p_name.is_null() {
        return None;
    }

    let name = CStr::from_ptr(p_name).to_str().ok()?;

    match name {
        // ── ICD entry points ────────────────────────────────
        "vk_icdNegotiateLoaderICDInterfaceVersion" => {
            Some(std::mem::transmute(
                vk_icdNegotiateLoaderICDInterfaceVersion as *const (),
            ))
        }
        "vk_icdGetInstanceProcAddr" => {
            Some(std::mem::transmute(
                vk_icdGetInstanceProcAddr as *const (),
            ))
        }
        "vk_icdGetPhysicalDeviceProcAddr" => {
            Some(std::mem::transmute(
                vk_icdGetPhysicalDeviceProcAddr as *const (),
            ))
        }

        // ── Instance ────────────────────────────────────────
        "vkCreateInstance" => {
            Some(std::mem::transmute(
                instance::vkCreateInstance as *const (),
            ))
        }
        "vkDestroyInstance" => {
            Some(std::mem::transmute(
                instance::vkDestroyInstance as *const (),
            ))
        }

        // ── Enumeration ─────────────────────────────────────
        "vkEnumeratePhysicalDevices" => {
            Some(std::mem::transmute(
                instance::vkEnumeratePhysicalDevices as *const (),
            ))
        }
        "vkEnumerateInstanceExtensionProperties" => {
            Some(std::mem::transmute(
                instance::vkEnumerateInstanceExtensionProperties as *const (),
            ))
        }
        "vkEnumerateInstanceLayerProperties" => {
            Some(std::mem::transmute(
                instance::vkEnumerateInstanceLayerProperties as *const (),
            ))
        }
        "vkEnumerateDeviceExtensionProperties" => {
            Some(std::mem::transmute(
                instance::vkEnumerateDeviceExtensionProperties as *const (),
            ))
        }

        // ── Physical Device Properties ──────────────────────
        "vkGetPhysicalDeviceProperties" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceProperties as *const (),
            ))
        }
        "vkGetPhysicalDeviceProperties2" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceProperties2 as *const (),
            ))
        }
        "vkGetPhysicalDeviceProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceFeatures" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFeatures as *const (),
            ))
        }
        "vkGetPhysicalDeviceFeatures2" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFeatures2 as *const (),
            ))
        }
        "vkGetPhysicalDeviceFeatures2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFeatures2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceMemoryProperties" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceMemoryProperties as *const (),
            ))
        }
        "vkGetPhysicalDeviceMemoryProperties2" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceMemoryProperties2 as *const (),
            ))
        }
        "vkGetPhysicalDeviceMemoryProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceMemoryProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceQueueFamilyProperties" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceQueueFamilyProperties as *const (),
            ))
        }
        "vkGetPhysicalDeviceQueueFamilyProperties2" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceQueueFamilyProperties2 as *const (),
            ))
        }
        "vkGetPhysicalDeviceQueueFamilyProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceQueueFamilyProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceFormatProperties" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFormatProperties as *const (),
            ))
        }
        "vkGetPhysicalDeviceFormatProperties2" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFormatProperties2 as *const (),
            ))
        }
        "vkGetPhysicalDeviceFormatProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFormatProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceSparseImageFormatProperties" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceSparseImageFormatProperties as *const (),
            ))
        }
        "vkGetPhysicalDeviceSparseImageFormatProperties2" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceSparseImageFormatProperties2 as *const (),
            ))
        }
        "vkGetPhysicalDeviceSparseImageFormatProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceSparseImageFormatProperties2KHR as *const (),
            ))
        }

        // ── Logical Device ──────────────────────────────────
        "vkCreateDevice" => {
            Some(std::mem::transmute(
                device::vkCreateDevice as *const (),
            ))
        }
        "vkDestroyDevice" => {
            Some(std::mem::transmute(
                device::vkDestroyDevice as *const (),
            ))
        }
        "vkGetDeviceQueue" => {
            Some(std::mem::transmute(
                device::vkGetDeviceQueue as *const (),
            ))
        }
        "vkDeviceWaitIdle" => {
            Some(std::mem::transmute(
                device::vkDeviceWaitIdle as *const (),
            ))
        }

        // ── Memory ──────────────────────────────────────────
        "vkAllocateMemory" => {
            Some(std::mem::transmute(
                memory::vkAllocateMemory as *const (),
            ))
        }
        "vkFreeMemory" => {
            Some(std::mem::transmute(
                memory::vkFreeMemory as *const (),
            ))
        }
        "vkMapMemory" => {
            Some(std::mem::transmute(
                memory::vkMapMemory as *const (),
            ))
        }
        "vkUnmapMemory" => {
            Some(std::mem::transmute(
                memory::vkUnmapMemory as *const (),
            ))
        }
        "vkFlushMappedMemoryRanges" => {
            Some(std::mem::transmute(
                memory::vkFlushMappedMemoryRanges as *const (),
            ))
        }
        "vkInvalidateMappedMemoryRanges" => {
            Some(std::mem::transmute(
                memory::vkInvalidateMappedMemoryRanges as *const (),
            ))
        }

        // ── Buffer ──────────────────────────────────────────
        "vkCreateBuffer" => {
            Some(std::mem::transmute(
                memory::vkCreateBuffer as *const (),
            ))
        }
        "vkDestroyBuffer" => {
            Some(std::mem::transmute(
                memory::vkDestroyBuffer as *const (),
            ))
        }
        "vkBindBufferMemory" => {
            Some(std::mem::transmute(
                memory::vkBindBufferMemory as *const (),
            ))
        }
        "vkGetBufferMemoryRequirements" => {
            Some(std::mem::transmute(
                memory::vkGetBufferMemoryRequirements as *const (),
            ))
        }

        // ── Shader Module ───────────────────────────────────
        "vkCreateShaderModule" => {
            Some(std::mem::transmute(
                pipeline::vkCreateShaderModule as *const (),
            ))
        }
        "vkDestroyShaderModule" => {
            Some(std::mem::transmute(
                pipeline::vkDestroyShaderModule as *const (),
            ))
        }

        // ── Descriptor Set Layout ───────────────────────────
        "vkCreateDescriptorSetLayout" => {
            Some(std::mem::transmute(
                pipeline::vkCreateDescriptorSetLayout as *const (),
            ))
        }
        "vkDestroyDescriptorSetLayout" => {
            Some(std::mem::transmute(
                pipeline::vkDestroyDescriptorSetLayout as *const (),
            ))
        }

        // ── Pipeline Layout ─────────────────────────────────
        "vkCreatePipelineLayout" => {
            Some(std::mem::transmute(
                pipeline::vkCreatePipelineLayout as *const (),
            ))
        }
        "vkDestroyPipelineLayout" => {
            Some(std::mem::transmute(
                pipeline::vkDestroyPipelineLayout as *const (),
            ))
        }

        // ── Compute Pipeline ────────────────────────────────
        "vkCreateComputePipelines" => {
            Some(std::mem::transmute(
                pipeline::vkCreateComputePipelines as *const (),
            ))
        }
        "vkDestroyPipeline" => {
            Some(std::mem::transmute(
                pipeline::vkDestroyPipeline as *const (),
            ))
        }

        // ── Image ────────────────────────────────────────────
        "vkCreateImage" => {
            Some(std::mem::transmute(
                image::vkCreateImage as *const (),
            ))
        }
        "vkDestroyImage" => {
            Some(std::mem::transmute(
                image::vkDestroyImage as *const (),
            ))
        }
        "vkGetImageMemoryRequirements" => {
            Some(std::mem::transmute(
                image::vkGetImageMemoryRequirements as *const (),
            ))
        }
        "vkBindImageMemory" => {
            Some(std::mem::transmute(
                image::vkBindImageMemory as *const (),
            ))
        }

        // ── Image View ───────────────────────────────────────
        "vkCreateImageView" => {
            Some(std::mem::transmute(
                image::vkCreateImageView as *const (),
            ))
        }
        "vkDestroyImageView" => {
            Some(std::mem::transmute(
                image::vkDestroyImageView as *const (),
            ))
        }

        // ── Render Pass ──────────────────────────────────────
        "vkCreateRenderPass" => {
            Some(std::mem::transmute(
                renderpass::vkCreateRenderPass as *const (),
            ))
        }
        "vkDestroyRenderPass" => {
            Some(std::mem::transmute(
                renderpass::vkDestroyRenderPass as *const (),
            ))
        }

        // ── Framebuffer ──────────────────────────────────────
        "vkCreateFramebuffer" => {
            Some(std::mem::transmute(
                renderpass::vkCreateFramebuffer as *const (),
            ))
        }
        "vkDestroyFramebuffer" => {
            Some(std::mem::transmute(
                renderpass::vkDestroyFramebuffer as *const (),
            ))
        }

        // ── Graphics Pipeline ────────────────────────────────
        "vkCreateGraphicsPipelines" => {
            Some(std::mem::transmute(
                graphics_pipeline::vkCreateGraphicsPipelines as *const (),
            ))
        }

        // ── Semaphore ────────────────────────────────────────
        "vkCreateSemaphore" => {
            Some(std::mem::transmute(
                sync::vkCreateSemaphore as *const (),
            ))
        }
        "vkDestroySemaphore" => {
            Some(std::mem::transmute(
                sync::vkDestroySemaphore as *const (),
            ))
        }

        // ── Descriptor Pool ─────────────────────────────────
        "vkCreateDescriptorPool" => {
            Some(std::mem::transmute(
                descriptor::vkCreateDescriptorPool as *const (),
            ))
        }
        "vkDestroyDescriptorPool" => {
            Some(std::mem::transmute(
                descriptor::vkDestroyDescriptorPool as *const (),
            ))
        }

        // ── Descriptor Set ──────────────────────────────────
        "vkAllocateDescriptorSets" => {
            Some(std::mem::transmute(
                descriptor::vkAllocateDescriptorSets as *const (),
            ))
        }
        "vkFreeDescriptorSets" => {
            Some(std::mem::transmute(
                descriptor::vkFreeDescriptorSets as *const (),
            ))
        }
        "vkUpdateDescriptorSets" => {
            Some(std::mem::transmute(
                descriptor::vkUpdateDescriptorSets as *const (),
            ))
        }

        // ── Command Pool ────────────────────────────────────
        "vkCreateCommandPool" => {
            Some(std::mem::transmute(
                command::vkCreateCommandPool as *const (),
            ))
        }
        "vkDestroyCommandPool" => {
            Some(std::mem::transmute(
                command::vkDestroyCommandPool as *const (),
            ))
        }
        "vkResetCommandPool" => {
            Some(std::mem::transmute(
                command::vkResetCommandPool as *const (),
            ))
        }

        // ── Command Buffer ──────────────────────────────────
        "vkAllocateCommandBuffers" => {
            Some(std::mem::transmute(
                command::vkAllocateCommandBuffers as *const (),
            ))
        }
        "vkFreeCommandBuffers" => {
            Some(std::mem::transmute(
                command::vkFreeCommandBuffers as *const (),
            ))
        }
        "vkBeginCommandBuffer" => {
            Some(std::mem::transmute(
                command::vkBeginCommandBuffer as *const (),
            ))
        }
        "vkEndCommandBuffer" => {
            Some(std::mem::transmute(
                command::vkEndCommandBuffer as *const (),
            ))
        }
        "vkResetCommandBuffer" => {
            Some(std::mem::transmute(
                command::vkResetCommandBuffer as *const (),
            ))
        }

        // ── vkCmd* Recording ────────────────────────────────
        "vkCmdBindPipeline" => {
            Some(std::mem::transmute(
                command::vkCmdBindPipeline as *const (),
            ))
        }
        "vkCmdBindDescriptorSets" => {
            Some(std::mem::transmute(
                command::vkCmdBindDescriptorSets as *const (),
            ))
        }
        "vkCmdDispatch" => {
            Some(std::mem::transmute(
                command::vkCmdDispatch as *const (),
            ))
        }
        "vkCmdPipelineBarrier" => {
            Some(std::mem::transmute(
                command::vkCmdPipelineBarrier as *const (),
            ))
        }
        "vkCmdCopyBuffer" => {
            Some(std::mem::transmute(
                command::vkCmdCopyBuffer as *const (),
            ))
        }
        "vkCmdFillBuffer" => {
            Some(std::mem::transmute(
                command::vkCmdFillBuffer as *const (),
            ))
        }
        "vkCmdUpdateBuffer" => {
            Some(std::mem::transmute(
                command::vkCmdUpdateBuffer as *const (),
            ))
        }
        "vkCmdBeginRenderPass" => {
            Some(std::mem::transmute(
                command::vkCmdBeginRenderPass as *const (),
            ))
        }
        "vkCmdEndRenderPass" => {
            Some(std::mem::transmute(
                command::vkCmdEndRenderPass as *const (),
            ))
        }
        "vkCmdDraw" => {
            Some(std::mem::transmute(
                command::vkCmdDraw as *const (),
            ))
        }
        "vkCmdDrawIndexed" => {
            Some(std::mem::transmute(
                command::vkCmdDrawIndexed as *const (),
            ))
        }
        "vkCmdBindVertexBuffers" => {
            Some(std::mem::transmute(
                command::vkCmdBindVertexBuffers as *const (),
            ))
        }
        "vkCmdBindIndexBuffer" => {
            Some(std::mem::transmute(
                command::vkCmdBindIndexBuffer as *const (),
            ))
        }
        "vkCmdSetViewport" => {
            Some(std::mem::transmute(
                command::vkCmdSetViewport as *const (),
            ))
        }
        "vkCmdSetScissor" => {
            Some(std::mem::transmute(
                command::vkCmdSetScissor as *const (),
            ))
        }
        "vkCmdCopyBufferToImage" => {
            Some(std::mem::transmute(
                command::vkCmdCopyBufferToImage as *const (),
            ))
        }
        "vkCmdCopyImageToBuffer" => {
            Some(std::mem::transmute(
                command::vkCmdCopyImageToBuffer as *const (),
            ))
        }

        // ── Fence ───────────────────────────────────────────
        "vkCreateFence" => {
            Some(std::mem::transmute(
                sync::vkCreateFence as *const (),
            ))
        }
        "vkDestroyFence" => {
            Some(std::mem::transmute(
                sync::vkDestroyFence as *const (),
            ))
        }
        "vkWaitForFences" => {
            Some(std::mem::transmute(
                sync::vkWaitForFences as *const (),
            ))
        }
        "vkResetFences" => {
            Some(std::mem::transmute(
                sync::vkResetFences as *const (),
            ))
        }
        "vkGetFenceStatus" => {
            Some(std::mem::transmute(
                sync::vkGetFenceStatus as *const (),
            ))
        }

        // ── Queue ───────────────────────────────────────────
        "vkQueueSubmit" => {
            Some(std::mem::transmute(
                sync::vkQueueSubmit as *const (),
            ))
        }
        "vkQueueWaitIdle" => {
            Some(std::mem::transmute(
                sync::vkQueueWaitIdle as *const (),
            ))
        }

        // ── Not implemented (return None) ───────────────────
        _ => None,
    }
}

/// Returns function pointers for physical device extension functions.
#[no_mangle]
pub unsafe extern "C" fn vk_icdGetPhysicalDeviceProcAddr(
    _instance: usize,
    p_name: *const c_char,
) -> Option<unsafe extern "C" fn()> {
    if p_name.is_null() {
        return None;
    }

    let name = CStr::from_ptr(p_name).to_str().ok()?;

    match name {
        "vkGetPhysicalDeviceProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceFeatures2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFeatures2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceMemoryProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceMemoryProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceQueueFamilyProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceQueueFamilyProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceFormatProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceFormatProperties2KHR as *const (),
            ))
        }
        "vkGetPhysicalDeviceSparseImageFormatProperties2KHR" => {
            Some(std::mem::transmute(
                physical_device::vkGetPhysicalDeviceSparseImageFormatProperties2KHR as *const (),
            ))
        }
        _ => None,
    }
}
