//! Render pass and framebuffer functions for the Vulkan ICD.

use ash::vk;
use ash::vk::Handle;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{
    SerializedAttachmentDescription, SerializedAttachmentReference, SerializedSubpassDependency,
    SerializedSubpassDescription, VulkanCommand, VulkanResponse,
};

// ── vkCreateRenderPass ───────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateRenderPass(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_render_pass: *mut vk::RenderPass,
) -> vk::Result {
    if p_create_info.is_null() || p_render_pass.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;

    // Serialize attachments
    let attachments: Vec<SerializedAttachmentDescription> = if !ci.p_attachments.is_null()
        && ci.attachment_count > 0
    {
        std::slice::from_raw_parts(ci.p_attachments, ci.attachment_count as usize)
            .iter()
            .map(|a| SerializedAttachmentDescription {
                flags: a.flags.as_raw(),
                format: a.format.as_raw(),
                samples: a.samples.as_raw(),
                load_op: a.load_op.as_raw(),
                store_op: a.store_op.as_raw(),
                stencil_load_op: a.stencil_load_op.as_raw(),
                stencil_store_op: a.stencil_store_op.as_raw(),
                initial_layout: a.initial_layout.as_raw(),
                final_layout: a.final_layout.as_raw(),
            })
            .collect()
    } else {
        Vec::new()
    };

    // Serialize subpasses
    let subpasses: Vec<SerializedSubpassDescription> = if !ci.p_subpasses.is_null()
        && ci.subpass_count > 0
    {
        std::slice::from_raw_parts(ci.p_subpasses, ci.subpass_count as usize)
            .iter()
            .map(|sp| {
                let input_attachments = if !sp.p_input_attachments.is_null()
                    && sp.input_attachment_count > 0
                {
                    std::slice::from_raw_parts(
                        sp.p_input_attachments,
                        sp.input_attachment_count as usize,
                    )
                    .iter()
                    .map(|r| SerializedAttachmentReference {
                        attachment: r.attachment,
                        layout: r.layout.as_raw(),
                    })
                    .collect()
                } else {
                    Vec::new()
                };

                let color_attachments = if !sp.p_color_attachments.is_null()
                    && sp.color_attachment_count > 0
                {
                    std::slice::from_raw_parts(
                        sp.p_color_attachments,
                        sp.color_attachment_count as usize,
                    )
                    .iter()
                    .map(|r| SerializedAttachmentReference {
                        attachment: r.attachment,
                        layout: r.layout.as_raw(),
                    })
                    .collect()
                } else {
                    Vec::new()
                };

                let resolve_attachments = if !sp.p_resolve_attachments.is_null()
                    && sp.color_attachment_count > 0
                {
                    std::slice::from_raw_parts(
                        sp.p_resolve_attachments,
                        sp.color_attachment_count as usize,
                    )
                    .iter()
                    .map(|r| SerializedAttachmentReference {
                        attachment: r.attachment,
                        layout: r.layout.as_raw(),
                    })
                    .collect()
                } else {
                    Vec::new()
                };

                let depth_stencil_attachment = if !sp.p_depth_stencil_attachment.is_null() {
                    let ds = &*sp.p_depth_stencil_attachment;
                    Some(SerializedAttachmentReference {
                        attachment: ds.attachment,
                        layout: ds.layout.as_raw(),
                    })
                } else {
                    None
                };

                let preserve_attachments = if !sp.p_preserve_attachments.is_null()
                    && sp.preserve_attachment_count > 0
                {
                    std::slice::from_raw_parts(
                        sp.p_preserve_attachments,
                        sp.preserve_attachment_count as usize,
                    )
                    .to_vec()
                } else {
                    Vec::new()
                };

                SerializedSubpassDescription {
                    flags: sp.flags.as_raw(),
                    pipeline_bind_point: sp.pipeline_bind_point.as_raw(),
                    input_attachments,
                    color_attachments,
                    resolve_attachments,
                    depth_stencil_attachment,
                    preserve_attachments,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Serialize dependencies
    let dependencies: Vec<SerializedSubpassDependency> = if !ci.p_dependencies.is_null()
        && ci.dependency_count > 0
    {
        std::slice::from_raw_parts(ci.p_dependencies, ci.dependency_count as usize)
            .iter()
            .map(|d| SerializedSubpassDependency {
                src_subpass: d.src_subpass,
                dst_subpass: d.dst_subpass,
                src_stage_mask: d.src_stage_mask.as_raw(),
                dst_stage_mask: d.dst_stage_mask.as_raw(),
                src_access_mask: d.src_access_mask.as_raw(),
                dst_access_mask: d.dst_access_mask.as_raw(),
                dependency_flags: d.dependency_flags.as_raw(),
            })
            .collect()
    } else {
        Vec::new()
    };

    let cmd = VulkanCommand::CreateRenderPass {
        device: dev_handle,
        attachments,
        subpasses,
        dependencies,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::RenderPassCreated { handle }) => {
            let local_id = handle_store::store_render_pass(handle);
            *p_render_pass = vk::RenderPass::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

// ── vkDestroyRenderPass ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkDestroyRenderPass(
    device: vk::Device,
    render_pass: vk::RenderPass,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if render_pass == vk::RenderPass::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = render_pass.as_raw();
    if let Some(handle) = handle_store::remove_render_pass(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyRenderPass {
            device: dev_handle,
            render_pass: handle,
        });
    }
}

// ── vkCreateFramebuffer ──────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateFramebuffer(
    device: vk::Device,
    p_create_info: *const vk::FramebufferCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_framebuffer: *mut vk::Framebuffer,
) -> vk::Result {
    if p_create_info.is_null() || p_framebuffer.is_null() {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let ci = &*p_create_info;

    let rp_handle = match handle_store::get_render_pass(ci.render_pass.as_raw()) {
        Some(h) => h,
        None => return vk::Result::ERROR_UNKNOWN,
    };

    let mut attachment_handles = Vec::new();
    if !ci.p_attachments.is_null() && ci.attachment_count > 0 {
        for i in 0..ci.attachment_count as usize {
            let iv = *ci.p_attachments.add(i);
            match handle_store::get_image_view(iv.as_raw()) {
                Some(h) => attachment_handles.push(h),
                None => return vk::Result::ERROR_UNKNOWN,
            }
        }
    }

    let cmd = VulkanCommand::CreateFramebuffer {
        device: dev_handle,
        render_pass: rp_handle,
        attachments: attachment_handles,
        width: ci.width,
        height: ci.height,
        layers: ci.layers,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::FramebufferCreated { handle }) => {
            let local_id = handle_store::store_framebuffer(handle);
            *p_framebuffer = vk::Framebuffer::from_raw(local_id);
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}

// ── vkDestroyFramebuffer ─────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkDestroyFramebuffer(
    device: vk::Device,
    framebuffer: vk::Framebuffer,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if framebuffer == vk::Framebuffer::null() {
        return;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return,
    };

    let local_id = framebuffer.as_raw();
    if let Some(handle) = handle_store::remove_framebuffer(local_id) {
        let _ = send_vulkan_command(VulkanCommand::DestroyFramebuffer {
            device: dev_handle,
            framebuffer: handle,
        });
    }
}
