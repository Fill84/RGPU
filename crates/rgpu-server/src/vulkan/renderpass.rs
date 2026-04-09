use ash::vk;
use tracing::debug;

use rgpu_protocol::handle::ResourceType;
use rgpu_protocol::vulkan_commands::*;

use crate::session::Session;
use super::VulkanExecutor;

impl VulkanExecutor {
    pub(crate) fn handle_create_render_pass(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, attachments, subpasses, dependencies) = match cmd {
            VulkanCommand::CreateRenderPass {
                device,
                attachments,
                subpasses,
                dependencies,
            } => (device, attachments, subpasses, dependencies),
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

    pub(crate) fn handle_destroy_render_pass(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, render_pass) = match cmd {
            VulkanCommand::DestroyRenderPass { device, render_pass } => (device, render_pass),
            _ => unreachable!(),
        };

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

    pub(crate) fn handle_create_framebuffer(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, render_pass, attachment_handles, width, height, layers) = match cmd {
            VulkanCommand::CreateFramebuffer {
                device,
                render_pass,
                attachments: attachment_handles,
                width,
                height,
                layers,
            } => (device, render_pass, attachment_handles, width, height, layers),
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

    pub(crate) fn handle_destroy_framebuffer(
        &self,
        session: &Session,
        cmd: VulkanCommand,
    ) -> VulkanResponse {
        let (device, framebuffer) = match cmd {
            VulkanCommand::DestroyFramebuffer {
                device,
                framebuffer,
            } => (device, framebuffer),
            _ => unreachable!(),
        };

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
}
