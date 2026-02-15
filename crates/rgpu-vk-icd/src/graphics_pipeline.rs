//! Graphics pipeline creation for the Vulkan ICD.

use std::ffi::CStr;

use ash::vk;
use ash::vk::Handle;

use crate::dispatch::DispatchableHandle;
use crate::handle_store;
use crate::send_vulkan_command;

use rgpu_protocol::vulkan_commands::{
    SerializedGraphicsPipelineCreateInfo, SerializedPipelineColorBlendAttachmentState,
    SerializedPipelineColorBlendStateCreateInfo, SerializedPipelineDepthStencilStateCreateInfo,
    SerializedPipelineDynamicStateCreateInfo, SerializedPipelineInputAssemblyStateCreateInfo,
    SerializedPipelineMultisampleStateCreateInfo,
    SerializedPipelineRasterizationStateCreateInfo, SerializedPipelineShaderStageCreateInfo,
    SerializedPipelineVertexInputStateCreateInfo, SerializedPipelineViewportStateCreateInfo,
    SerializedRect2D, SerializedStencilOpState, SerializedVertexInputAttributeDescription,
    SerializedVertexInputBindingDescription, SerializedViewport, VulkanCommand, VulkanResponse,
};

// ── vkCreateGraphicsPipelines ────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn vkCreateGraphicsPipelines(
    device: vk::Device,
    _pipeline_cache: vk::PipelineCache,
    create_info_count: u32,
    p_create_infos: *const vk::GraphicsPipelineCreateInfo<'_>,
    _p_allocator: *const vk::AllocationCallbacks<'_>,
    p_pipelines: *mut vk::Pipeline,
) -> vk::Result {
    if p_create_infos.is_null() || p_pipelines.is_null() || create_info_count == 0 {
        return vk::Result::ERROR_OUT_OF_HOST_MEMORY;
    }

    let disp = device.as_raw() as *const DispatchableHandle;
    let dev_local_id = DispatchableHandle::get_id(disp);

    let dev_handle = match handle_store::get_device(dev_local_id) {
        Some(h) => h,
        None => return vk::Result::ERROR_DEVICE_LOST,
    };

    let mut serialized_cis = Vec::new();

    for i in 0..create_info_count as usize {
        let ci = &*p_create_infos.add(i);

        // Shader stages
        let mut stages = Vec::new();
        if !ci.p_stages.is_null() && ci.stage_count > 0 {
            for j in 0..ci.stage_count as usize {
                let stage = &*ci.p_stages.add(j);
                let module_handle = match handle_store::get_shader_module(stage.module.as_raw()) {
                    Some(h) => h,
                    None => return vk::Result::ERROR_UNKNOWN,
                };
                let entry_point = if !stage.p_name.is_null() {
                    CStr::from_ptr(stage.p_name)
                        .to_str()
                        .unwrap_or("main")
                        .to_string()
                } else {
                    "main".to_string()
                };
                stages.push(SerializedPipelineShaderStageCreateInfo {
                    module: module_handle,
                    entry_point,
                    stage: stage.stage.as_raw(),
                });
            }
        }

        // Vertex input state
        let vertex_input_state = if !ci.p_vertex_input_state.is_null() {
            let vis = &*ci.p_vertex_input_state;
            let bindings = if !vis.p_vertex_binding_descriptions.is_null()
                && vis.vertex_binding_description_count > 0
            {
                std::slice::from_raw_parts(
                    vis.p_vertex_binding_descriptions,
                    vis.vertex_binding_description_count as usize,
                )
                .iter()
                .map(|b| SerializedVertexInputBindingDescription {
                    binding: b.binding,
                    stride: b.stride,
                    input_rate: b.input_rate.as_raw(),
                })
                .collect()
            } else {
                Vec::new()
            };
            let attrs = if !vis.p_vertex_attribute_descriptions.is_null()
                && vis.vertex_attribute_description_count > 0
            {
                std::slice::from_raw_parts(
                    vis.p_vertex_attribute_descriptions,
                    vis.vertex_attribute_description_count as usize,
                )
                .iter()
                .map(|a| SerializedVertexInputAttributeDescription {
                    location: a.location,
                    binding: a.binding,
                    format: a.format.as_raw(),
                    offset: a.offset,
                })
                .collect()
            } else {
                Vec::new()
            };
            SerializedPipelineVertexInputStateCreateInfo {
                vertex_binding_descriptions: bindings,
                vertex_attribute_descriptions: attrs,
            }
        } else {
            SerializedPipelineVertexInputStateCreateInfo {
                vertex_binding_descriptions: Vec::new(),
                vertex_attribute_descriptions: Vec::new(),
            }
        };

        // Input assembly state
        let input_assembly_state = if !ci.p_input_assembly_state.is_null() {
            let ias = &*ci.p_input_assembly_state;
            SerializedPipelineInputAssemblyStateCreateInfo {
                topology: ias.topology.as_raw(),
                primitive_restart_enable: ias.primitive_restart_enable != 0,
            }
        } else {
            SerializedPipelineInputAssemblyStateCreateInfo {
                topology: 0,
                primitive_restart_enable: false,
            }
        };

        // Viewport state
        let viewport_state = if !ci.p_viewport_state.is_null() {
            let vps = &*ci.p_viewport_state;
            let viewports = if !vps.p_viewports.is_null() && vps.viewport_count > 0 {
                std::slice::from_raw_parts(vps.p_viewports, vps.viewport_count as usize)
                    .iter()
                    .map(|v| SerializedViewport {
                        x: v.x,
                        y: v.y,
                        width: v.width,
                        height: v.height,
                        min_depth: v.min_depth,
                        max_depth: v.max_depth,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let scissors = if !vps.p_scissors.is_null() && vps.scissor_count > 0 {
                std::slice::from_raw_parts(vps.p_scissors, vps.scissor_count as usize)
                    .iter()
                    .map(|s| SerializedRect2D {
                        offset: [s.offset.x, s.offset.y],
                        extent: [s.extent.width, s.extent.height],
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Some(SerializedPipelineViewportStateCreateInfo {
                viewports,
                scissors,
            })
        } else {
            None
        };

        // Rasterization state
        let rasterization_state = if !ci.p_rasterization_state.is_null() {
            let rs = &*ci.p_rasterization_state;
            SerializedPipelineRasterizationStateCreateInfo {
                depth_clamp_enable: rs.depth_clamp_enable != 0,
                rasterizer_discard_enable: rs.rasterizer_discard_enable != 0,
                polygon_mode: rs.polygon_mode.as_raw(),
                cull_mode: rs.cull_mode.as_raw(),
                front_face: rs.front_face.as_raw(),
                depth_bias_enable: rs.depth_bias_enable != 0,
                depth_bias_constant_factor: rs.depth_bias_constant_factor,
                depth_bias_clamp: rs.depth_bias_clamp,
                depth_bias_slope_factor: rs.depth_bias_slope_factor,
                line_width: rs.line_width,
            }
        } else {
            SerializedPipelineRasterizationStateCreateInfo {
                depth_clamp_enable: false,
                rasterizer_discard_enable: false,
                polygon_mode: 0,
                cull_mode: 0,
                front_face: 0,
                depth_bias_enable: false,
                depth_bias_constant_factor: 0.0,
                depth_bias_clamp: 0.0,
                depth_bias_slope_factor: 0.0,
                line_width: 1.0,
            }
        };

        // Multisample state
        let multisample_state = if !ci.p_multisample_state.is_null() {
            let ms = &*ci.p_multisample_state;
            Some(SerializedPipelineMultisampleStateCreateInfo {
                rasterization_samples: ms.rasterization_samples.as_raw(),
                sample_shading_enable: ms.sample_shading_enable != 0,
                min_sample_shading: ms.min_sample_shading,
                alpha_to_coverage_enable: ms.alpha_to_coverage_enable != 0,
                alpha_to_one_enable: ms.alpha_to_one_enable != 0,
            })
        } else {
            None
        };

        // Depth stencil state
        let depth_stencil_state = if !ci.p_depth_stencil_state.is_null() {
            let ds = &*ci.p_depth_stencil_state;
            Some(SerializedPipelineDepthStencilStateCreateInfo {
                depth_test_enable: ds.depth_test_enable != 0,
                depth_write_enable: ds.depth_write_enable != 0,
                depth_compare_op: ds.depth_compare_op.as_raw(),
                depth_bounds_test_enable: ds.depth_bounds_test_enable != 0,
                stencil_test_enable: ds.stencil_test_enable != 0,
                front: SerializedStencilOpState {
                    fail_op: ds.front.fail_op.as_raw(),
                    pass_op: ds.front.pass_op.as_raw(),
                    depth_fail_op: ds.front.depth_fail_op.as_raw(),
                    compare_op: ds.front.compare_op.as_raw(),
                    compare_mask: ds.front.compare_mask,
                    write_mask: ds.front.write_mask,
                    reference: ds.front.reference,
                },
                back: SerializedStencilOpState {
                    fail_op: ds.back.fail_op.as_raw(),
                    pass_op: ds.back.pass_op.as_raw(),
                    depth_fail_op: ds.back.depth_fail_op.as_raw(),
                    compare_op: ds.back.compare_op.as_raw(),
                    compare_mask: ds.back.compare_mask,
                    write_mask: ds.back.write_mask,
                    reference: ds.back.reference,
                },
                min_depth_bounds: ds.min_depth_bounds,
                max_depth_bounds: ds.max_depth_bounds,
            })
        } else {
            None
        };

        // Color blend state
        let color_blend_state = if !ci.p_color_blend_state.is_null() {
            let cbs = &*ci.p_color_blend_state;
            let attachments = if !cbs.p_attachments.is_null() && cbs.attachment_count > 0 {
                std::slice::from_raw_parts(cbs.p_attachments, cbs.attachment_count as usize)
                    .iter()
                    .map(|a| SerializedPipelineColorBlendAttachmentState {
                        blend_enable: a.blend_enable != 0,
                        src_color_blend_factor: a.src_color_blend_factor.as_raw(),
                        dst_color_blend_factor: a.dst_color_blend_factor.as_raw(),
                        color_blend_op: a.color_blend_op.as_raw(),
                        src_alpha_blend_factor: a.src_alpha_blend_factor.as_raw(),
                        dst_alpha_blend_factor: a.dst_alpha_blend_factor.as_raw(),
                        alpha_blend_op: a.alpha_blend_op.as_raw(),
                        color_write_mask: a.color_write_mask.as_raw(),
                    })
                    .collect()
            } else {
                Vec::new()
            };
            Some(SerializedPipelineColorBlendStateCreateInfo {
                logic_op_enable: cbs.logic_op_enable != 0,
                logic_op: cbs.logic_op.as_raw(),
                attachments,
                blend_constants: cbs.blend_constants,
            })
        } else {
            None
        };

        // Dynamic state
        let dynamic_state = if !ci.p_dynamic_state.is_null() {
            let dyn_s = &*ci.p_dynamic_state;
            if !dyn_s.p_dynamic_states.is_null() && dyn_s.dynamic_state_count > 0 {
                let states: Vec<i32> = std::slice::from_raw_parts(
                    dyn_s.p_dynamic_states,
                    dyn_s.dynamic_state_count as usize,
                )
                .iter()
                .map(|d| d.as_raw())
                .collect();
                Some(SerializedPipelineDynamicStateCreateInfo {
                    dynamic_states: states,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Resolve layout and render pass
        let layout_handle = match handle_store::get_pipeline_layout(ci.layout.as_raw()) {
            Some(h) => h,
            None => return vk::Result::ERROR_UNKNOWN,
        };
        let rp_handle = match handle_store::get_render_pass(ci.render_pass.as_raw()) {
            Some(h) => h,
            None => return vk::Result::ERROR_UNKNOWN,
        };

        serialized_cis.push(SerializedGraphicsPipelineCreateInfo {
            flags: ci.flags.as_raw(),
            stages,
            vertex_input_state,
            input_assembly_state,
            viewport_state,
            rasterization_state,
            multisample_state,
            depth_stencil_state,
            color_blend_state,
            dynamic_state,
            layout: layout_handle,
            render_pass: rp_handle,
            subpass: ci.subpass,
        });
    }

    let cmd = VulkanCommand::CreateGraphicsPipelines {
        device: dev_handle,
        create_infos: serialized_cis,
    };

    match send_vulkan_command(cmd) {
        Ok(VulkanResponse::PipelinesCreated { handles }) => {
            for (i, h) in handles.iter().enumerate() {
                let local_id = handle_store::store_pipeline(*h);
                *p_pipelines.add(i) = vk::Pipeline::from_raw(local_id);
            }
            vk::Result::SUCCESS
        }
        Ok(VulkanResponse::Error { code, .. }) => vk::Result::from_raw(code),
        _ => vk::Result::ERROR_UNKNOWN,
    }
}
