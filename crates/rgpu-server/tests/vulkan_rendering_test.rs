//! Integration test: Vulkan Rendering Pipeline
//!
//! Tests off-screen triangle rendering via VulkanExecutor (no networking).
//! Creates images, render passes, framebuffers, graphics pipelines, records
//! draw commands, submits them, and reads back pixel data.
//!
//! Run with: cargo test --test vulkan_rendering_test -- --nocapture

use rgpu_protocol::vulkan_commands::*;
use rgpu_server::session::Session;
use rgpu_server::vulkan_executor::VulkanExecutor;

fn make_session() -> Session {
    Session::new(1, 0, "render_test".to_string())
}

/// Compile WGSL to SPIR-V bytes using naga.
fn compile_wgsl_to_spirv(
    wgsl_source: &str,
    stage: naga::ShaderStage,
    entry_point: &str,
) -> Vec<u8> {
    let module = naga::front::wgsl::parse_str(wgsl_source).expect("failed to parse WGSL");

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("WGSL validation failed");

    let options = naga::back::spv::Options {
        lang_version: (1, 0),
        ..Default::default()
    };
    let pipeline_options = naga::back::spv::PipelineOptions {
        shader_stage: stage,
        entry_point: entry_point.to_string(),
    };

    let mut writer =
        naga::back::spv::Writer::new(&options).expect("failed to create SPIR-V writer");
    let mut words = Vec::new();
    writer
        .write(&module, &info, Some(&pipeline_options), &None, &mut words)
        .expect("failed to generate SPIR-V");
    let mut spirv_bytes = Vec::new();
    for word in &words {
        spirv_bytes.extend_from_slice(&word.to_le_bytes());
    }
    spirv_bytes
}

fn compile_vertex_shader() -> Vec<u8> {
    let wgsl = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(0.0, -0.5),
        vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, 0.5),
    );
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.color = colors[vertex_index];
    return out;
}
"#;
    compile_wgsl_to_spirv(wgsl, naga::ShaderStage::Vertex, "main")
}

fn compile_fragment_shader() -> Vec<u8> {
    let wgsl = r#"
@fragment
fn main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(color, 1.0);
}
"#;
    compile_wgsl_to_spirv(wgsl, naga::ShaderStage::Fragment, "main")
}

/// Helper: create instance, enumerate physical devices, create device + queue.
fn setup_device() -> (
    VulkanExecutor,
    Session,
    rgpu_protocol::handle::NetworkHandle,
    rgpu_protocol::handle::NetworkHandle,
    rgpu_protocol::handle::NetworkHandle,
    rgpu_protocol::handle::NetworkHandle,
    u32,
) {
    let executor = VulkanExecutor::new();
    let session = make_session();

    let instance = match executor.execute(
        &session,
        VulkanCommand::CreateInstance {
            app_name: Some("RenderTest".to_string()),
            app_version: 1,
            engine_name: None,
            engine_version: 0,
            api_version: ash::vk::make_api_version(0, 1, 0, 0),
            enabled_extensions: Vec::new(),
            enabled_layers: Vec::new(),
        },
    ) {
        VulkanResponse::InstanceCreated { handle } => handle,
        other => panic!("expected InstanceCreated, got {:?}", other),
    };

    let phys_dev = match executor.execute(
        &session,
        VulkanCommand::EnumeratePhysicalDevices { instance },
    ) {
        VulkanResponse::PhysicalDevices { handles } => {
            assert!(!handles.is_empty(), "no physical devices");
            handles[0]
        }
        other => panic!("expected PhysicalDevices, got {:?}", other),
    };

    // Find graphics queue family
    let queue_family = match executor.execute(
        &session,
        VulkanCommand::GetPhysicalDeviceQueueFamilyProperties {
            physical_device: phys_dev,
        },
    ) {
        VulkanResponse::QueueFamilyProperties { families } => families
            .iter()
            .enumerate()
            .find(|(_, qf)| qf.queue_flags & 0x00000001 != 0) // GRAPHICS_BIT
            .map(|(i, _)| i as u32)
            .expect("no graphics queue family"),
        other => panic!("expected QueueFamilyProperties, got {:?}", other),
    };

    let device = match executor.execute(
        &session,
        VulkanCommand::CreateDevice {
            physical_device: phys_dev,
            queue_create_infos: vec![DeviceQueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
            enabled_extensions: Vec::new(),
            enabled_features: None,
        },
    ) {
        VulkanResponse::DeviceCreated { handle } => handle,
        other => panic!("expected DeviceCreated, got {:?}", other),
    };

    let queue = match executor.execute(
        &session,
        VulkanCommand::GetDeviceQueue {
            device,
            queue_family_index: queue_family,
            queue_index: 0,
        },
    ) {
        VulkanResponse::QueueRetrieved { handle } => handle,
        other => panic!("expected Queue, got {:?}", other),
    };

    (executor, session, instance, phys_dev, device, queue, queue_family)
}

/// Helper: find memory type matching requirements.
fn find_memory_type(
    executor: &VulkanExecutor,
    session: &Session,
    phys_dev: rgpu_protocol::handle::NetworkHandle,
    type_bits: u32,
    properties: u32,
) -> u32 {
    match executor.execute(
        session,
        VulkanCommand::GetPhysicalDeviceMemoryProperties {
            physical_device: phys_dev,
        },
    ) {
        VulkanResponse::PhysicalDeviceMemoryProperties {
            memory_types, ..
        } => memory_types
            .iter()
            .enumerate()
            .find(|(i, mt)| {
                (type_bits & (1 << i)) != 0 && (mt.property_flags & properties) == properties
            })
            .map(|(i, _)| i as u32)
            .expect(&format!(
                "no memory type matching bits=0x{:x} props=0x{:x}",
                type_bits, properties
            )),
        other => panic!("expected PhysicalDeviceMemoryProperties, got {:?}", other),
    }
}

#[test]
fn test_create_image_and_image_view() {
    let (executor, session, instance, phys_dev, device, _queue, _qf) = setup_device();

    // Create image (64x64, R8G8B8A8_UNORM, COLOR_ATTACHMENT | TRANSFER_SRC)
    let image = match executor.execute(
        &session,
        VulkanCommand::CreateImage {
            device,
            create_info: SerializedImageCreateInfo {
                flags: 0,
                image_type: 1, // VK_IMAGE_TYPE_2D
                format: 37,   // VK_FORMAT_R8G8B8A8_UNORM
                extent: [64, 64, 1],
                mip_levels: 1,
                array_layers: 1,
                samples: 1,                       // SAMPLE_COUNT_1
                tiling: 0,                         // OPTIMAL
                usage: 0x00000010 | 0x00000001,    // COLOR_ATTACHMENT | TRANSFER_SRC
                sharing_mode: 0,
                queue_family_indices: Vec::new(),
                initial_layout: 0, // UNDEFINED
            },
        },
    ) {
        VulkanResponse::ImageCreated { handle } => {
            println!("Image created: {:?}", handle);
            handle
        }
        other => panic!("expected ImageCreated, got {:?}", other),
    };

    // Get image memory requirements
    let (mem_size, mem_type_bits) = match executor.execute(
        &session,
        VulkanCommand::GetImageMemoryRequirements { device, image },
    ) {
        VulkanResponse::MemoryRequirements {
            size,
            memory_type_bits,
            ..
        } => {
            println!(
                "Image mem: size={}, types=0x{:x}",
                size, memory_type_bits
            );
            (size, memory_type_bits)
        }
        other => panic!("expected MemoryRequirements, got {:?}", other),
    };

    // Allocate device-local memory
    let mem_type_idx = find_memory_type(&executor, &session, phys_dev, mem_type_bits, 0x01);
    let image_memory = match executor.execute(
        &session,
        VulkanCommand::AllocateMemory {
            device,
            alloc_size: mem_size,
            memory_type_index: mem_type_idx,
        },
    ) {
        VulkanResponse::MemoryAllocated { handle } => handle,
        other => panic!("expected MemoryAllocated, got {:?}", other),
    };

    // Bind image memory
    match executor.execute(
        &session,
        VulkanCommand::BindImageMemory {
            device,
            image,
            memory: image_memory,
            memory_offset: 0,
        },
    ) {
        VulkanResponse::Success => println!("Image bound to memory"),
        other => panic!("expected Success, got {:?}", other),
    }

    // Create image view
    let image_view = match executor.execute(
        &session,
        VulkanCommand::CreateImageView {
            device,
            image,
            view_type: 1, // 2D
            format: 37,   // R8G8B8A8_UNORM
            components: SerializedComponentMapping {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
            subresource_range: SerializedImageSubresourceRange {
                aspect_mask: 0x00000001, // COLOR
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        },
    ) {
        VulkanResponse::ImageViewCreated { handle } => {
            println!("ImageView created: {:?}", handle);
            handle
        }
        other => panic!("expected ImageViewCreated, got {:?}", other),
    };

    // Cleanup
    executor.execute(&session, VulkanCommand::DestroyImageView { device, image_view });
    executor.execute(&session, VulkanCommand::DestroyImage { device, image });
    executor.execute(
        &session,
        VulkanCommand::FreeMemory {
            device,
            memory: image_memory,
        },
    );
    executor.execute(&session, VulkanCommand::DestroyDevice { device });
    executor.execute(&session, VulkanCommand::DestroyInstance { instance });
    println!("test_create_image_and_image_view PASSED");
}

#[test]
fn test_render_pass_and_framebuffer() {
    let (executor, session, instance, phys_dev, device, _queue, _qf) = setup_device();

    // Create image + view for attachment
    let image = match executor.execute(
        &session,
        VulkanCommand::CreateImage {
            device,
            create_info: SerializedImageCreateInfo {
                flags: 0,
                image_type: 1,
                format: 37,
                extent: [64, 64, 1],
                mip_levels: 1,
                array_layers: 1,
                samples: 1,
                tiling: 0,
                usage: 0x00000010 | 0x00000001,
                sharing_mode: 0,
                queue_family_indices: Vec::new(),
                initial_layout: 0,
            },
        },
    ) {
        VulkanResponse::ImageCreated { handle } => handle,
        other => panic!("expected ImageCreated, got {:?}", other),
    };

    let (mem_size, mem_type_bits) = match executor.execute(
        &session,
        VulkanCommand::GetImageMemoryRequirements { device, image },
    ) {
        VulkanResponse::MemoryRequirements {
            size,
            memory_type_bits,
            ..
        } => (size, memory_type_bits),
        other => panic!("expected MemoryRequirements, got {:?}", other),
    };
    let mem_type = find_memory_type(&executor, &session, phys_dev, mem_type_bits, 0x01);
    let image_mem = match executor.execute(
        &session,
        VulkanCommand::AllocateMemory {
            device,
            alloc_size: mem_size,
            memory_type_index: mem_type,
        },
    ) {
        VulkanResponse::MemoryAllocated { handle } => handle,
        other => panic!("expected MemoryAllocated, got {:?}", other),
    };
    executor.execute(
        &session,
        VulkanCommand::BindImageMemory {
            device,
            image,
            memory: image_mem,
            memory_offset: 0,
        },
    );

    let image_view = match executor.execute(
        &session,
        VulkanCommand::CreateImageView {
            device,
            image,
            view_type: 1,
            format: 37,
            components: SerializedComponentMapping {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
            subresource_range: SerializedImageSubresourceRange {
                aspect_mask: 0x00000001,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        },
    ) {
        VulkanResponse::ImageViewCreated { handle } => handle,
        other => panic!("expected ImageViewCreated, got {:?}", other),
    };

    // Create render pass
    let render_pass = match executor.execute(
        &session,
        VulkanCommand::CreateRenderPass {
            device,
            attachments: vec![SerializedAttachmentDescription {
                flags: 0,
                format: 37,
                samples: 1,
                load_op: 1,           // CLEAR
                store_op: 0,          // STORE
                stencil_load_op: 2,   // DONT_CARE
                stencil_store_op: 1,  // DONT_CARE
                initial_layout: 0,    // UNDEFINED
                final_layout: 7,      // TRANSFER_SRC_OPTIMAL
            }],
            subpasses: vec![SerializedSubpassDescription {
                flags: 0,
                pipeline_bind_point: 0, // GRAPHICS
                input_attachments: Vec::new(),
                color_attachments: vec![SerializedAttachmentReference {
                    attachment: 0,
                    layout: 2, // COLOR_ATTACHMENT_OPTIMAL
                }],
                resolve_attachments: Vec::new(),
                depth_stencil_attachment: None,
                preserve_attachments: Vec::new(),
            }],
            dependencies: Vec::new(),
        },
    ) {
        VulkanResponse::RenderPassCreated { handle } => {
            println!("RenderPass created: {:?}", handle);
            handle
        }
        other => panic!("expected RenderPassCreated, got {:?}", other),
    };

    // Create framebuffer
    let framebuffer = match executor.execute(
        &session,
        VulkanCommand::CreateFramebuffer {
            device,
            render_pass,
            attachments: vec![image_view],
            width: 64,
            height: 64,
            layers: 1,
        },
    ) {
        VulkanResponse::FramebufferCreated { handle } => {
            println!("Framebuffer created: {:?}", handle);
            handle
        }
        other => panic!("expected FramebufferCreated, got {:?}", other),
    };

    // Cleanup
    executor.execute(
        &session,
        VulkanCommand::DestroyFramebuffer {
            device,
            framebuffer,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyRenderPass {
            device,
            render_pass,
        },
    );
    executor.execute(&session, VulkanCommand::DestroyImageView { device, image_view });
    executor.execute(&session, VulkanCommand::DestroyImage { device, image });
    executor.execute(
        &session,
        VulkanCommand::FreeMemory {
            device,
            memory: image_mem,
        },
    );
    executor.execute(&session, VulkanCommand::DestroyDevice { device });
    executor.execute(&session, VulkanCommand::DestroyInstance { instance });
    println!("test_render_pass_and_framebuffer PASSED");
}

#[test]
fn test_triangle_render() {
    let (executor, session, instance, phys_dev, device, queue, queue_family) = setup_device();

    println!("=== Triangle Render Test ===");

    // 1. Create render target image (64x64, R8G8B8A8_UNORM)
    let render_image = match executor.execute(
        &session,
        VulkanCommand::CreateImage {
            device,
            create_info: SerializedImageCreateInfo {
                flags: 0,
                image_type: 1,
                format: 37, // R8G8B8A8_UNORM
                extent: [64, 64, 1],
                mip_levels: 1,
                array_layers: 1,
                samples: 1,
                tiling: 0,                         // OPTIMAL
                usage: 0x00000010 | 0x00000001,    // COLOR_ATTACHMENT | TRANSFER_SRC
                sharing_mode: 0,
                queue_family_indices: Vec::new(),
                initial_layout: 0,
            },
        },
    ) {
        VulkanResponse::ImageCreated { handle } => handle,
        other => panic!("expected ImageCreated, got {:?}", other),
    };
    println!("  render image created");

    // Allocate + bind image memory
    let (img_mem_size, img_mem_bits) = match executor.execute(
        &session,
        VulkanCommand::GetImageMemoryRequirements {
            device,
            image: render_image,
        },
    ) {
        VulkanResponse::MemoryRequirements {
            size,
            memory_type_bits,
            ..
        } => (size, memory_type_bits),
        other => panic!("expected MemoryRequirements, got {:?}", other),
    };
    let img_mem_type = find_memory_type(&executor, &session, phys_dev, img_mem_bits, 0x01);
    let image_memory = match executor.execute(
        &session,
        VulkanCommand::AllocateMemory {
            device,
            alloc_size: img_mem_size,
            memory_type_index: img_mem_type,
        },
    ) {
        VulkanResponse::MemoryAllocated { handle } => handle,
        other => panic!("expected MemoryAllocated, got {:?}", other),
    };
    executor.execute(
        &session,
        VulkanCommand::BindImageMemory {
            device,
            image: render_image,
            memory: image_memory,
            memory_offset: 0,
        },
    );

    // Create image view
    let image_view = match executor.execute(
        &session,
        VulkanCommand::CreateImageView {
            device,
            image: render_image,
            view_type: 1,
            format: 37,
            components: SerializedComponentMapping {
                r: 0,
                g: 0,
                b: 0,
                a: 0,
            },
            subresource_range: SerializedImageSubresourceRange {
                aspect_mask: 0x00000001,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        },
    ) {
        VulkanResponse::ImageViewCreated { handle } => handle,
        other => panic!("expected ImageViewCreated, got {:?}", other),
    };
    println!("  image view created");

    // 2. Create readback buffer (64*64*4 = 16384 bytes)
    let readback_size: u64 = 64 * 64 * 4;
    let readback_buffer = match executor.execute(
        &session,
        VulkanCommand::CreateBuffer {
            device,
            size: readback_size,
            usage: 0x00000002, // TRANSFER_DST
            sharing_mode: 0,
            queue_family_indices: Vec::new(),
        },
    ) {
        VulkanResponse::BufferCreated { handle } => handle,
        other => panic!("expected BufferCreated, got {:?}", other),
    };
    let (buf_mem_size, buf_mem_bits) = match executor.execute(
        &session,
        VulkanCommand::GetBufferMemoryRequirements {
            device,
            buffer: readback_buffer,
        },
    ) {
        VulkanResponse::MemoryRequirements {
            size,
            memory_type_bits,
            ..
        } => (size, memory_type_bits),
        other => panic!("expected MemoryRequirements, got {:?}", other),
    };
    // HOST_VISIBLE | HOST_COHERENT
    let buf_mem_type = find_memory_type(&executor, &session, phys_dev, buf_mem_bits, 0x06);
    let buffer_memory = match executor.execute(
        &session,
        VulkanCommand::AllocateMemory {
            device,
            alloc_size: buf_mem_size,
            memory_type_index: buf_mem_type,
        },
    ) {
        VulkanResponse::MemoryAllocated { handle } => handle,
        other => panic!("expected MemoryAllocated, got {:?}", other),
    };
    executor.execute(
        &session,
        VulkanCommand::BindBufferMemory {
            device,
            buffer: readback_buffer,
            memory: buffer_memory,
            memory_offset: 0,
        },
    );
    println!("  readback buffer created");

    // 3. Create render pass
    let render_pass = match executor.execute(
        &session,
        VulkanCommand::CreateRenderPass {
            device,
            attachments: vec![SerializedAttachmentDescription {
                flags: 0,
                format: 37,
                samples: 1,
                load_op: 1,          // CLEAR
                store_op: 0,         // STORE
                stencil_load_op: 2,  // DONT_CARE
                stencil_store_op: 1, // DONT_CARE
                initial_layout: 0,   // UNDEFINED
                final_layout: 7,     // TRANSFER_SRC_OPTIMAL
            }],
            subpasses: vec![SerializedSubpassDescription {
                flags: 0,
                pipeline_bind_point: 0,
                input_attachments: Vec::new(),
                color_attachments: vec![SerializedAttachmentReference {
                    attachment: 0,
                    layout: 2, // COLOR_ATTACHMENT_OPTIMAL
                }],
                resolve_attachments: Vec::new(),
                depth_stencil_attachment: None,
                preserve_attachments: Vec::new(),
            }],
            dependencies: Vec::new(),
        },
    ) {
        VulkanResponse::RenderPassCreated { handle } => handle,
        other => panic!("expected RenderPassCreated, got {:?}", other),
    };
    println!("  render pass created");

    // 4. Create framebuffer
    let framebuffer = match executor.execute(
        &session,
        VulkanCommand::CreateFramebuffer {
            device,
            render_pass,
            attachments: vec![image_view],
            width: 64,
            height: 64,
            layers: 1,
        },
    ) {
        VulkanResponse::FramebufferCreated { handle } => handle,
        other => panic!("expected FramebufferCreated, got {:?}", other),
    };
    println!("  framebuffer created");

    // 5. Create shader modules
    let vert_spirv = compile_vertex_shader();
    let frag_spirv = compile_fragment_shader();
    println!(
        "  compiled shaders: vert={} bytes, frag={} bytes",
        vert_spirv.len(),
        frag_spirv.len()
    );

    let vert_module = match executor.execute(
        &session,
        VulkanCommand::CreateShaderModule {
            device,
            code: vert_spirv,
        },
    ) {
        VulkanResponse::ShaderModuleCreated { handle } => handle,
        other => panic!("expected ShaderModuleCreated for vertex, got {:?}", other),
    };
    let frag_module = match executor.execute(
        &session,
        VulkanCommand::CreateShaderModule {
            device,
            code: frag_spirv,
        },
    ) {
        VulkanResponse::ShaderModuleCreated { handle } => handle,
        other => panic!("expected ShaderModuleCreated for fragment, got {:?}", other),
    };
    println!("  shader modules created");

    // 6. Pipeline layout (empty) + graphics pipeline
    let pipeline_layout = match executor.execute(
        &session,
        VulkanCommand::CreatePipelineLayout {
            device,
            set_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
        },
    ) {
        VulkanResponse::PipelineLayoutCreated { handle } => handle,
        other => panic!("expected PipelineLayoutCreated, got {:?}", other),
    };

    let pipeline = match executor.execute(
        &session,
        VulkanCommand::CreateGraphicsPipelines {
            device,
            create_infos: vec![SerializedGraphicsPipelineCreateInfo {
                flags: 0,
                stages: vec![
                    SerializedPipelineShaderStageCreateInfo {
                        module: vert_module,
                        entry_point: "main".to_string(),
                        stage: 0x00000001, // VERTEX
                    },
                    SerializedPipelineShaderStageCreateInfo {
                        module: frag_module,
                        entry_point: "main".to_string(),
                        stage: 0x00000010, // FRAGMENT
                    },
                ],
                vertex_input_state: SerializedPipelineVertexInputStateCreateInfo {
                    vertex_binding_descriptions: Vec::new(),
                    vertex_attribute_descriptions: Vec::new(),
                },
                input_assembly_state: SerializedPipelineInputAssemblyStateCreateInfo {
                    topology: 3, // TRIANGLE_LIST
                    primitive_restart_enable: false,
                },
                viewport_state: Some(SerializedPipelineViewportStateCreateInfo {
                    viewports: vec![SerializedViewport {
                        x: 0.0,
                        y: 0.0,
                        width: 64.0,
                        height: 64.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                    scissors: vec![SerializedRect2D {
                        offset: [0, 0],
                        extent: [64, 64],
                    }],
                }),
                rasterization_state: SerializedPipelineRasterizationStateCreateInfo {
                    depth_clamp_enable: false,
                    rasterizer_discard_enable: false,
                    polygon_mode: 0, // FILL
                    cull_mode: 0,    // NONE
                    front_face: 0,   // CCW
                    depth_bias_enable: false,
                    depth_bias_constant_factor: 0.0,
                    depth_bias_clamp: 0.0,
                    depth_bias_slope_factor: 0.0,
                    line_width: 1.0,
                },
                multisample_state: Some(SerializedPipelineMultisampleStateCreateInfo {
                    rasterization_samples: 1,
                    sample_shading_enable: false,
                    min_sample_shading: 1.0,
                    alpha_to_coverage_enable: false,
                    alpha_to_one_enable: false,
                }),
                depth_stencil_state: None,
                color_blend_state: Some(SerializedPipelineColorBlendStateCreateInfo {
                    logic_op_enable: false,
                    logic_op: 0,
                    attachments: vec![SerializedPipelineColorBlendAttachmentState {
                        blend_enable: false,
                        src_color_blend_factor: 0,
                        dst_color_blend_factor: 0,
                        color_blend_op: 0,
                        src_alpha_blend_factor: 0,
                        dst_alpha_blend_factor: 0,
                        alpha_blend_op: 0,
                        color_write_mask: 0xF,
                    }],
                    blend_constants: [0.0, 0.0, 0.0, 0.0],
                }),
                dynamic_state: None,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
            }],
        },
    ) {
        VulkanResponse::PipelinesCreated { handles } => {
            assert_eq!(handles.len(), 1);
            handles[0]
        }
        other => panic!("expected PipelinesCreated, got {:?}", other),
    };
    println!("  graphics pipeline created");

    // 7. Command pool + buffer + fence
    let cmd_pool = match executor.execute(
        &session,
        VulkanCommand::CreateCommandPool {
            device,
            queue_family_index: queue_family,
            flags: 0x00000002, // RESET_COMMAND_BUFFER
        },
    ) {
        VulkanResponse::CommandPoolCreated { handle } => handle,
        other => panic!("expected CommandPoolCreated, got {:?}", other),
    };

    let cmd_buf = match executor.execute(
        &session,
        VulkanCommand::AllocateCommandBuffers {
            device,
            command_pool: cmd_pool,
            level: 0, // PRIMARY
            count: 1,
        },
    ) {
        VulkanResponse::CommandBuffersAllocated { handles } => {
            assert_eq!(handles.len(), 1);
            handles[0]
        }
        other => panic!("expected CommandBuffersAllocated, got {:?}", other),
    };

    let fence = match executor.execute(
        &session,
        VulkanCommand::CreateFence {
            device,
            signaled: false,
        },
    ) {
        VulkanResponse::FenceCreated { handle } => handle,
        other => panic!("expected FenceCreated, got {:?}", other),
    };
    println!("  command pool, buffer, fence created");

    // 8. Record commands and submit
    // Clear color: dark blue (0.0, 0.0, 0.2, 1.0)
    let mut clear_value = [0u8; 16];
    clear_value[0..4].copy_from_slice(&0.0f32.to_le_bytes());  // R
    clear_value[4..8].copy_from_slice(&0.0f32.to_le_bytes());  // G
    clear_value[8..12].copy_from_slice(&0.2f32.to_le_bytes()); // B
    clear_value[12..16].copy_from_slice(&1.0f32.to_le_bytes()); // A

    let recorded_commands = vec![
        RecordedCommand::BeginRenderPass {
            render_pass,
            framebuffer,
            render_area: SerializedRect2D {
                offset: [0, 0],
                extent: [64, 64],
            },
            clear_values: vec![SerializedClearValue {
                data: clear_value,
            }],
            contents: 0, // VK_SUBPASS_CONTENTS_INLINE
        },
        RecordedCommand::BindPipeline {
            pipeline_bind_point: 0, // GRAPHICS
            pipeline,
        },
        RecordedCommand::Draw {
            vertex_count: 3,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        },
        RecordedCommand::EndRenderPass,
        // Image is now TRANSFER_SRC_OPTIMAL (render pass final_layout)
        RecordedCommand::CopyImageToBuffer {
            src_image: render_image,
            src_image_layout: 7, // TRANSFER_SRC_OPTIMAL
            dst_buffer: readback_buffer,
            regions: vec![SerializedBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: SerializedImageSubresourceLayers {
                    aspect_mask: 0x00000001, // COLOR
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: [0, 0, 0],
                image_extent: [64, 64, 1],
            }],
        },
    ];

    // Submit recorded commands
    match executor.execute(
        &session,
        VulkanCommand::SubmitRecordedCommands {
            command_buffer: cmd_buf,
            commands: recorded_commands,
        },
    ) {
        VulkanResponse::Success => {}
        other => panic!("expected Success for SubmitRecordedCommands, got {:?}", other),
    }

    // Queue submit
    match executor.execute(
        &session,
        VulkanCommand::QueueSubmit {
            queue,
            submits: vec![SerializedSubmitInfo {
                wait_semaphores: Vec::new(),
                wait_dst_stage_masks: Vec::new(),
                command_buffers: vec![cmd_buf],
                signal_semaphores: Vec::new(),
            }],
            fence: Some(fence),
        },
    ) {
        VulkanResponse::Success => {}
        other => panic!("expected Success for QueueSubmit, got {:?}", other),
    }

    // Wait for fence
    match executor.execute(
        &session,
        VulkanCommand::WaitForFences {
            device,
            fences: vec![fence],
            wait_all: true,
            timeout_ns: 5_000_000_000, // 5 seconds
        },
    ) {
        VulkanResponse::FenceWaitResult { result } => {
            assert_eq!(result, 0, "fence wait timed out or failed");
        }
        other => panic!("expected FenceWaitResult, got {:?}", other),
    }
    println!("  render complete, reading back pixels");

    // 9. Map readback memory and verify
    let pixel_data = match executor.execute(
        &session,
        VulkanCommand::MapMemory {
            device,
            memory: buffer_memory,
            offset: 0,
            size: readback_size,
            flags: 0,
        },
    ) {
        VulkanResponse::MemoryMapped { data } => data,
        other => panic!("expected MemoryMapped, got {:?}", other),
    };

    assert_eq!(pixel_data.len(), readback_size as usize);

    // Check that not all pixels are zero (the triangle should have colored some pixels)
    let non_zero_pixels = pixel_data
        .chunks(4)
        .filter(|px| px[0] != 0 || px[1] != 0 || px[2] != 0)
        .count();
    println!("  non-zero pixels: {} / {}", non_zero_pixels, 64 * 64);
    assert!(
        non_zero_pixels > 0,
        "expected some non-zero pixels from triangle render"
    );

    // Check that corner pixels match clear color (dark blue: R=0, G=0, B~=51, A=255)
    // Top-left corner pixel
    let corner = &pixel_data[0..4];
    println!(
        "  corner pixel: R={} G={} B={} A={}",
        corner[0], corner[1], corner[2], corner[3]
    );
    // Clear color is (0, 0, 0.2, 1.0) = (0, 0, 51, 255) in UNORM
    assert_eq!(corner[0], 0, "corner R should be 0");
    assert_eq!(corner[1], 0, "corner G should be 0");
    assert!(corner[2] >= 45 && corner[2] <= 55, "corner B should be ~51 (got {})", corner[2]);

    // Unmap
    executor.execute(
        &session,
        VulkanCommand::UnmapMemory {
            device,
            memory: buffer_memory,
            written_data: None,
            offset: 0,
        },
    );

    // 10. Cleanup
    executor.execute(
        &session,
        VulkanCommand::DestroyFence { device, fence },
    );
    executor.execute(
        &session,
        VulkanCommand::FreeCommandBuffers {
            device,
            command_pool: cmd_pool,
            command_buffers: vec![cmd_buf],
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyCommandPool {
            device,
            command_pool: cmd_pool,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyPipeline { device, pipeline },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyPipelineLayout {
            device,
            layout: pipeline_layout,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyShaderModule {
            device,
            shader_module: vert_module,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyShaderModule {
            device,
            shader_module: frag_module,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyFramebuffer {
            device,
            framebuffer,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyRenderPass {
            device,
            render_pass,
        },
    );
    executor.execute(&session, VulkanCommand::DestroyImageView { device, image_view });
    executor.execute(
        &session,
        VulkanCommand::DestroyImage {
            device,
            image: render_image,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::FreeMemory {
            device,
            memory: image_memory,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::DestroyBuffer {
            device,
            buffer: readback_buffer,
        },
    );
    executor.execute(
        &session,
        VulkanCommand::FreeMemory {
            device,
            memory: buffer_memory,
        },
    );
    executor.execute(&session, VulkanCommand::DestroyDevice { device });
    executor.execute(&session, VulkanCommand::DestroyInstance { instance });

    println!("=== test_triangle_render PASSED ===");
}
