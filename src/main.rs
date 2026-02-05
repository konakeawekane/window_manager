use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};
use wgpu::util::DeviceExt;
mod structures;
mod loaders;

// Import functions from loaders module
use loaders::{create_default_texture, load_texture};

// Import types from structures module
use structures::{DirectionalLight, PointLight, Scene, Vertex, Mesh, Material, Light, LightType};
use cgmath::InnerSpace;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    camera_position: [f32; 3],
    _padding1: f32,
    num_directional_lights: u32,
    num_point_lights: u32,
    _padding2: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DirectionalLights {
    lights: [DirectionalLight; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PointLights {
    lights: [PointLight; 8],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialProperties {
    albedo: [f32; 3],
    _padding_after_albedo: f32,  // Padding to align vec3 to 16 bytes (WGSL requirement)
    metallic: f32,
    roughness: f32,
    _padding: [f32; 2],
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    directional_lights_buffer: wgpu::Buffer,
    point_lights_buffer: wgpu::Buffer,
    default_material_bind_group: wgpu::BindGroup,
    depth_texture: wgpu::Texture,
    scene: Scene,
    _window: &'static winit::window::Window,
}

fn create_view_proj_matrix(aspect_ratio: f32) ->
cgmath::Matrix4<f32> {
    let view = cgmath::Matrix4::look_at_rh(
        cgmath::Point3::new(2.0, 4.0, 6.0),
        cgmath::Point3::new(0.0, 0.0, 0.0),
        cgmath::Vector3::unit_y(),
    );
    let proj = cgmath::perspective(cgmath::Deg(45.0),
    aspect_ratio, 0.1, 100.0);
    return proj * view;
}

fn create_model_matrix() -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_scale(1.0)
}

impl State {
    async fn new(window: winit::window::Window) -> Result<Self, Box<dyn std::error::Error>> {
        let size = window.inner_size();

        // Request Vulkan backend explicitly
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        // Create surface - window is moved into State
        // Use Box::leak to create a 'static reference for wgpu surface
        let window_box = Box::new(window);
        let window_static: &'static winit::window::Window = Box::leak(window_box);
        let surface = instance.create_surface(window_static)?;

        // Request adapter with Vulkan preference
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find a Vulkan adapter. Make sure Vulkan drivers are installed.")?;

        // Check if adapter is Vulkan
        if !matches!(adapter.get_info().backend, wgpu::Backend::Vulkan) {
            eprintln!("Error: Vulkan backend not available. Found backend: {:?}", adapter.get_info().backend);
            std::process::exit(1);
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);


        
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("Depth Texture"),
            view_formats: &[],
        });

        // Load shaders from external files
        // Try multiple paths to handle different working directories
        let vert_paths = [
            "src/shaders/pbr.vert.wgsl",
            "window_manager/src/shaders/pbr.vert.wgsl",
        ];
        let mut shader_source = None;
        for path in &vert_paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                shader_source = Some(content);
                break;
            }
        }

        let aspect = config.width as f32 / config.height as f32;
        let view_proj = create_view_proj_matrix(aspect);
        let model = create_model_matrix();

        let uniforms = Uniforms {
            view_proj: view_proj.into(),
            model: model.into(),
            camera_position: [0.0, 0.0, 3.0],
            _padding1: 0.0,
            num_directional_lights: 0,
            num_point_lights: 0,
            _padding2: [0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count:None,
                },
            ],
        });

        // Create light buffers
        let directional_lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Directional Lights Buffer"),
            size: std::mem::size_of::<DirectionalLights>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let point_lights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Point Lights Buffer"),
            size: std::mem::size_of::<PointLights>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry{
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: directional_lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: point_lights_buffer.as_entire_binding(),
                }
            ],
        });

        let material_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Bind Group Layout"),
            entries: &[
                // Albedo texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Albedo texture sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Metallic-roughness texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Metallic-roughness sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Normal texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // Normal sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Material properties (albedo, metallic, roughness)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let lights_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Lights Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let lights_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lights Bind Group"),
            layout: &lights_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: directional_lights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: point_lights_buffer.as_entire_binding(),
                },
            ],
        });
        
        let shader_source = shader_source.ok_or("Failed to find triangle.vert.wgsl shader file")?;
        let vertex_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let frag_paths = [
            "src/shaders/pbr.frag.wgsl",
            "window_manager/src/shaders/pbr.frag.wgsl",
        ];
        let mut shader_source = None;
        for path in &frag_paths {
            if let Ok(content) = std::fs::read_to_string(path) {
                shader_source = Some(content);
                break;
            }
        }
        let shader_source = shader_source.ok_or("Failed to find triangle.frag.wgsl shader file")?;
        let fragment_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &uniform_bind_group_layout,   // @group(0) - uniforms and lights
                &material_bind_group_layout,  // @group(1) - material textures and properties
            ],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // Create a simple scene
        let mut scene = Scene::new();

        // Create a default material
        let default_texture = create_default_texture(&device, &queue);
        let default_texture_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create separate textures for each material property (wgpu::Texture doesn't implement Clone)
        let metallic_roughness_texture = create_default_texture(&device, &queue);
        let metallic_roughness_texture_view = metallic_roughness_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_texture = create_default_texture(&device, &queue);
        let normal_texture_view = normal_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut default_material = Material::default();
        default_material.albedo_texture = Some(default_texture);
        default_material.metallic_roughness_texture = Some(metallic_roughness_texture);
        default_material.normal_texture = Some(normal_texture);

        // Create material properties buffer
        let material_props = MaterialProperties {
            albedo: default_material.albedo,
            _padding_after_albedo: 0.0,
            metallic: default_material.metallic,
            roughness: default_material.roughness,
            _padding: [0.0; 2],
        };

        let material_props_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Properties Buffer"),
            contents: bytemuck::cast_slice(&[material_props]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create material bind group for the material
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Material Bind Group"),
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&metallic_roughness_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&normal_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: material_props_buffer.as_entire_binding(),
                },
            ],
        });

        // Create a separate default material bind group for State (can't clone BindGroup)
        let default_material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Default Material Bind Group"),
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&metallic_roughness_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&normal_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: material_props_buffer.as_entire_binding(),
                },
            ],
        });

        default_material.bind_group = Some(material_bind_group);
        let material_index = scene.add_material(default_material);

        fn create_cube_vertices() -> Vec<Vertex> {
            vec![
                // Front face
                Vertex { position: [-1.0, -1.0,  1.0], normal: [0.0, 0.0, 1.0], tex_coords: [0.0, 1.0] },
                Vertex { position: [ 1.0, -1.0,  1.0], normal: [0.0, 0.0, 1.0], tex_coords: [1.0, 1.0] },
                Vertex { position: [ 1.0,  1.0,  1.0], normal: [0.0, 0.0, 1.0], tex_coords: [1.0, 0.0] },
                Vertex { position: [-1.0,  1.0,  1.0], normal: [0.0, 0.0, 1.0], tex_coords: [0.0, 0.0] },
                // Back face
                Vertex { position: [-1.0, -1.0, -1.0], normal: [0.0, 0.0, -1.0], tex_coords: [1.0, 1.0] },
                Vertex { position: [-1.0,  1.0, -1.0], normal: [0.0, 0.0, -1.0], tex_coords: [1.0, 0.0] },
                Vertex { position: [ 1.0,  1.0, -1.0], normal: [0.0, 0.0, -1.0], tex_coords: [0.0, 0.0] },
                Vertex { position: [ 1.0, -1.0, -1.0], normal: [0.0, 0.0, -1.0], tex_coords: [0.0, 1.0] },
                // Top face
                Vertex { position: [-1.0,  1.0, -1.0], normal: [0.0, 1.0, 0.0], tex_coords: [0.0, 1.0] },
                Vertex { position: [-1.0,  1.0,  1.0], normal: [0.0, 1.0, 0.0], tex_coords: [0.0, 0.0] },
                Vertex { position: [ 1.0,  1.0,  1.0], normal: [0.0, 1.0, 0.0], tex_coords: [1.0, 0.0] },
                Vertex { position: [ 1.0,  1.0, -1.0], normal: [0.0, 1.0, 0.0], tex_coords: [1.0, 1.0] },
                // Bottom face
                Vertex { position: [-1.0, -1.0, -1.0], normal: [0.0, -1.0, 0.0], tex_coords: [1.0, 1.0] },
                Vertex { position: [ 1.0, -1.0, -1.0], normal: [0.0, -1.0, 0.0], tex_coords: [0.0, 1.0] },
                Vertex { position: [ 1.0, -1.0,  1.0], normal: [0.0, -1.0, 0.0], tex_coords: [0.0, 0.0] },
                Vertex { position: [-1.0, -1.0,  1.0], normal: [0.0, -1.0, 0.0], tex_coords: [1.0, 0.0] },
                // Right face
                Vertex { position: [ 1.0, -1.0, -1.0], normal: [1.0, 0.0, 0.0], tex_coords: [1.0, 1.0] },
                Vertex { position: [ 1.0,  1.0, -1.0], normal: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0] },
                Vertex { position: [ 1.0,  1.0,  1.0], normal: [1.0, 0.0, 0.0], tex_coords: [0.0, 0.0] },
                Vertex { position: [ 1.0, -1.0,  1.0], normal: [1.0, 0.0, 0.0], tex_coords: [0.0, 1.0] },
                // Left face
                Vertex { position: [-1.0, -1.0, -1.0], normal: [-1.0, 0.0, 0.0], tex_coords: [0.0, 1.0] },
                Vertex { position: [-1.0, -1.0,  1.0], normal: [-1.0, 0.0, 0.0], tex_coords: [1.0, 1.0] },
                Vertex { position: [-1.0,  1.0,  1.0], normal: [-1.0, 0.0, 0.0], tex_coords: [1.0, 0.0] },
                Vertex { position: [-1.0,  1.0, -1.0], normal: [-1.0, 0.0, 0.0], tex_coords: [0.0, 0.0] },
            ]
        }
        
        fn create_cube_indices() -> Vec<u32> {
            vec![
                0,  1,  2,   2,  3,  0,   // front
                4,  5,  6,   6,  7,  4,   // back
                8,  9,  10,  10, 11, 8,   // top
                12, 13, 14,  14, 15, 12,  // bottom
                16, 17, 18,  18, 19, 16,  // right
                20, 21, 22,  22, 23, 20,  // left
            ]
        }
        
        // Create a simple cube mesh (you can replace this with model loading)
        let cube_vertices = create_cube_vertices();
        let cube_indices = create_cube_indices();
        let cube_mesh = Mesh::new(
            "Cube".to_string(),
            &cube_vertices,
            &cube_indices,
            material_index,
            &device,
        );
        scene.add_mesh(cube_mesh);

        // Add a directional light
        let sun_light = Light {
            name: "Sun".to_string(),
            light_type: LightType::Directional {
                direction: cgmath::Vector3::new(-0.5, -1.0, -0.3).normalize(),
                color: cgmath::Vector3::new(1.0, 1.0, 0.95),
                intensity: 0.0,
            },
        };
        scene.add_light(sun_light);

        // Add a point light
        let point_light = Light {
            name: "Point Light".to_string(),
            light_type: LightType::Point {
                position: cgmath::Point3::new(2.0, 2.0, 2.0),
                color: cgmath::Vector3::new(1.0, 1.0, 1.0),
                intensity: 2.0,
                radius: 0.0,
            },
        };
        scene.add_light(point_light);


        // Create vertex buffer with triangle
        let vertices = [
            Vertex {
                position: [0.0, 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.5, 0.0],
            },
            Vertex {
                position: [-0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [0.0, 1.0],
            },
            Vertex {
                position: [0.5, -0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                tex_coords: [1.0, 1.0],
            },
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            uniform_buffer,
            uniform_bind_group,
            directional_lights_buffer,
            point_lights_buffer,
            default_material_bind_group,
            depth_texture,
            scene,
            _window: window_static,
        })
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d {
                    width: new_size.width,
                    height: new_size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                label: Some("Depth Texture"),
                view_formats: &[],
            });
        }
    }

    pub fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {
        // Future: Add update logic here (animation, physics, etc.)
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {

        self.update_light_buffers();

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = self.depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            // Set default material bind group - will be overridden per mesh if needed
            render_pass.set_bind_group(1, &self.default_material_bind_group, &[]);

            for mesh in &self.scene.meshes {
                let uniforms = Uniforms {
                    view_proj: create_view_proj_matrix(
                        self.config.width as f32 / self.config.height as f32
                    ).into(),
                    model: mesh.transform.into(),
                    camera_position: self.scene.camera_position.into(),
                    num_directional_lights: self.count_directional_lights() as u32,
                    num_point_lights: self.count_point_lights() as u32,
                    _padding1: 0.0,
                    _padding2: [0.0; 2],
                };

                self.queue.write_buffer(
                    &self.uniform_buffer,
                    0,
                    bytemuck::cast_slice(&[uniforms]),
                );

                // Set material bind group (use default if material doesn't have one)
                if let Some(material) = self.scene.materials.get(mesh.material_index) {
                    if let Some(bind_group) = &material.bind_group {
                        render_pass.set_bind_group(1, bind_group, &[]);
                    } else {
                        render_pass.set_bind_group(1, &self.default_material_bind_group, &[]);
                    }
                } else {
                    render_pass.set_bind_group(1, &self.default_material_bind_group, &[]);
                }

                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass.set_index_buffer(mesh.index_buffer.slice(..),
                wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update_light_buffers(&mut self) {
        let mut directional_lights_array = [DirectionalLight {
            direction: [0.0; 3],
            _padding: 0.0,
            color: [0.0; 3],
            intensity: 0.0,
        }; 4];
    
        let mut point_lights_array = [PointLight {
            position: [0.0; 3],
            radius: 0.0,
            color: [0.0; 3],
            intensity: 0.0,
        }; 8];
    
        let mut dir_idx = 0;
        let mut point_idx = 0;
    
        for light in &self.scene.lights {
        match &light.light_type {
            LightType::Directional { direction, color, intensity } if dir_idx < 4 => {
                directional_lights_array[dir_idx] = DirectionalLight {
                    direction: [direction.x, direction.y, direction.z],
                    _padding: 0.0,
                    color: [color.x, color.y, color.z],
                    intensity: *intensity,
                };
                dir_idx += 1;
            }
            LightType::Point { position, color, intensity, radius } if point_idx < 8 => {
                point_lights_array[point_idx] = PointLight {
                    position: [position.x, position.y, position.z],
                    radius: *radius,
                    color: [color.x, color.y, color.z],
                    intensity: *intensity,
                };
                point_idx += 1;
            }
            _ => {}
        }
        }
    
        self.queue.write_buffer(
            &self.directional_lights_buffer,
            0,
            bytemuck::cast_slice(&[DirectionalLights {
                lights: directional_lights_array,
            }]),
        );
    
        self.queue.write_buffer(
            &self.point_lights_buffer,
            0,
            bytemuck::cast_slice(&[PointLights {
                lights: point_lights_array,
            }]),
        );
    }

    fn count_directional_lights(&self) -> usize {
        self.scene.lights.iter()
        .filter(|l| matches!(l.light_type, LightType::Directional {..}))
        .count()
    }
    
    fn count_point_lights(&self) -> usize {
        self.scene.lights.iter()
            .filter(|l| matches!(l.light_type, LightType::Point {..}))
            .count()
    }

}

fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let window = event_loop.create_window(winit::window::WindowAttributes::default()
        .with_title("WGPU Hello Triangle"))
        .unwrap();

    // Use pollster to block on async operations
    // Window is moved into State
    let mut state = pollster::block_on(State::new(window))
        .expect("Failed to initialize wgpu state");
    

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == state._window.id() => {
                if !state.input(event) {
                    match event {
                        WindowEvent::CloseRequested => {
                            elwt.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { .. } => {
                            state.resize(state._window.inner_size());
                        }
                        WindowEvent::RedrawRequested => {
                            state.update();
                            match state.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    eprintln!("Out of memory");
                                    elwt.exit();
                                }
                                Err(e) => eprintln!("{:?}", e),
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                state._window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
