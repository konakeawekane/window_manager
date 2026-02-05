#[derive(Debug)]
pub struct Material {
    pub name: String,
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub albedo_texture: Option<wgpu::Texture>,
    pub metallic_roughness_texture: Option<wgpu::Texture>,
    pub normal_texture: Option<wgpu::Texture>,
    pub bind_group: Option<wgpu::BindGroup>,
}

impl Material {
    pub fn new(
        name: String, 
        albedo: [f32; 3], 
        metallic: f32, 
        roughness: f32
    ) -> Self {
        Self {
            name,
            albedo,
            metallic,
            roughness,
            albedo_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            bind_group: None,
        }
    }

    pub fn default() -> Self {
        Self::new(
            "Default".to_string(),
            [0.8, 0.8, 0.8],
            0.0,
            0.5,
        )
    }
}