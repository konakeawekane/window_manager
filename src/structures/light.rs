use cgmath;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    pub _padding: f32,  // Padding for alignment
    pub color: [f32; 3],
    pub intensity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointLight {
    pub position: [f32; 3],
    pub radius: f32,
    pub color: [f32; 3],
    pub intensity: f32,
}

#[derive(Debug, Clone)]
pub struct Light {
    pub name: String,
    pub light_type: LightType,
}

#[derive(Debug, Clone)]
pub enum LightType {
    Directional {
        direction: cgmath::Vector3<f32>,
        color: cgmath::Vector3<f32>,
        intensity: f32,
    },
    Point {
        position: cgmath::Point3<f32>,
        color: cgmath::Vector3<f32>,
        intensity: f32,
        radius: f32,
    },
}