use super::{Mesh, Material, Light};
use cgmath;

#[derive(Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub lights: Vec<Light>,
    pub camera_position: cgmath::Point3<f32>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            materials: Vec::new(),
            lights: Vec::new(),
            camera_position: cgmath::Point3::new(0.0, 0.0, 3.0),
        }
    }

    pub fn add_mesh(&mut self, mesh: Mesh) {
        self.meshes.push(mesh);
    }

    pub fn add_material(&mut self, material: Material) -> usize {
        let index = self.materials.len();
        self.materials.push(material);
        index
    }

    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }
}