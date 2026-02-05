pub mod light;
pub mod static_mesh;
pub mod material;
pub mod scene;

// Re-export commonly used types for convenience
pub use light::{DirectionalLight, PointLight, Light, LightType};
pub use static_mesh::{Mesh, Vertex};
pub use material::Material;
pub use scene::Scene;