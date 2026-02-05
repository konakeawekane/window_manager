struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_position: vec3<f32>,
    _padding1: f32,
    num_directional_lights: u32,
    num_point_lights: u32,
    _padding2: vec2<f32>,
}

struct DirectionalLight {
    direction: vec3<f32>,
    _padding: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct PointLight {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}

struct MaterialProperties {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<uniform> directional_lights: array<DirectionalLight, 4>;

@group(0) @binding(2)
var<uniform> point_lights: array<PointLight, 8>;

@group(1) @binding(0)
var albedo_texture: texture_2d<f32>;

@group(1) @binding(1)
var albedo_sampler: sampler;

@group(1) @binding(2)
var metallic_roughness_texture: texture_2d<f32>;

@group(1) @binding(3)
var metallic_roughness_sampler: sampler;

@group(1) @binding(4)
var normal_texture: texture_2d<f32>;

@group(1) @binding(5)
var normal_sampler: sampler;

@group(1) @binding(6)
var<uniform> material_props: MaterialProperties;

struct FragmentInput {
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
    return a2 / (3.14159265359 * denom * denom);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = (roughness + 1.0);
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

fn calculate_lighting(
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    light_intensity: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    if n_dot_l <= 0.0 {
        return vec3<f32>(0.0);
    }
    
    let half_dir = normalize(view_dir + light_dir);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_h = max(dot(normal, half_dir), 0.0);
    
    // F0 for dielectrics is 0.04, for metals use albedo
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    
    // Cook-Torrance BRDF
    let D = distribution_ggx(n_dot_h, roughness);
    let G = geometry_smith(n_dot_v, n_dot_l, roughness);
    let F = fresnel_schlick(max(dot(half_dir, view_dir), 0.0), f0);
    
    let numerator = D * G * F;
    let denominator = 4.0 * n_dot_v * n_dot_l + 0.001;
    let specular = numerator / denominator;
    
    // Energy conservation
    let kS = F;
    let kD = (1.0 - kS) * (1.0 - metallic);
    
    // Diffuse and specular
    let diffuse = kD * albedo / 3.14159265359;
    let final_color = (diffuse + specular) * light_color * light_intensity * n_dot_l;
    
    return final_color;
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Sample textures
    let albedo_sample = textureSample(albedo_texture, albedo_sampler, input.tex_coords).rgb;
    let metallic_roughness_sample = textureSample(metallic_roughness_texture, metallic_roughness_sampler, input.tex_coords);
    
    // Combine texture values with material properties
    let albedo = albedo_sample * material_props.albedo;
    let metallic = metallic_roughness_sample.b * material_props.metallic;
    let roughness = metallic_roughness_sample.g * material_props.roughness;
    
    // Sample and apply normal map
    var normal = normalize(input.normal);
    let normal_sample = textureSample(normal_texture, normal_sampler, input.tex_coords).rgb;
    // Convert from [0,1] to [-1,1] and apply normal mapping
    let normal_map = normalize(normal_sample * 2.0 - 1.0);
    // Simple tangent space normal mapping (simplified - assumes TBN is identity)
    // For full implementation, you'd need tangent and bitangent vectors
    normal = normalize(normal + normal_map * 0.1);
    
    // View direction
    let view_dir = normalize(uniforms.camera_position - input.world_position);
    
    var final_color = vec3<f32>(0.0);
    
    // Accumulate directional lights
    for (var i: u32 = 0u; i < uniforms.num_directional_lights; i++) {
        let light = directional_lights[i];
        let light_dir = normalize(-light.direction);
        final_color += calculate_lighting(
            albedo,
            metallic,
            roughness,
            normal,
            view_dir,
            light_dir,
            light.color,
            light.intensity
        );
    }
    
    // Accumulate point lights
    for (var i: u32 = 0u; i < uniforms.num_point_lights; i++) {
        let light = point_lights[i];
        let light_dir = normalize(light.position - input.world_position);
        let distance = length(light.position - input.world_position);
        let attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
        let effective_intensity = light.intensity * attenuation;
        
        final_color += calculate_lighting(
            albedo,
            metallic,
            roughness,
            normal,
            view_dir,
            light_dir,
            light.color,
            effective_intensity
        );
    }
    
    // Add ambient lighting
    let ambient = albedo * 0.0;
    final_color += ambient;
    
    // Tone mapping and gamma correction
    final_color = final_color / (final_color + vec3<f32>(1.0));
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));
    
    return vec4<f32>(final_color, 1.0);
}