struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    camera_position: vec3<f32>,
    _padding1: f32,
    num_directional_lights: u32,
    num_point_lights: u32,
    _padding2: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let model_space = vec4<f32>(input.position, 1.0);
    let world_space = uniforms.model * model_space;
    let clip_space = uniforms.view_proj * world_space;

    out.clip_position = clip_space;
    out.world_position = world_space.xyz;

    let normal_matrix = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz
    );
    out.normal = normalize(normal_matrix * input.normal);

    out.tex_coords = input.tex_coords;

    return out;
}