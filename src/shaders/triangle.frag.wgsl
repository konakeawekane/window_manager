struct FragmentInput {
    @location(0) barycentric: vec3<f32>,
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0) vec4<f32> {
    // Visualize barycentric coordinates as RGB colors
    // Each corner corresponds to a primary color (R, G, B)
    // Interior pixels are smoothly interpolated
    return vec4<f32>(in.barycentric, 1.0);
}
