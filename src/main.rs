use core::f32;
use std::sync::Arc;
use std::env;

pub mod math;
pub mod texture;
pub mod scene;
pub mod render;
pub mod ring_buffer;
pub mod octree;
pub mod rasterizer;
pub mod raytracer;

use crate::math::*;
use crate::texture::*;
use crate::scene::*;
use crate::render::*;
use crate::rasterizer::*;
use crate::raytracer::*;

fn main() {
    stderrlog::new().module(module_path!()).verbosity(2).init().unwrap();

    let config = RenderConfig {
        output_file: String::from("render.png"),
        image_width: 1920,
        image_height: 1200,
        render_mode: RenderMode::Filled,
        shading_model: ShadingModel::Phong,
        background_color: rgb(24, 24, 24),
        rasterizer_config: Some(RasterizerConfig {
            vertex_processors: 1,
            fragment_processors: 1,
            backface_culling: true,
            show_face_normals: false,
            show_vertex_normals: false,
            show_bounding_boxes: false,
            show_wireframe: false,
            face_normal_length: 0.2,
            vertex_normal_length: 0.2,
            face_normal_color: RED,
            vertex_normal_color: GREEN,
            bounding_box_color: CYAN,
            wireframe_color: rgb(200, 200, 200),
        }),
        raytracer_config: Some(RaytracerConfig {
            worker_count: 24,
            octree_leaf_capacity: 50,
            octree_min_node_size: 1e-3,
            render_octree: false
        })
    };

    log::info!("creating scene");
    let mut scene = Scene::new();
    let material_index = scene.create_material(Material::default());
    // let mesh_index = scene.create_plane_mesh(2.0, 2.0, 1, 1);
    // let mesh_index = scene.create_box_mesh(1.0, 1.0, 1.0);
    // let mesh_index = scene.load_wavefront_obj("data/icosahedron.obj").unwrap();
    let base_index = scene.create_box_mesh(8.0, 0.1, 8.0);
    let box_index = scene.create_box_mesh(4.0, 0.5, 0.5);
    let torus_index = scene.load_wavefront_obj("data/torus.obj").unwrap();

    let angle = 0.3 * std::f32::consts::PI;
    let axis = vec3![0.0, 1.0, 0.0];
    let rot = Mat4::rotation(angle, axis);
    let base_mm = rot * Mat4::translation(vec3![0.0, -0.05, 0.0]);
    let box1_mm = rot * Mat4::translation(vec3![0.0, 0.24, -1.1]);
    let box2_mm = rot * Mat4::translation(vec3![0.0, 0.25, 1.1]);
    let torus1_mm = Mat4::translation(vec3![0.0, 0.85, 0.0]) * rot;
    let torus2_mm = Mat4::translation(vec3![0.0, 1.85, 0.0]) * rot;

    // _ = scene.create_model(mesh_index, material_index, rot);
    _ = scene.create_model(base_index, material_index, base_mm);
    _ = scene.create_model(box_index, material_index, box1_mm);
    _ = scene.create_model(box_index, material_index, box2_mm);
    _ = scene.create_model(torus_index, material_index, torus1_mm);
    _ = scene.create_model(torus_index, material_index, torus2_mm);

    _ = scene.create_point_light(PointLight {
        index: INVALID_INDEX,
        pos: vec3![-2.0, 3.0, 2.0],
        color: vec3![1.0, 1.0, 1.0],
        attenuation: vec3![1.0, 0.06, 0.003]
    });
   
    let w = config.image_width as f32;
    let h = config.image_height as f32;
    let camera_index = scene.create_camera(Camera {
        index: INVALID_INDEX,
        pos: vec3![3.0, 3.0, 0.0],
        look: Vec3::normalize(vec3![-1.0, -1.0, 0.0]),
        up: Vec3::normalize(vec3![-1.0, 1.0, 0.0]),
        znear: 0.1,
        zfar: 100.0,
        horizontal_fov: std::f32::consts::PI * 0.5,
        aspect_ratio: w/h
    });

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        log::error!("missing renderer type, give it as argument");
        std::process::exit(1);
    }

    log::info!("rendering scene");
    let scene = Arc::new(scene);
    if args[1] == "ras" {
        log::info!("using rasterizing renderer");
        let mut rend = Rasterizer::new(scene.clone(), config.clone());
        let frame = rend.render_frame(camera_index);
        _ = frame.save(&config.output_file).unwrap();
    } else if args[1] == "ray" {
        log::info!("using raytracing renderer");
        let mut rend = Raytracer::new(scene.clone(), config.clone());
        let rtconfig = config.raytracer_config.unwrap();
        if rtconfig.render_octree {
            log::info!("rendering octree to octree.ong");
            _ = rend.octree.render_to_file(config.clone(), camera_index, "octree.png");
        }
        let frame = rend.render_frame(camera_index);
        let frame_lock = frame.read().unwrap();
        _ = frame_lock.save(&config.output_file).unwrap();
    } else {
        log::error!("invalid renderer type '{}'", args[1]);
        std::process::exit(1);
    }
}
