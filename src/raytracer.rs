use std::sync::{Arc, RwLock, Mutex};
use std::thread;

use crate::ring_buffer::RingBuffer;
use crate::math::*;
use crate::texture::*;
use crate::scene::*;
use crate::octree::*;
use crate::render::*;

#[derive(Clone, Copy)]
pub struct Tile {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    top_left: Vec3,
    right_step: Vec3,
    down_step: Vec3
}

pub struct Raytracer {
    pub scene: Arc<Scene>,
    pub octree: Arc<Octree>,
    pub config: RenderConfig,
    pub frame: Arc<RwLock<Texture>>,
    pub camera_index: usize,
    pub tile_queue: Arc<Mutex<RingBuffer<Tile>>>,
    pub tile_width: i32,
    pub tile_height: i32,
}

fn possible_tile_sizes(n: i32) -> Vec<i32> {
    let limit = f64::ceil(f64::sqrt(n as f64)) as i32;
    let mut sizes = Vec::<i32>::new();
    for i in 2..=limit {
        if n % i == 0 {
            let size = n/i;
            if size > 0 {
                sizes.push(n/i);
            }
        }
    }
    sizes
}

fn compute_tile_size(image_width: i32, image_height: i32) -> (i32, i32) {
    let wsizes = possible_tile_sizes(image_width);
    let hsizes = possible_tile_sizes(image_height);
    let mut min_diff = i32::MAX;
    let mut tile_width = i32::MAX;
    let mut tile_height = i32::MAX;
    for tw in &wsizes {
        for th in &hsizes {
            let diff = i32::abs(*tw - *th);
            if diff <= min_diff && (*tw < tile_width || *th < tile_width) {
                min_diff = diff;
                tile_width = *tw;
                tile_height = *th;
            }
        }
    }
    (tile_width, tile_height)
}

impl Raytracer {
    pub fn new(scene: Arc<Scene>, config: RenderConfig) -> Self {
        if config.raytracer_config.is_none() {
            log::error!("raytracer config is missing, exiting");
            std::process::exit(1);
        }
        let (imgw, imgh) = (config.image_width as i32, config.image_height as i32);
        let (tile_width, tile_height) = compute_tile_size(imgw, imgh);
        let tile_cols = (imgw / tile_width) as usize;
        let tile_rows = (imgh / tile_height) as usize;
        let tile_count = tile_cols * tile_rows;
        let rtconfig = config.raytracer_config.unwrap();
        let leafcap = rtconfig.octree_leaf_capacity;
        let minsize = rtconfig.octree_min_node_size;
        let octree = Arc::new(Octree::new(scene.clone(), leafcap, minsize));
        let frame = Arc::new(RwLock::new(Texture::new(config.image_width, config.image_height)));
        let tile_queue = Arc::new(Mutex::new(RingBuffer::<Tile>::new(tile_count)));
        Raytracer {
            scene: scene.clone(),
            octree,
            config,
            frame,
            camera_index: INVALID_INDEX,
            tile_queue,
            tile_width,
            tile_height,
        }
    }

    pub fn render_frame(&mut self, camera_index: usize) -> &Arc<RwLock<Texture>> {
        self.camera_index = camera_index;
        let cam = self.scene.get_camera(camera_index);
        let mut frame_lock = self.frame.write().unwrap();
        frame_lock.fill(self.config.background_color);
        drop(frame_lock);

        // image rect
        let left = Vec3::normalize(Vec3::cross(cam.look, cam.up));
        let center = cam.pos + cam.look*cam.znear;
        let left_offset = cam.znear * f32::tan(0.5*cam.horizontal_fov);
        let up_offset = cam.znear * f32::tan(0.5*cam.vertical_fov());
        let right_step = -left * (2.0 * left_offset / (self.config.image_width as f32));
        let down_step = -cam.up * (2.0 * up_offset / (self.config.image_height as f32));
        let top_left = center + (left*left_offset + cam.up*up_offset);
        let tile_right_step = right_step * (self.tile_width as f32);
        let tile_down_step = down_step * (self.tile_height as f32);
        
        let mut row_start = top_left;
        let mut tile = Tile {
            x: 0, y: 0,
            width: self.tile_width, height: self.tile_height,
            top_left,
            right_step,
            down_step
        };

        let (w, h) = (self.config.image_width as usize, self.config.image_height as usize);
        let (tw, th) = (self.tile_width as usize, self.tile_height as usize);
        let mut tqlock = self.tile_queue.lock().unwrap();
        for y in (0..h).step_by(th) {
            tile.y = y as i32;
            tile.top_left = row_start;
            for x in (0..w).step_by(tw) {
                tile.x = x as i32;
                _ = tqlock.push(tile).unwrap();
                tile.top_left += tile_right_step;
            }
            row_start += tile_down_step;
        }
        drop(tqlock);

        if self.config.raytracer_config.is_none() {
            log::error!("raytracer config is missing");
            return &self.frame;
        }
        let worker_count = self.config.raytracer_config.unwrap().worker_count;
        let mut worker_handles = Vec::<thread::JoinHandle<()>>::new();
        for i in 0..worker_count {
            let id = i;
            let tile_queue = self.tile_queue.clone();
            let camera = *self.scene.get_camera(self.camera_index);
            let scene = self.scene.clone();
            let octree = self.octree.clone();
            let shading_model = self.config.shading_model;
            let frame = self.frame.clone();
            let thread_handle = thread::spawn(move || {
                log::trace!("starting worker {}", id);
                let mut rendered_tiles = 0usize;
                let mut ray = Ray{origin: camera.pos, dir: vec3![0.0, 0.0, 0.0]};
                loop {
                    let mut qlock = tile_queue.lock().unwrap();
                    if qlock.is_empty() {
                        drop(qlock);
                        break;
                    }
                    let tile = qlock.pop().unwrap();
                    drop(qlock);
                    let mut row_start = tile.top_left;
                    for y in tile.y..tile.y+tile.height {
                        let mut imgrect_point = row_start;
                        for x in tile.x..tile.x+tile.width {
                            imgrect_point += tile.right_step;
                            ray.dir = Vec3::normalize(imgrect_point - camera.pos);
                            if let Some(qres) = octree.query(&ray) {
                                let model = scene.get_model(qres.model_index);
                                let material = scene.get_material(model.material_index);
                                let trifrags = triangle_fragments(
                                    scene.clone(),
                                    shading_model,
                                    &camera,
                                    model,
                                    qres.face_index
                                );
                                let mut frag = Fragment::scaled(&trifrags[0], qres.barycentric[0]);
                                for i in 1..3 {
                                    let tf = Fragment::scaled(&trifrags[i], qres.barycentric[i]);
                                    frag.add(&tf);
                                }

                                let color = if shading_model == ShadingModel::None {
                                    material.diffuse_color
                                } else {
                                    phong_shade(scene.clone(), &frag, material, &camera)
                                };

                                let mut frame_lock = frame.write().unwrap();
                                frame_lock.set_pixel(x as u32, y as u32, rgba_from_vec3(color));
                                drop(frame_lock);
                            }
                        }
                        row_start += tile.down_step;
                    }
                    rendered_tiles += 1;
                }
                log::trace!("worker {} rendered {} tiles", id, rendered_tiles);
            });
            worker_handles.push(thread_handle);
        }
        for handle in worker_handles { _ = handle.join() }
        &self.frame
    }
}

fn triangle_fragments(
    scene: Arc<Scene>,
    shading_model: ShadingModel,
    cam: &Camera,
    model: &Model,
    face_index: usize)
    -> [Fragment; 3]
{
    let mesh = scene.get_mesh(model.mesh_index);
    let material = scene.get_material(model.material_index);
    let view_matrix = cam.view_matrix();
    let proj_matrix = cam.perspective_matrix();
    let normal_matrix = Mat4::transpose(Mat4::inverse(model.model_matrix));
    let model_view_matrix = view_matrix * model.model_matrix;
    // let model_view_proj_matrix = proj_matrix * model_view_matrix;
    let face = &mesh.faces[face_index];
    let mut frags = [Fragment::new(); 3];
    for i in 0..3 {
        let frag = &mut frags[i];
        let vertex = mesh.vertices[face[i].vertex];
        let uv = mesh.vertex_uvs[face[i].uv];
        frag.world_pos = Vec3::transform_point(vertex, model.model_matrix);
        frag.screen_pos = model_view_matrix * vertex.to_point();
        frag.uv = vec3![uv[0], uv[1], 1.0]/frag.screen_pos[2];
        frag.screen_pos = proj_matrix * frag.screen_pos;
        frag.screen_pos /= frag.screen_pos[3];
        frag.normal = if shading_model == ShadingModel::Flat {
            mesh.face_normals[face_index]
        } else {
            mesh.vertex_normals[face[i].normal]
        };
        frag.normal = Vec3::transform_dir(frag.normal, normal_matrix);
        frag.color = if shading_model == ShadingModel::None {
            material.diffuse_coeff
        } else {
            phong_shade(scene.clone(), frag, material, cam)
        }
    }
    frags
}

fn get_color(scene: Arc<Scene>, map_index: usize, u: f32, v: f32, default: Vec3) -> Vec3 {
    if map_index == INVALID_INDEX { return default; }
    let tex = scene.get_texture(map_index);
    rgba_to_vec3(tex.get_pixel_uv(u, v))
}

fn ambient_color(scene: Arc<Scene>, material: &Material, u: f32, v: f32) -> Vec3 {
    get_color(scene.clone(), material.diffuse_map_index, u, v, vec3![1.0, 1.0, 1.0])
}

fn diffuse_color(scene: Arc<Scene>, material: &Material, u: f32, v: f32) -> Vec3 {
    get_color(scene.clone(), material.diffuse_map_index, u, v, material.diffuse_color)
}

fn phong_shade(
    scene: Arc<Scene>,
    frag: &Fragment,
    material: &Material,
    cam: &Camera) -> Vec3
{
    let ucorr = frag.uv[0]/frag.uv[2];
    let vcorr = frag.uv[1]/frag.uv[2];
    let v = Vec3::normalize(cam.pos - frag.world_pos);
    let ambcolor = ambient_color(scene.clone(), material, ucorr, vcorr);
    let mut intensity = vec3![0.05, 0.05, 0.05]*material.ambient_coeff*ambcolor;
    for light in &scene.point_lights {
        let mut l = light.pos - frag.world_pos;
        let d = Vec3::norm(l);
        l /= d;
        let mut nl = Vec3::dot(frag.normal, l);
        let r = 2.0 * nl * (frag.normal - l);
        let att = f32::min(1.0/(light.attenuation[0] + light.attenuation[1]*d + light.attenuation[2]*d*d), 1.0);
        nl = f32::max(nl, 0.0);
        let rv = f32::powf(f32::max(Vec3::dot(r, v), 0.0), material.specular_exp);
        let diffcolor = diffuse_color(scene.clone(), material, ucorr, vcorr);
        let diff = nl * material.diffuse_coeff * diffcolor;
        let spec = rv * material.specular_coeff * material.specular_color;
        for i in 0..3 {
            intensity[i] += light.color[i] * att * (diff[i] + spec[i]);
        }
    }
    Vec3::clamp(intensity, 0.0, 1.0)
}
