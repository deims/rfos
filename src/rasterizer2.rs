use std::thread;
use std::sync::{Mutex, RwLock, Condvar, Arc};
use image::{Rgba, ImageResult};
use crossbeam_channel::{bounded, Receiver, RecvError};

use crate::math::*;
use crate::ring_buffer::*;
use crate::texture::*;
use crate::scene::*;
use crate::render::*;

pub struct BlockingQueue<T> {
    queue: Mutex<RingBuffer<T>>,
    cond: Condvar
}

impl<T: Copy> BlockingQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Mutex::new(RingBuffer::<T>::new(capacity)),
            cond: Condvar::new()
        }
    }

    pub fn enqueue(&self, value: T) {
        let mut q = self.queue.lock().unwrap();
        while q.is_full() {
            q = self.cond.wait(q).unwrap();
        }
        _ = q.push(value).unwrap();
        self.cond.notify_one();
    }

    pub fn dequeue(&self) -> T {
        let mut q = self.queue.lock().unwrap();
        while q.is_empty() {
            q = self.cond.wait(q).unwrap();
        }
        let ret = q.pop().unwrap();
        self.cond.notify_one();
        ret
    }

    pub fn len(&self) -> usize {
        self.queue.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.lock().unwrap().is_empty()
    }
}

// struct FrameBuffer {
//    pub color_buffer: Texture,
//    pub zbuffer: DepthBuffer,
// }
//
// impl FrameBuffer {
//     pub fn new(width: u32, height: u32) -> Self {
//         FrameBuffer {
//             color_buffer: Texture::new(width, height),
//             zbuffer: DepthBuffer::new(width, height)
//         }
//     }
//
//     pub fn is_within(&self, x: i32, y: i32, z: f32) -> bool {
//         self.color_buffer.is_within(x, y) && z < self.zbuffer.get(x, y)
//     }
//
//     pub fn set(&mut self, x: i32, y: i32, z: f32, color: Rgba<u8>) {
//         self.color_buffer.set_pixel(x as u32, y as u32, color);
//         self.zbuffer.set(x, y, z);
//     }
// }

pub struct Rasterizer2<'a> {
    scene: &'a Scene,
    config: &'a RenderConfig,
    color_buffer: Arc<RwLock<Texture>>,
    zbuffer: Arc<RwLock<DepthBuffer>>,
    camera: Option<&'a Camera>,
    model: Option<&'a Model>,
    mesh: Option<&'a Mesh>,
    material: Option<&'a Material>,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    view_proj_matrix: Mat4,
    normal_matrix: Mat4,
    model_view_matrix: Mat4,
    model_view_proj_matrix: Mat4,
    color_buffer_width: f32,
    color_buffer_height: f32,
    // primitive_queue: Arc<BlockingQueue<RasterPrimitive>>,
}

impl<'a> Rasterizer2<'a> {
    pub fn new(scene: &'a Scene, config: &'a RenderConfig) -> Self {
        let (w, h) = (config.image_width, config.image_height);
        Rasterizer2 {
            scene,
            config,
            color_buffer: Arc::new(RwLock::new(Texture::new(w, h))),
            zbuffer: Arc::new(RwLock::new(DepthBuffer::new(w, h))),
            camera: None,
            model: None,
            mesh: None,
            material: None,
            view_matrix: Mat4::new(),
            proj_matrix: Mat4::new(),
            view_proj_matrix: Mat4::new(),
            normal_matrix: Mat4::new(),
            model_view_matrix: Mat4::new(),
            model_view_proj_matrix: Mat4::new(),
            color_buffer_width: config.image_width as f32,
            color_buffer_height: config.image_height as f32,
            // primitive_queue: Arc::new(BlockingQueue::new(PRIMITIVE_QUEUE_CAPACITY)),
        }
    }

    fn frag_pixel_pos(&self, screen_pos: Vec4) -> Vec2 {
        let color_buffer = self.color_buffer.read().unwrap();
        let tmin = color_buffer.urange[0];
        let tmax = color_buffer.urange[1];
        vec2![
            lerp(0.0, self.color_buffer_width, tmin, tmax, screen_pos[0]),
            lerp(self.color_buffer_height, 0.0, tmin, tmax, screen_pos[1])
        ]
    }

    pub fn render_frame(&mut self, cam: &'a Camera) -> ImageResult<()> {
        self.camera = Some(cam);
        self.view_matrix = cam.view_matrix();
        self.proj_matrix = cam.perspective_matrix();
        self.view_proj_matrix =  self.proj_matrix * self.view_matrix;
       
        let mut color_buffer = self.color_buffer.write().unwrap();
        color_buffer.urange = vec2![-1.0, 1.0];
        color_buffer.vrange = vec2![-1.0, 1.0];
        color_buffer.fill(self.config.background_color);
        let mut zbuffer = self.zbuffer.write().unwrap();
        zbuffer.depth_range = vec2![0.0, 1.0];
        let maxdepth = zbuffer.depth_range[1];
        zbuffer.fill(maxdepth);
        drop(color_buffer);
        drop(zbuffer);

        let worker_count: usize = 2;
        let mut rasterizer_handles = Vec::<thread::JoinHandle<()>>::new();
        const PRIMITIVE_QUEUE_CAPACITY: usize = 1024;
        let (prim_sender, prim_receiver) = bounded(PRIMITIVE_QUEUE_CAPACITY);
        for _i in 0..worker_count {
            let color_buffer = self.color_buffer.clone();
            let zbuffer = self.zbuffer.clone();
            let worker_receiver = prim_receiver.clone();
            rasterizer_handles.push(thread::spawn(move || {
                rasterizer_worker(color_buffer, zbuffer, worker_receiver); 
            }));
        }

        let mut fragtri = FragmentTriangle::new();
        for model in &self.scene.models {
            self.model = Some(model);
            self.material = Some(self.scene.get_material(model.material_index));
            self.mesh = Some(self.scene.get_mesh(model.mesh_index));
            self.normal_matrix = Mat4::transpose(Mat4::inverse(model.model_matrix));
            self.model_view_matrix = self.view_matrix * model.model_matrix;
            self.model_view_proj_matrix = self.proj_matrix * self.model_view_matrix;
            let mesh = self.mesh.unwrap();
            let material = self.material.unwrap();
            if self.config.render_mode == RenderMode::Points {
                for vertex in &mesh.vertices {
                    let world_pos = Vec3::transform_point(*vertex, model.model_matrix);
                    let mut screen_pos = self.model_view_proj_matrix * vertex.to_point();
                    screen_pos /= screen_pos[3];
                    let fpixpos = self.frag_pixel_pos(screen_pos);
                    _ = prim_sender.send(RasterPrimitive::Point(Fragment {
                        world_pos,
                        screen_pos,
                        normal: Vec3::new(),
                        uv: Vec3::new(),
                        pixel_pos: fpixpos,
                        color: rgba_to_vec3(self.config.wireframe_color)
                    })).unwrap();
                }
            } else {
                for face_index in 0..mesh.faces.len() {
                    let face = &mesh.faces[face_index];
                    let mut face_normal = mesh.face_normals[face_index];
                    face_normal = Vec3::transform_dir(face_normal, self.normal_matrix);
                    if self.config.backface_culling {
                        let mut v = mesh.vertices[face[0].vertex];
                        v = Vec3::transform_point(v, model.model_matrix);
                        if Vec3::dot(v - cam.pos, face_normal) >= 0.0 {
                            continue;
                        }
                    }

                    for i in 0..3 {
                        let frag = &mut fragtri.vertices[i];
                        let vertex = mesh.vertices[face[i].vertex];
                        let uv = mesh.vertex_uvs[face[i].uv];
                        frag.world_pos = Vec3::transform_point(vertex, model.model_matrix);
                        frag.screen_pos = self.model_view_matrix * vertex.to_point();
                        frag.uv = vec3![uv[0], uv[1], 1.0]/frag.screen_pos[2];
                        if self.config.shading_model == ShadingModel::Flat {
                            frag.normal = face_normal;
                        } else {
                            let vnormal = mesh.vertex_normals[face[i].normal];
                            frag.normal = Vec3::transform_dir(vnormal, self.normal_matrix);
                        }
                    }

                    const ONE_THIRD: f32 = 1.0/3.0;
                    fragtri.centroid.world_pos = Vec3::new();
                    fragtri.centroid.normal = Vec3::new();
                    fragtri.centroid.uv = Vec3::new();
                    fragtri.centroid.screen_pos = Vec4::new();
                    for i in 0..3 {
                        let mut frag = fragtri.vertices[i];
                        frag.screen_pos = self.proj_matrix * frag.screen_pos;
                        frag.screen_pos /= frag.screen_pos[3];
                        frag.pixel_pos = self.frag_pixel_pos(frag.screen_pos);
                        if self.config.shading_model == ShadingModel::None {
                            frag.color = material.diffuse_color;
                        } else {
                            frag.color = self.phong_shade(&frag);
                        }
                        fragtri.centroid.world_pos += frag.world_pos;
                        fragtri.centroid.uv += frag.uv;
                        fragtri.centroid.screen_pos += frag.screen_pos;
                        fragtri.vertices[i] = frag;
                    }
                    fragtri.centroid.world_pos *= ONE_THIRD;
                    fragtri.centroid.normal = face_normal;
                    fragtri.centroid.uv *= ONE_THIRD;
                    fragtri.centroid.screen_pos *= ONE_THIRD;
                    fragtri.centroid.pixel_pos = self.frag_pixel_pos(fragtri.centroid.screen_pos);
                    if self.config.shading_model == ShadingModel::None {
                        fragtri.centroid.color = material.diffuse_color;
                    } else {
                        fragtri.centroid.color = self.phong_shade(&fragtri.centroid);
                    }

                    if self.config.render_mode == RenderMode::Filled {
                        fragtri.winding = fragtri.compute_winding();
                        fragtri.area = fragtri.compute_area();
                        _ = prim_sender.send(RasterPrimitive::Triangle(fragtri)).unwrap();
                    } else {
                        for i in 0..3 {
                            _ = prim_sender.send(RasterPrimitive::Line(
                                fragtri.vertices[i],
                                fragtri.vertices[(i+1) % 3]
                            )).unwrap();
                        }
                    }

                    if self.config.show_face_normals {
                        let color = rgba_to_vec3(self.config.face_normal_color);
                        let mut end = Fragment::new();
                        end.world_pos = self.config.face_normal_length*fragtri.centroid.normal + fragtri.centroid.world_pos;
                        end.screen_pos = self.view_proj_matrix * end.world_pos.to_point();
                        end.screen_pos /= end.screen_pos[3];
                        end.pixel_pos = self.frag_pixel_pos(end.screen_pos);
                        end.color = color;
                        let mut start = fragtri.centroid;
                        start.color = color;
                        _ = prim_sender.send(RasterPrimitive::Line(start, end)).unwrap();
                    }

                    if self.config.show_vertex_normals {
                        let nlen = self.config.vertex_normal_length;
                        let color = rgba_to_vec3(self.config.vertex_normal_color);
                        let mut end = Fragment::new();
                        for frag in &fragtri.vertices {
                            end.world_pos = nlen*frag.normal + frag.world_pos;
                            end.screen_pos = self.view_proj_matrix * end.world_pos.to_point();
                            end.screen_pos /= end.screen_pos[3];
                            end.pixel_pos = self.frag_pixel_pos(end.screen_pos);
                            end.color = color;
                            let mut start = *frag;
                            start.color = color;
                            _ = prim_sender.send(RasterPrimitive::Line(start, end)).unwrap();
                        }
                    }
                }
            }

            if self.config.show_bounding_boxes {
                let corners = model.bounding_box.corners_vec4();
                let mut bbox_frags = [Fragment::new(); 8];
                let color = rgba_to_vec3(self.config.bounding_box_color);
                for i in 0..8 {
                    bbox_frags[i].screen_pos = self.view_proj_matrix * corners[i];
                    bbox_frags[i].screen_pos /= bbox_frags[i].screen_pos[3];
                    bbox_frags[i].pixel_pos = self.frag_pixel_pos(bbox_frags[i].screen_pos);
                    bbox_frags[i].color = color;
                }
                for i in 0..4 {
                    let j = (i+1) % 4;
                    _ = prim_sender.send(RasterPrimitive::Line(bbox_frags[i], bbox_frags[j])).unwrap();
                    _ = prim_sender.send(RasterPrimitive::Line(bbox_frags[i+4], bbox_frags[j+4])).unwrap();
                    _ = prim_sender.send(RasterPrimitive::Line(bbox_frags[i], bbox_frags[i+4])).unwrap();
                }
            }
        }
        drop(prim_sender);
        for handle in rasterizer_handles {
            _ = handle.join().unwrap();
        }
        let cbuf = self.color_buffer.write().unwrap();
        cbuf.save(&self.config.output_file)
    }

    fn phong_shade(&self, _frag: &Fragment) -> Vec3 {
        let material = self.material.unwrap();
        material.diffuse_color
    }
}

fn rasterizer_worker(
    color_buffer: Arc<RwLock<Texture>>,
    zbuffer: Arc<RwLock<DepthBuffer>>,
    receiver: Receiver<RasterPrimitive>)
{
    let mut points: u64 = 0;
    let mut lines: u64 = 0;
    let mut tris: u64 = 0;
    loop {
        match receiver.recv() {
            Err(RecvError) => break,
            Ok(RasterPrimitive::Point(frag)) => {
                points += 1;
                let pos = frag.pixel_pos.elems.map(|x| f32::round(x) as i32);
                let cbuf = color_buffer.clone();
                let mut cb = cbuf.write().unwrap();
                let mut zb = zbuffer.write().unwrap();
                if cb.is_within(pos[0], pos[1]) && frag.screen_pos[2] < zb.get(pos[0], pos[1]) {
                    cb.set_pixel(pos[0] as u32, pos[1] as u32, rgba_from_vec3(frag.color));
                    zb.set(pos[0], pos[1], frag.screen_pos[2]);
                }
            },
            Ok(RasterPrimitive::Line(start, end)) => {
                lines += 1;
                let startpos = start.pixel_pos.elems.map(|x| f32::round(x) as i32);
                let endpos = end.pixel_pos.elems.map(|x| f32::round(x) as i32);
                let color = rgba_from_vec3(start.color);
                let cb = color_buffer.clone();
                let zb = zbuffer.clone();
                if i32::abs(endpos[1] - startpos[1]) < i32::abs(endpos[0] - startpos[0]) {
                    if startpos[0] > endpos[0] {
                        render_line_low(cb, zb, &end, endpos, &start, startpos, color);
                    } else {
                        render_line_low(cb, zb, &start, startpos, &end, endpos, color);
                    }
                } else {
                    if startpos[1] > endpos[1] {
                        render_line_high(cb, zb, &end, endpos, &start, startpos, color);
                    } else {
                        render_line_high(cb, zb, &start, startpos, &end, endpos, color);
                    }
                }
            },
            Ok(RasterPrimitive::Triangle(fragtri)) => {
                tris += 1;
                let (min, max) = fragtri.pixel_extents();
                let mut cb = color_buffer.write().unwrap();
                let mut zb = zbuffer.write().unwrap();
                for y in min[1]..=max[1] {
                    for x in min[0]..=max[0] {
                        if !cb.is_within(x, y) { continue; }
                        let p = vec2![x as f32 + 0.5, y as f32 + 0.5];
                        let mut bc = vec3! [
                            fragtri.edge_function(1, 2, p[0], p[1]),
                            fragtri.edge_function(2, 0, p[0], p[1]),
                            fragtri.edge_function(0, 1, p[0], p[1])
                        ];
                        if bc[0] >= 0.0 && bc[1] >= 0.0 && bc[2] >= 0.0 {
                            bc /= fragtri.area;
                            let frag = fragtri.weighted_sum(bc);
                            if frag.screen_pos[2] < zb.get(x, y) {
                                let color = vec3![0.5, 0.5, 0.5];
                                cb.set_pixel(x as u32, y as u32, rgba_from_vec3(color));
                                zb.set(x, y, frag.screen_pos[2]);
                            }
                        }
                    }
                }
            }
        }
    }
    println!("points: {}\tlines: {}\ttris: {}", points, lines, tris);
}

fn render_line_low(
    color_buffer: Arc<RwLock<Texture>>,
    zbuffer: Arc<RwLock<DepthBuffer>>,
    start: &Fragment, startpos: [i32; 2],
    end: &Fragment, endpos: [i32; 2],
    color: Rgba<u8>)
{
    let dx = endpos[0] - startpos[0];
    let dz = (end.screen_pos[2] - start.screen_pos[2]) / (dx as f32);
    let mut dy = endpos[1] - startpos[1];
    let mut yi = 1i32;
    if dy < 0 {
        yi = -1;
        dy = -dy;
    }
    let mut d = 2*dy - dx;
    let mut y = startpos[1];
    let mut z = start.screen_pos[2];
    let mut color_buffer = color_buffer.write().unwrap();
    let mut zbuffer = zbuffer.write().unwrap();
    for x in startpos[0]..=endpos[0] {
        if color_buffer.is_within(x, y) && z < zbuffer.get(x, y) {
            color_buffer.set_pixel(x as u32, y as u32, color);
            zbuffer.set(x, y, z);
        }
        if d > 0 {
            y += yi;
            d -= 2*dx;
        }
        d += 2*dy;
        z += dz;
    }
}

fn render_line_high(
    color_buffer: Arc<RwLock<Texture>>,
    zbuffer: Arc<RwLock<DepthBuffer>>,
    start: &Fragment, startpos: [i32; 2],
    end: &Fragment, endpos: [i32; 2],
    color: Rgba<u8>)
{
    let dy = endpos[1] - startpos[1];
    let dz = (end.screen_pos[2] - start.screen_pos[2]) / (dy as f32);
    let mut dx = endpos[0] - startpos[0];
    let mut xi = 1i32;
    if dx < 0 {
        xi = -1;
        dx = -dx;
    }
    let mut d = 2*dx - dy;
    let mut x = startpos[0];
    let mut z = start.screen_pos[2];
    let mut color_buffer = color_buffer.write().unwrap();
    let mut zbuffer = zbuffer.write().unwrap();
    for y in startpos[1]..=endpos[1] {
        if color_buffer.is_within(x, y) && z < zbuffer.get(x, y) {
            color_buffer.set_pixel(x as u32, y as u32, color);
            zbuffer.set(x, y, z);
        }
        if d > 0 {
            x += xi;
            d -= 2*dy;
        }
        d += 2*dx;
        z += dz;
    }
}
