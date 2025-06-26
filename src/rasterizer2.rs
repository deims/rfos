use std::thread;
use std::sync::Arc;
use image::Rgba;
use crossbeam_channel::{bounded, Receiver, RecvError};

use crate::math::*;
use crate::texture::*;
use crate::scene::*;
use crate::render::*;

pub struct FragmentProcessor {
    pub scene: Option<Arc<Scene>>,
    pub config: RenderConfig,
    pub color_buffer: Texture,
    pub zbuffer: DepthBuffer,
    pub camera_index: usize,
    pub receiver: Receiver<RasterPrimitive>,
}

impl FragmentProcessor {
    pub fn new(scene: Option<Arc<Scene>>, config: RenderConfig, camera_index: usize,
        receiver: Receiver<RasterPrimitive>) -> Self {
        if scene.is_none() {
            log::warn!("no scene provided for fragment processor; cannot render filled triangled");
        }
        let (w, h) = (config.image_width, config.image_height);
        let color_buffer = Texture::new(w, h);
        let zbuffer = DepthBuffer::new(w, h);
        FragmentProcessor {
            scene,
            config,
            color_buffer,
            zbuffer,
            camera_index,
            receiver
        }
    }

    pub fn init(&mut self) {
        self.color_buffer.urange = vec2![-1.0, 1.0];
        self.color_buffer.vrange = vec2![-1.0, 1.0];
        self.color_buffer.fill(self.config.background_color);
        self.zbuffer.depth_range = vec2![0.0, 1.0];
        self.zbuffer.fill(1.0);
    }

    fn process_point(&mut self, frag: &Fragment) {
        let pos = frag.pixel_pos.elems.map(|x| f32::round(x) as i32);
        let iswithin = self.color_buffer.is_within(pos[0], pos[1]);
        let z = frag.screen_pos[2];
        if iswithin && z < self.zbuffer.get(pos[0], pos[1]) {
            let color = rgba_from_vec3(frag.color);
            self.color_buffer.set_pixel(pos[0] as u32, pos[1] as u32, color);
            self.zbuffer.set(pos[0], pos[1], frag.screen_pos[2]);
        }
    }

    fn process_line(&mut self, start: &Fragment, end: &Fragment) {
        let startpos = start.pixel_pos.elems.map(|x| f32::round(x) as i32);
        let endpos = end.pixel_pos.elems.map(|x| f32::round(x) as i32);
        let color = rgba_from_vec3(start.color);
        if i32::abs(endpos[1] - startpos[1]) < i32::abs(endpos[0] - startpos[0]) {
            if startpos[0] > endpos[0] {
                render_line_low(&mut self.color_buffer, &mut self.zbuffer, &end, endpos, &start,
                    startpos, color);
            } else {
                render_line_low(&mut self.color_buffer, &mut self.zbuffer, &start, startpos,
                    &end, endpos, color);
            }
        } else {
            if startpos[1] > endpos[1] {
                render_line_high(&mut self.color_buffer, &mut self.zbuffer, &end, endpos,
                    &start, startpos, color);
            } else {
                render_line_high(&mut self.color_buffer, &mut self.zbuffer, &start, startpos,
                    &end, endpos, color);
            }
        }
    }

    fn process_tri(&mut self, tri: &FragmentTriangle) {
        let scene = self.scene.clone();
        if scene.is_none() {return;}
        let (min, max) = tri.pixel_extents();
        let scene = scene.unwrap();
        let camera = scene.get_camera(self.camera_index);
        let material = scene.get_material(tri.material_index);
        for y in min[1]..=max[1] {
            for x in min[0]..=max[0] {
                if !self.color_buffer.is_within(x, y) { continue; }
                let p = vec2![x as f32 + 0.5, y as f32 + 0.5];
                let mut bc = vec3! [
                    tri.edge_function(1, 2, p[0], p[1]),
                    tri.edge_function(2, 0, p[0], p[1]),
                    tri.edge_function(0, 1, p[0], p[1])
                ];
                if bc[0] >= 0.0 && bc[1] >= 0.0 && bc[2] >= 0.0 {
                    bc /= tri.area;
                    let frag = tri.weighted_sum(bc);
                    if frag.screen_pos[2] < self.zbuffer.get(x, y) {
                        let color = rgba_from_vec3(phong_shade(scene.clone(), &frag, material,
                            camera));
                        self.color_buffer.set_pixel(x as u32, y as u32, color);
                        self.zbuffer.set(x, y, frag.screen_pos[2]);
                    }
                }
            }
        }
    }

    pub fn run(&mut self) {
        loop {
            match self.receiver.recv() {
                Err(RecvError) => break,
                Ok(RasterPrimitive::Point(frag)) => self.process_point(&frag),
                Ok(RasterPrimitive::Line(start, end)) => self.process_line(&start, &end),
                Ok(RasterPrimitive::Triangle(tri)) => self.process_tri(&tri),
            }
        }
    }
}

pub struct Rasterizer2 {
    scene: Arc<Scene>,
    config: RenderConfig,
    color_buffer: Texture,
    zbuffer: DepthBuffer,
    camera_index: usize,
    model_index: usize,
    mesh_index: usize,
    material_index: usize,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    view_proj_matrix: Mat4,
    normal_matrix: Mat4,
    model_view_matrix: Mat4,
    model_view_proj_matrix: Mat4,
    color_buffer_width: f32,
    color_buffer_height: f32,
}

impl Rasterizer2 {
    pub fn new(scene: Arc<Scene>, config: RenderConfig) -> Self {
        if config.rasterizer_config.is_none() {
            log::error!("rasterizer config is missing, exiting");
            std::process::exit(1);
        }
        let (w, h) = (config.image_width, config.image_height);
        Rasterizer2 {
            scene,
            config: config.clone(),
            color_buffer: Texture::new(w, h),
            zbuffer: DepthBuffer::new(w, h),
            camera_index: INVALID_INDEX,
            model_index: INVALID_INDEX,
            mesh_index: INVALID_INDEX,
            material_index: INVALID_INDEX,
            view_matrix: Mat4::new(),
            proj_matrix: Mat4::new(),
            view_proj_matrix: Mat4::new(),
            normal_matrix: Mat4::new(),
            model_view_matrix: Mat4::new(),
            model_view_proj_matrix: Mat4::new(),
            color_buffer_width: config.image_width as f32,
            color_buffer_height: config.image_height as f32,
        }
    }

    fn frag_pixel_pos(&self, screen_pos: Vec4) -> Vec2 {
        let tminx = self.color_buffer.urange[0];
        let tmaxx = self.color_buffer.urange[1];
        let tminy = self.color_buffer.vrange[0];
        let tmaxy = self.color_buffer.vrange[1];
        vec2![
            lerp(0.0, self.color_buffer_width, tminx, tmaxx, screen_pos[0]),
            lerp(self.color_buffer_height, 0.0, tminy, tmaxy, screen_pos[1])
        ]
    }

    fn min_depth_color(&self, fragprocs: &Vec<FragmentProcessor>, x: i32, y: i32)
        -> (f32, Rgba<u8>) {
        let mut min_depth = f32::MAX;
        let mut min_depth_color = rgba(0, 0, 0, 255);
        for proc in fragprocs {
            let depth = proc.zbuffer.get(x, y);
            if depth < min_depth {
                min_depth = depth;
                min_depth_color = proc.color_buffer.get_pixel(x as u32, y as u32);
            }
        }
        (min_depth, min_depth_color)
    }

    pub fn render_frame(&mut self, camera_index: usize) -> &Texture {
        let camera = self.scene.get_camera(camera_index);
        self.camera_index = camera_index;
        self.view_matrix = camera.view_matrix();
        self.proj_matrix = camera.perspective_matrix();
        self.view_proj_matrix = self.proj_matrix * self.view_matrix;

        self.color_buffer.urange = vec2![-1.0, 1.0];
        self.color_buffer.vrange = vec2![-1.0, 1.0];
        self.color_buffer.fill(self.config.background_color);
        self.zbuffer.depth_range = vec2![0.0, 1.0];
        self.zbuffer.fill(1.0);

        const PRIM_QUEUE_CAP: usize = 1024;
        let (prim_sender, prim_receiver) = bounded(PRIM_QUEUE_CAP);
        let rasconfig = self.config.rasterizer_config.unwrap();

        let fragproc_handles: Vec<thread::JoinHandle<FragmentProcessor>> = 
            (0..rasconfig.fragment_processors).map(|_| {
            let scene = self.scene.clone();
            let config = self.config.clone();
            let camera_index = self.camera_index;
            let receiver = prim_receiver.clone();
            thread::spawn(move || {
                let mut fragproc = FragmentProcessor::new(Some(scene), config, camera_index,
                    receiver);
                fragproc.init();
                fragproc.run();
                fragproc
            })
        }).collect();

        let wireframe_color = rgba_to_vec3(rasconfig.wireframe_color);
        let mut fragtri = FragmentTriangle::new();
        for model in &self.scene.models {
            self.model_index = model.index;
            self.material_index = model.material_index;
            self.mesh_index = model.mesh_index;
            self.normal_matrix = Mat4::transpose(Mat4::inverse(model.model_matrix));
            self.model_view_matrix = self.view_matrix * model.model_matrix;
            self.model_view_proj_matrix = self.proj_matrix * self.model_view_matrix;
            let mesh = self.scene.get_mesh(model.mesh_index);
            let material = self.scene.get_material(model.material_index);
            fragtri.material_index = model.material_index;
            if self.config.render_mode == RenderMode::Points {
                for vertex in &mesh.vertices {
                    let world_pos = Vec3::transform_point(*vertex, model.model_matrix);
                    let mut screen_pos = self.model_view_proj_matrix * vertex.to_point();
                    screen_pos /= screen_pos[3];
                    let pixel_pos = self.frag_pixel_pos(screen_pos);
                    println!("{}", pixel_pos);
                    _ = prim_sender.send(RasterPrimitive::Point(Fragment {
                        world_pos,
                        screen_pos,
                        normal: Vec3::new(),
                        uv: Vec3::new(),
                        pixel_pos,
                        color: wireframe_color
                    })).unwrap();
                }
            } else {
                for face_index in 0..mesh.faces.len() {
                    let face = &mesh.faces[face_index];
                    let mut face_normal = mesh.face_normals[face_index];
                    face_normal = Vec3::transform_dir(face_normal, self.normal_matrix);
                    if rasconfig.backface_culling {
                        let mut v = mesh.vertices[face[0].vertex];
                        v = Vec3::transform_point(v, model.model_matrix);
                        if Vec3::dot(v - camera.pos, face_normal) >= 0.0 {
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
                        } else if self.config.shading_model == ShadingModel::Phong {
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
                            frag.color = phong_shade(self.scene.clone(), &frag, material, camera);
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
                        fragtri.centroid.color = phong_shade(self.scene.clone(),
                            &fragtri.centroid, material, camera);
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

                    if rasconfig.show_face_normals {
                        let color = rgba_to_vec3(rasconfig.face_normal_color);
                        let mut end = Fragment::new();
                        end.world_pos = rasconfig.face_normal_length*fragtri.centroid.normal
                            + fragtri.centroid.world_pos;
                        end.screen_pos = self.view_proj_matrix * end.world_pos.to_point();
                        end.screen_pos /= end.screen_pos[3];
                        end.pixel_pos = self.frag_pixel_pos(end.screen_pos);
                        end.color = color;
                        let mut start = fragtri.centroid;
                        start.color = color;
                        _ = prim_sender.send(RasterPrimitive::Line(start, end)).unwrap();
                    }

                    if rasconfig.show_vertex_normals {
                        let nlen = rasconfig.vertex_normal_length;
                        let color = rgba_to_vec3(rasconfig.vertex_normal_color);
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

            if rasconfig.show_bounding_boxes {
                let corners = model.bounding_box.corners_vec4();
                let mut bbox_frags = [Fragment::new(); 8];
                let color = rgba_to_vec3(rasconfig.bounding_box_color);
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
        let fragprocs: Vec<FragmentProcessor> = fragproc_handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let (w, h) = (self.config.image_width as i32, self.config.image_height as i32);
        for y in 0..h {
            for x in 0..w {
                let (z, color) = self.min_depth_color(&fragprocs, x, y);
                self.zbuffer.set(x, y, z);
                self.color_buffer.set_pixel(x as u32, y as u32, color);
            }
        }
        &self.color_buffer
    }
}

fn render_line_low(
    color_buffer: &mut Texture,
    zbuffer: &mut DepthBuffer,
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
    color_buffer: &mut Texture,
    zbuffer: &mut DepthBuffer,
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
