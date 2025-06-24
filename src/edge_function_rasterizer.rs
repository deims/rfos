use image::Rgba;
use crate::math::*;
use crate::texture::*;
use crate::scene::*;
use crate::render::*;
use crate::ring_buffer::*;

fn clip_planes(cam: &Camera) -> [Plane; 6] {
    let zero = vec3![0.0, 0.0, 0.0];
    [
        Plane{normal: vec3![0.0, 0.0, 1.0], point: vec3![0.0, 0.0, cam.znear/cam.zfar]},
        Plane{normal: vec3![0.0, 0.0, -1.0], point: vec3![0.0, 0.0, 1.0]},
        Plane{normal: Vec3::normalize(vec3![1.0, 0.0, 1.0]), point: zero},
        Plane{normal: Vec3::normalize(vec3![-1.0, 0.0, 1.0]), point: zero},
        Plane{normal: Vec3::normalize(vec3![0.0, -1.0, 1.0]), point: zero},
        Plane{normal: Vec3::normalize(vec3![0.0, 1.0, 1.0]), point: zero}
    ]
}

fn mirror_clip_planes(planes: &mut [Plane; 6]) {
    for i in 0..6 {
        planes[i].normal[2] = -planes[i].normal[2];
        planes[i].point[2] = -planes[i].point[2];
    }
}

fn plane_edge_intersection(plane: &Plane, a: &Fragment, b: &Fragment) -> Fragment {
    let ap = plane.point - a.screen_pos.to_vec3();
    let ab = (b.screen_pos - a.screen_pos).to_vec3();
    let t = Vec3::dot(plane.normal, ap) / Vec3::dot(plane.normal, ab);
    Fragment::lerp(a, b, 0.0, 1.0, t)
}

pub struct EdgeFunctionRasterizer<'a> {
    scene: &'a Scene,
    config: &'a RenderConfig,
    color_buffer: Texture,
    zbuffer: DepthBuffer,
    camera: Option<&'a Camera>,
    model: Option<&'a Model>,
    material: Option<&'a Material>,
    mesh: Option<&'a Mesh>,
    view_matrix: Mat4,
    proj_matrix: Mat4,
    view_proj_matrix: Mat4,
    model_view_matrix: Mat4,
    model_view_proj_matrix: Mat4,
    normal_matrix: Mat4,
    rendered_face_count: u64,
    color_buffer_width: f32,
    color_buffer_height: f32,
    fragtri: FragmentTriangle,
    clipped_faces: RingBuffer<Fragment>
}

impl<'a> EdgeFunctionRasterizer<'a> {
    pub fn new(scene: &'a Scene, config: &'a RenderConfig) -> Self {
        EdgeFunctionRasterizer {
            scene,
            config,
            color_buffer: Texture::new(config.image_width, config.image_height),
            zbuffer: DepthBuffer::new(config.image_width, config.image_height),
            camera: None,
            model: None,
            material: None,
            mesh: None,
            view_matrix: Mat4::new(),
            proj_matrix: Mat4::new(),
            view_proj_matrix: Mat4::new(),
            model_view_matrix: Mat4::new(),
            model_view_proj_matrix: Mat4::new(),
            normal_matrix: Mat4::new(),
            rendered_face_count: 0,
            color_buffer_width: config.image_width as f32,
            color_buffer_height: config.image_height as f32,
            fragtri: FragmentTriangle::new(),
            clipped_faces: RingBuffer::new(8)
        }
    }

    pub fn get_rendered_face_count(&self) -> u64 { self.rendered_face_count }

    pub fn float_pixel_pos(&self, screen_pos: Vec4) -> Vec2 {
        let tmin = self.color_buffer.urange[0];
        let tmax = self.color_buffer.urange[1];
        vec2![
            lerp(0.0, self.color_buffer_width, tmin, tmax, screen_pos[0]),
            lerp(self.color_buffer_height, 0.0, tmin, tmax, screen_pos[1])
        ]
    }

    pub fn render_frame(&mut self, cam: &'a Camera) -> &Texture {
        self.camera = Some(cam);
        self.view_matrix = cam.view_matrix();
        self.proj_matrix = cam.perspective_matrix();
        self.view_proj_matrix = self.proj_matrix * self.view_matrix;
        self.rendered_face_count = 0;

        self.color_buffer.urange = vec2![-1.0, 1.0];
        self.color_buffer.vrange = vec2![-1.0, 1.0];
        self.color_buffer.fill(self.config.background_color);
        self.zbuffer.depth_range = vec2![0.0, 1.0];
        self.zbuffer.fill(self.zbuffer.depth_range[1]);

        for model in &self.scene.models {
            self.model = Some(model);
            self.material = Some(self.scene.get_material(model.material_index));
            self.mesh = Some(self.scene.get_mesh(model.mesh_index));
            self.model_view_matrix = self.view_matrix * model.model_matrix;
            self.model_view_proj_matrix = self.proj_matrix * self.model_view_matrix;
            self.normal_matrix = Mat4::transpose(Mat4::inverse(model.model_matrix));
            let mesh = self.mesh.unwrap();
            let material = self.material.unwrap();
            if self.config.render_mode == RenderMode::Points {
                for v in &mesh.vertices {
                    let v = self.model_view_proj_matrix * v.to_point();
                    let v = v * (1.0 / v[3]);
                    let (x, y) = self.color_buffer.map_uv(v[0], v[1]);
                    if self.color_buffer.is_within(x, y) {
                        let (x, y) = (x as u32, y as u32);
                        self.color_buffer.set_pixel(x, y, self.config.wireframe_color);
                    }
                }
            } else {
                for face_index in 0..mesh.faces.len() {
                    let face = &mesh.faces[face_index];
                    let mut face_normal = mesh.face_normals[face_index];
                    face_normal = Vec3::transform_dir(face_normal, self.normal_matrix);
                    if self.config.backface_culling {
                        let v = mesh.vertices[face[0].vertex];
                        let v = Vec3::transform_point(v, model.model_matrix);
                        if Vec3::dot(v - cam.pos, face_normal) >= 0.0 {
                            continue;
                        }
                    }

                    for i in 0..3 {
                        let frag = &mut self.fragtri.vertices[i];
                        let v = mesh.vertices[face[i].vertex];
                        let uv = mesh.vertex_uvs[face[i].uv];
                        frag.world_pos = Vec3::transform_point(v, model.model_matrix);
                        frag.screen_pos = self.model_view_matrix * v.to_point();
                        if self.config.shading_model == ShadingModel::Flat {
                            frag.normal = face_normal;
                        } else {
                            let vnormal = mesh.vertex_normals[face[i].normal];
                            frag.normal = Vec3::transform_dir(vnormal, self.normal_matrix);
                        }
                        frag.uv = vec3![uv[0], uv[1], 1.0] / frag.screen_pos[2];
                    }

                    self.clipped_faces.clear();

                    const ONE_THIRD: f32 = 1.0/3.0;
                    self.fragtri.centroid.world_pos = Vec3::new();
                    self.fragtri.centroid.normal = Vec3::new();
                    self.fragtri.centroid.uv = Vec3::new();
                    self.fragtri.centroid.screen_pos = Vec4::new();
                    for i in 0..3 {
                        let mut frag = self.fragtri.vertices[i];
                        frag.screen_pos = self.proj_matrix * frag.screen_pos;
                        frag.screen_pos /= frag.screen_pos[3];
                        frag.pixel_pos = self.float_pixel_pos(frag.screen_pos);
                        if self.config.shading_model == ShadingModel::None {
                            frag.color = material.diffuse_color;
                        } else {
                            frag.color = self.phong_shade(&frag);
                        }
                        self.fragtri.centroid.world_pos += frag.world_pos;
                        self.fragtri.centroid.uv += frag.uv;
                        self.fragtri.centroid.screen_pos += frag.screen_pos;
                        self.fragtri.vertices[i] = frag;
                    }
                    self.fragtri.centroid.world_pos *= ONE_THIRD;
                    self.fragtri.centroid.normal = face_normal;
                    self.fragtri.centroid.uv *= ONE_THIRD;
                    self.fragtri.centroid.screen_pos *= ONE_THIRD;
                    self.fragtri.centroid.pixel_pos = self.float_pixel_pos(self.fragtri.centroid.screen_pos);
                    if self.config.shading_model == ShadingModel::None {
                        self.fragtri.centroid.color = material.diffuse_color;
                    } else {
                        self.fragtri.centroid.color = self.phong_shade(&self.fragtri.centroid);
                    }

                    self.render_face();
                    self.rendered_face_count += 1;
                }
            }
            if self.config.show_bounding_boxes {
                let corners = model.bounding_box.corners_vec4();
                let mut bbox_frags = [Fragment::new(); 8];
                for i in 0..8 {
                    bbox_frags[i].screen_pos = self.view_proj_matrix * corners[i];
                    bbox_frags[i].screen_pos /= bbox_frags[i].screen_pos[3];
                    bbox_frags[i].pixel_pos = self.float_pixel_pos(bbox_frags[i].screen_pos);
                }
                for i in 0..4 {
                    let j = (i+1) % 4;
                    self.render_line(&bbox_frags[i],   &bbox_frags[j],   self.config.bounding_box_color);
                    self.render_line(&bbox_frags[i+4], &bbox_frags[j+4], self.config.bounding_box_color);
                    self.render_line(&bbox_frags[i],   &bbox_frags[i+4], self.config.bounding_box_color);
                }
            }
        }
        &self.color_buffer
    }

    // fn pop_triangle(&mut self) -> crate::ring_buffer::Result<[Fragment; 3]> {
    //     let mut tri = [Fragment::new(); 3];
    //     for i in 0..3 {
    //         match self.clipped_faces.pop() {
    //             Ok(frag) => tri[i] = frag,
    //             Err(err) => {
    //                 log::error!("clipping queue underflow");
    //                 return Err(err)
    //             }
    //         }
    //     }
    //     Ok(tri)
    // }
    //
    // fn push_triangle(&mut self, tri: &[Fragment; 3]) -> crate::ring_buffer::Result<()> {
    //     for i in 0..3 {
    //         if let Err(err) = self.clipped_faces.push(tri[i]) {
    //             log::error!("clipping queue overflow");
    //             return Err(err)
    //         }
    //     }
    //     Ok(())
    // }
    //
    // fn clip_face(&mut self, plane: &Plane) {
    //     let mut newtri = [Fragment::new(); 3];
    //     let mut oldtri = [Fragment::new(); 3];
    //     let queue_size = self.clipped_faces.len();
    //     for i in 0..queue_size/3 {
    //         if let Ok(tri) = self.pop_triangle() { oldtri = tri; }
    //         else { return; }
    //         let d0 = Plane::signed_dist_vec4(plane, oldtri[0].screen_pos);
    //         let d1 = Plane::signed_dist_vec4(plane, oldtri[1].screen_pos);
    //         let d2 = Plane::signed_dist_vec4(plane, oldtri[2].screen_pos);
    //         if d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0 {
    //             if let Err(_) = self.push_triangle(&oldtri) { return; }
    //             continue;
    //         }
    //         if d0 < 0.0 && d1 < 0.0 && d2 < 0.0 { continue; }
    //     }
    // }

    fn render_face(&mut self) {
        if self.config.render_mode == RenderMode::Wireframe {
            let f = self.fragtri.vertices;
            self.render_line(&f[0], &f[1], self.config.wireframe_color);
            self.render_line(&f[1], &f[2], self.config.wireframe_color);
            self.render_line(&f[2], &f[0], self.config.wireframe_color);
        } else if self.config.render_mode == RenderMode::Filled {
            self.fragtri.winding = self.fragtri.compute_winding();
            self.fragtri.area = self.fragtri.compute_area();
            let (min, max) = self.fragtri.pixel_extents();
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    if !self.color_buffer.is_within(x, y) { continue; }
                    let p = vec2![x as f32 + 0.5, y as f32 + 0.5];
                    let mut bc = vec3! [
                        self.fragtri.edge_function(1, 2, p[0], p[1]),
                        self.fragtri.edge_function(2, 0, p[0], p[1]),
                        self.fragtri.edge_function(0, 1, p[0], p[1])
                    ];
                    if bc[0] >= 0.0 && bc[1] >= 0.0 && bc[2] >= 0.0 {
                        bc /= self.fragtri.area;
                        let frag = self.fragtri.weighted_sum(bc);
                        if frag.screen_pos[2] < self.zbuffer.get(x, y) {
                            let color = self.phong_shade(&frag);
                            self.color_buffer.set_pixel(x as u32, y as u32, rgba_from_vec3(color));
                            self.zbuffer.set(x, y, frag.screen_pos[2]);
                        }
                    }
                }
            }
        } else {
            unreachable!();
        }

        if self.config.show_face_normals {
            let mut end = Fragment::new();
            end.world_pos = self.config.face_normal_length*self.fragtri.centroid.normal + self.fragtri.centroid.world_pos;
            end.screen_pos = self.view_proj_matrix * end.world_pos.to_point();
            end.screen_pos /= end.screen_pos[3];
            end.pixel_pos = self.float_pixel_pos(end.screen_pos);
            let start = self.fragtri.centroid;
            self.render_line(&start, &end, self.config.face_normal_color);
        }

        if self.config.show_vertex_normals {
            let mut end = Fragment::new();
            for i in 0..3 {
                end.world_pos = self.config.vertex_normal_length*self.fragtri.vertices[i].normal + self.fragtri.vertices[i].world_pos;
                end.screen_pos = self.view_proj_matrix * end.world_pos.to_point();
                end.screen_pos /= end.screen_pos[3];
                end.pixel_pos = self.float_pixel_pos(end.screen_pos);
                let start = self.fragtri.vertices[i];
                self.render_line(&start, &end, self.config.vertex_normal_color);
            }
        }
    }

    fn render_line(&mut self, start: &Fragment, end: &Fragment, color: Rgba<u8>) {
        let startpos = start.pixel_pos.elems.map(|x| f32::round(x) as i32);
        let endpos = end.pixel_pos.elems.map(|x| f32::round(x) as i32);
        if i32::abs(endpos[1] - startpos[1]) < i32::abs(endpos[0] - startpos[0]) {
            if startpos[0] > endpos[0] {
                self.render_line_low(end, endpos, start, startpos, color);
            } else {
                self.render_line_low(start, startpos, end, endpos, color);
            }
        } else {
            if startpos[1] > endpos[1] {
                self.render_line_high(end, endpos, start, startpos, color);
            } else {
                self.render_line_high(start, startpos, end, endpos, color);
            }
        }
    }

    fn render_line_low(&mut self, start: &Fragment, startpos: [i32; 2], end: &Fragment,
        endpos: [i32; 2], color: Rgba<u8>) {

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
            if self.color_buffer.is_within(x, y) && z < self.zbuffer.get(x, y) {
                self.color_buffer.set_pixel(x as u32, y as u32, color);
                self.zbuffer.set(x, y, z);
            }
            if d > 0 {
                y += yi;
                d -= 2*dx;
            }
            d += 2*dy;
            z += dz;
        }
    }

    fn render_line_high(&mut self, start: &Fragment, startpos: [i32; 2], end: &Fragment,
        endpos: [i32; 2], color: Rgba<u8>) {

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
            if self.color_buffer.is_within(x, y) && z < self.zbuffer.get(x, y) {
                self.color_buffer.set_pixel(x as u32, y as u32, color);
                self.zbuffer.set(x, y, z);
            }
            if d > 0 {
                x += xi;
                d -= 2*dy;
            }
            d += 2*dx;
            z += dz;
        }
    }

    fn phong_shade(&self, frag: &Fragment) -> Vec3 {
        self.material.unwrap().diffuse_color
    }
}


