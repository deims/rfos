use std::sync::{Arc, RwLock};
use image::Rgba;
use crate::math::*;
use crate::texture::Texture;
use crate::scene::{INVALID_INDEX, Camera};

pub struct DepthBuffer {
   pub buffer: Vec<f32>,
   pub width: u32,
   pub height: u32,
   pub depth_range: Vec2,
}

impl DepthBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        if width <= 0 || height <= 0 {
            log::error!("invalid depth buffer dimensions: {} x {}", width, height);
        }
        let (w, h) = (width as usize, height as usize);
        DepthBuffer {
            buffer: vec![0.0f32; w * h],
            width, height,
            depth_range: vec2![0.0, 1.0]
        }
    }

    pub fn coords_to_index(&self, x: i32, y: i32) -> usize {
        let x = x as usize;
        let y = y as usize;
        let w = self.width as usize;
        y*w + x
    }

    pub fn get(&self, x: i32, y: i32) -> f32 {
        self.buffer[self.coords_to_index(x,y)]
    }

    pub fn set(&mut self, x: i32, y: i32, z: f32) {
        let index = self.coords_to_index(x, y);
        self.buffer[index] = z;
    }

    pub fn fill(&mut self, z: f32) {
        self.buffer.fill(z);
    }
}

#[derive(Copy, Clone)]
pub struct Fragment {
    pub world_pos: Vec3,
    pub normal: Vec3,
    pub uv: Vec3,
    pub screen_pos: Vec4,
    pub pixel_pos: Vec2,
    pub color: Vec3
}

impl Fragment {
    pub fn new() -> Self {
        Fragment{
            world_pos: Vec3::new(),
            normal: Vec3::new(),
            uv: Vec3::new(),
            screen_pos: Vec4::new(),
            pixel_pos: Vec2::new(),
            color: Vec3::new()
        }
    }

    pub fn add(&mut self, frag: &Fragment) {
        self.world_pos += frag.world_pos;
        self.normal += frag.normal;
        self.uv += frag.uv;
        self.screen_pos += frag.screen_pos;
        self.pixel_pos += frag.pixel_pos;
        self.color += frag.color;
    }

    pub fn sub(&mut self, frag: &Fragment) {
        self.world_pos -= frag.world_pos;
        self.normal -= frag.normal;
        self.uv -= frag.uv;
        self.screen_pos -= frag.screen_pos;
        self.pixel_pos -= frag.pixel_pos;
        self.color -= frag.color;
    }

    pub fn scale(&mut self, scalar: f32) {
        self.world_pos *= scalar;
        self.normal *= scalar;
        self.uv *= scalar;
        self.screen_pos *= scalar;
        self.pixel_pos *= scalar;
        self.color *= scalar;
    }

    pub fn sum(a: &Fragment, b: &Fragment) -> Fragment {
        Fragment {
            world_pos: a.world_pos + b.world_pos,
            normal: a.normal + b.normal,
            uv: a.uv + b.uv,
            screen_pos: a.screen_pos + b.screen_pos,
            pixel_pos: a.pixel_pos + b.pixel_pos,
            color: a.color + b.color
        }
    }

    pub fn diff(a: &Fragment, b: &Fragment) -> Fragment {
        Fragment {
            world_pos: a.world_pos - b.world_pos,
            normal: a.normal - b.normal,
            uv: a.uv - b.uv,
            screen_pos: a.screen_pos - b.screen_pos,
            pixel_pos: a.pixel_pos - b.pixel_pos,
            color: a.color - b.color
        }
    }

    pub fn scaled(a: &Fragment, scalar: f32) -> Fragment {
        Fragment {
            world_pos: a.world_pos * scalar,
            normal: a.normal * scalar,
            uv: a.uv * scalar,
            screen_pos: a.screen_pos * scalar,
            pixel_pos: a.pixel_pos * scalar,
            color: a.color * scalar
        }
    }

    pub fn lerp(start: &Fragment, end: &Fragment, tmin: f32, tmax: f32, t: f32) -> Fragment {
        Fragment {
            world_pos: Vec3::lerp(start.world_pos, end.world_pos, tmin, tmax, t),
            normal: Vec3::lerp(start.normal, end.normal, tmin, tmax, t),
            uv: Vec3::lerp(start.uv, end.uv, tmin, tmax, t),
            screen_pos: Vec4::lerp(start.screen_pos, end.screen_pos, tmin, tmax, t),
            pixel_pos: Vec2::lerp(start.pixel_pos, end.pixel_pos, tmin, tmax, t),
            color: Vec3::lerp(start.color, end.color, tmin, tmax, t)
        }
    }
}

#[derive(Clone, Copy)]
pub struct FragmentTriangle {
   pub vertices: [Fragment; 3],
   pub centroid: Fragment,
   pub material_index: usize,
   pub winding: i64,
   pub area: f32,
   pub clipped: bool
}

pub fn int_pixel_pos(v: Vec2) -> [i64; 2] {
    [f32::round(v[0]) as i64, f32::round(v[1]) as i64]
}

impl FragmentTriangle {
    pub fn new() -> Self {
        FragmentTriangle {
            vertices: [Fragment::new(); 3],
            centroid: Fragment::new(),
            material_index: INVALID_INDEX,
            winding: 0,
            area: 0.0,
            clipped: false
        }
    }

    pub fn compute_winding(&self) -> i64 {
        let v0 = int_pixel_pos(self.vertices[0].pixel_pos);
        let v1 = int_pixel_pos(self.vertices[1].pixel_pos);
        let v2 = int_pixel_pos(self.vertices[2].pixel_pos);
        (v1[0] - v0[0])*(v2[1] - v0[1]) - (v1[1] - v0[1])*(v2[0] - v0[0])
    }

    pub fn compute_area(&self) -> f32 {
        let x = self.vertices[2].pixel_pos[0];
        let y = self.vertices[2].pixel_pos[1];
        self.edge_function(0, 1, x, y)
    }

    pub fn edge_function(&self, ai: usize, bi: usize, x: f32, y: f32) -> f32 {
        let a = self.vertices[ai].pixel_pos;
        let b = self.vertices[bi].pixel_pos;
        if self.winding > 0 {
            (x - a[0])*(b[1] - a[1]) - (y - a[1])*(b[0] - a[0])
        } else {
            (a[0] - b[0])*(y - a[1]) - (a[1] - b[1])*(x - a[0])
        }
    }

    pub fn pixel_extents(&self) -> ([i32; 2], [i32; 2]) {
        let mut min = [i32::MAX; 2];
        let mut max = [i32::MIN; 2];
        for i in 0..3 {
            let pos = int_pixel_pos(self.vertices[i].pixel_pos);
            let pos = [pos[0] as i32, pos[1] as i32];
            min[0] = i32::min(min[0], pos[0]);
            min[1] = i32::min(min[1], pos[1]);
            max[0] = i32::max(max[0], pos[0]);
            max[1] = i32::max(max[1], pos[1]);
        }
        (min, max)
    }

    pub fn weighted_sum(&self, weights: Vec3) -> Fragment {
        let mut frag = Fragment {
            world_pos: self.vertices[0].world_pos * weights[0],
            normal: self.vertices[0].normal * weights[0],
            uv: self.vertices[0].uv * weights[0],
            screen_pos: self.vertices[0].screen_pos * weights[0],
            color: self.vertices[0].color * weights[0],
            pixel_pos: self.vertices[0].pixel_pos * weights[0]
        };

        for i in 1..3 {
            frag.world_pos += self.vertices[i].world_pos * weights[i];
            frag.normal += self.vertices[i].normal * weights[i];
            frag.uv += self.vertices[i].uv * weights[i];
            frag.screen_pos += self.vertices[i].screen_pos * weights[i];
            frag.color += self.vertices[i].color * weights[i];
            frag.pixel_pos += self.vertices[i].pixel_pos * weights[i]
        }
        frag
    }
}

#[derive(Copy, Clone)]
pub enum RasterPrimitive {
    Point(Fragment),
    Line(Fragment, Fragment),
    Triangle(FragmentTriangle)
}

#[derive(PartialEq, Copy, Clone)]
pub enum RenderMode {Points, Wireframe, Filled}

#[derive(PartialEq, Copy, Clone)]
pub enum ShadingModel {None, Flat, Phong}

#[derive(Copy, Clone)]
pub struct RasterizerConfig {
    pub vertex_processors: usize,
    pub fragment_processors: usize,
    pub backface_culling: bool,
    pub show_face_normals: bool,
    pub show_vertex_normals: bool,
    pub show_bounding_boxes: bool,
    pub show_wireframe: bool,
    pub face_normal_length: f32,
    pub vertex_normal_length: f32,
    pub face_normal_color: Rgba<u8>,
    pub vertex_normal_color: Rgba<u8>,
    pub bounding_box_color: Rgba<u8>,
    pub wireframe_color: Rgba<u8>,
}

#[derive(Copy, Clone)]
pub struct RaytracerConfig {
    pub worker_count: usize,
    pub octree_leaf_capacity: usize,
    pub octree_min_node_size: f32
}

#[derive(Clone)]
pub struct RenderConfig {
    pub output_file: String,
    pub image_width: u32,
    pub image_height: u32,
    pub render_mode: RenderMode,
    pub shading_model: ShadingModel,
    pub background_color: Rgba<u8>,
    pub rasterizer_config: Option<RasterizerConfig>,
    pub raytracer_config: Option<RaytracerConfig>
}

pub trait Renderer {
    fn render_frame(&mut self, cam: &Camera) -> &Arc<RwLock<Texture>>;
}
