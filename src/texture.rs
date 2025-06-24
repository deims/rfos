use std::path::Path;
use image::{Rgba, RgbaImage, ImageResult};

use crate::{math::*, scene::INVALID_INDEX};

pub static BLACK:  Rgba<u8> = Rgba::<u8>([0  , 0  , 0  , 255]);
pub static WHITE:  Rgba<u8> = Rgba::<u8>([255, 255, 255, 255]);
pub static RED:    Rgba<u8> = Rgba::<u8>([255, 0  , 0  , 255]);
pub static GREEN:  Rgba<u8> = Rgba::<u8>([0  , 255, 0  , 255]);
pub static BLUE:   Rgba<u8> = Rgba::<u8>([0  , 0  , 255, 255]);
pub static YELLOW: Rgba<u8> = Rgba::<u8>([255, 255, 0  , 255]);
pub static CYAN:   Rgba<u8> = Rgba::<u8>([0  , 255, 255, 255]);
pub static PURPLE: Rgba<u8> = Rgba::<u8>([255, 0  , 255, 255]);

pub fn rgba(r: u8, g: u8, b: u8, a: u8) -> Rgba<u8> {
    Rgba::<u8>([r, g, b, a])
}

pub fn rgb(r: u8, g: u8, b: u8) -> Rgba<u8> {
    rgba(r, g, b, 255)
}

pub fn rgba_from_u32(color: u32) -> Rgba<u8> {
    Rgba::<u8>([
        ((color >> 0)  & 0x000000ff) as u8,
        ((color >> 8)  & 0x000000ff) as u8,
        ((color >> 16) & 0x000000ff) as u8,
        ((color >> 24) & 0x000000ff) as u8,
    ])
}

pub fn rgba_to_u32(color: Rgba<u8>) -> u32 {
    let r = color[0] as u32;
    let g = color[1] as u32;
    let b = color[2] as u32;
    let a = color[3] as u32;
    (a << 24) | (b << 16) | (g << 8) | r
}

pub fn rgba_to_vec4(color: Rgba<u8>) -> Vec4 {
    let (r, g, b, a) = (color[0] as f32, color[1] as f32, color[2] as f32, color[3] as f32);
    vec4![r/255.0, g/255.0, b/255.0, a/255.0]
}

pub fn rgba_to_vec3(color: Rgba<u8>) -> Vec3 {
    rgba_to_vec4(color).to_vec3()
}

pub fn rgba_from_vec4(color: Vec4) -> Rgba<u8> {
    Rgba::<u8>([
        f32::round(color[0] * 255.0) as u8,
        f32::round(color[1] * 255.0) as u8,
        f32::round(color[2] * 255.0) as u8,
        f32::round(color[3] * 255.0) as u8
    ])
}

pub fn rgba_from_vec3(color: Vec3) -> Rgba<u8> {
    rgba_from_vec4(color.to_vec4(1.0))
}

pub struct Texture {
   pub index: usize,
   pub image: RgbaImage,
   pub urange: Vec2,
   pub vrange: Vec2
}

impl Texture {
    pub fn new(width: u32, height: u32) -> Self {
        Texture {
            index: INVALID_INDEX,
            image: RgbaImage::new(width, height),
            urange: vec2![0.0, 1.0],
            vrange: vec2![0.0, 1.0]
        }
    }

    pub fn width(&self) -> u32 { self.image.width() }
    pub fn height(&self) -> u32 { self.image.height() }

    pub fn is_within(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 { return false; }
        let (w, h) = self.image.dimensions();
        return (x as u32) < w && (y as u32) < h;
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> Rgba<u8> {
        *self.image.get_pixel(x, y)
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, p: Rgba<u8>) {
        self.image.put_pixel(x, y, p);
    }

    pub fn fill(&mut self, p: Rgba<u8>) {
        let (w, h) = self.image.dimensions();
        for y in 0..h {
            for x in 0..w {
                self.set_pixel(x, y, p);
            }
        }
    }

    pub fn map_u(&self, u: f32) -> i32 {
        let w = self.image.width() as f32;
        f32::round(lerp(0.0, w, self.urange[0], self.urange[1], u)) as i32
    }

    pub fn map_v(&self, v: f32) -> i32 {
        let h = self.image.height() as f32;
        f32::round(lerp(h, 0.0, self.vrange[0], self.vrange[1], v)) as i32
    }

    pub fn map_uv(&self, u: f32, v: f32) -> (i32, i32) {
        (self.map_u(u), self.map_v(v))
    }

    pub fn get_pixel_uv(&self, u: f32, v: f32) -> Rgba<u8> {
        let (x, y) = self.map_uv(u, v);
        self.get_pixel(x as u32, y as u32)
    }

    pub fn set_pixel_uv(&mut self, u: f32, v: f32, p: Rgba<u8>) {
        let (x, y) = self.map_uv(u, v);
        self.set_pixel(x as u32, y as u32, p);
    }

    pub fn save<P>(&self, path: P) -> ImageResult<()>  where P: AsRef<Path> {
        self.image.save(path)
    }
}
