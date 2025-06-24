use std::path::Path;
use std::io::{Result, Error, ErrorKind};
use core::str::SplitAsciiWhitespace;
use image::{RgbaImage, ImageResult};

use crate::math::*;
use crate::texture::*;

pub static INVALID_INDEX: usize = usize::MAX >> 1;

#[derive(Copy, Clone, Default)]
pub struct VertexIndex {
    pub vertex: usize,
    pub normal: usize,
    pub uv: usize
}

pub type Face = [VertexIndex; 3];

pub struct Mesh {
    pub index: usize,
    pub vertices: Vec<Vec3>,
    pub vertex_normals: Vec<Vec3>,
    pub vertex_uvs: Vec<Vec2>,
    pub faces: Vec<Face>,
    pub face_normals: Vec<Vec3>
}

pub struct Model {
    pub index: usize,
    pub mesh_index: usize,
    pub material_index: usize,
    pub model_matrix: Mat4,
    pub bounding_box: AABB,
}

pub struct Material {
    pub index: usize,
    pub diffuse_color: Vec3,
    pub specular_color: Vec3,
    pub ambient_coeff: Vec3,
    pub diffuse_coeff: Vec3,
    pub specular_coeff: Vec3,
    pub specular_exp: f32,
    pub diffuse_map_index: usize,
    pub specular_map_index: usize,
    pub normal_map_index: usize
}

impl Material {
    pub fn default() -> Self {
        Material {
            index: INVALID_INDEX,
            diffuse_color: vec3![0.5, 0.5, 0.5],
            specular_color: vec3![1.0, 1.0, 1.0],
            ambient_coeff: vec3![1.0, 1.0, 1.0],
            diffuse_coeff: vec3![0.8, 0.8, 0.8],
            specular_coeff: vec3![0.5, 0.5, 0.5],
            specular_exp: 8.0,
            diffuse_map_index: INVALID_INDEX,
            specular_map_index: INVALID_INDEX,
            normal_map_index: INVALID_INDEX
        }
    }
}

pub struct PointLight {
    pub index: usize,
    pub pos: Vec3,
    pub color: Vec3,
    pub attenuation: Vec3
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub index: usize,
    pub pos: Vec3,
    pub look: Vec3,
    pub up: Vec3,
    pub znear: f32,
    pub zfar: f32,
    pub horizontal_fov: f32,
    pub aspect_ratio: f32
}

impl Camera {
    pub fn vertical_fov(&self) -> f32 {
        2.0 * f32::atan(f32::tan(0.5 * self.horizontal_fov) / self.aspect_ratio)
    }

    pub fn view_matrix(&self) -> Mat4 {
        let t = Mat4::translation(-self.pos);
        let n = -self.look;
        let v = Vec3::normalize((-Vec3::dot(self.up, n) * n) + self.up);
        let w = Vec3::normalize(Vec3::cross(v, n));
        let r = mat4!(
            w[0], w[1], w[2], 0.0,
            v[0], v[1], v[2], 0.0,
            n[0], n[1], n[2], 0.0,
            0.0 , 0.0 , 0.0 , 1.0
        );
        let vfov = self.vertical_fov();
        let s1 = Mat4::scaling(vec3![
            1.0/f32::tan(self.horizontal_fov*0.5),
            1.0/f32::tan(vfov*0.5),
            1.0
        ]);
        let s2 = Mat4::scaling(vec3![1.0/self.zfar, 1.0/self.zfar, 1.0/self.zfar]);
        s2 * s1 * r * t
    }

    pub fn perspective_matrix(&self) -> Mat4 {
        let k = self.znear / self.zfar;
        mat4!(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0/(k - 1.0), k/(k - 1.0),
            0.0, 0.0, -1.0, 0.0
        )
    }
}

pub struct Scene {
    pub textures: Vec<Texture>,
    pub meshes: Vec<Mesh>,
    pub models: Vec<Model>,
    pub materials: Vec<Material>,
    pub point_lights: Vec<PointLight>,
    pub cameras: Vec<Camera>,
}

pub fn compute_aabb(mesh: &Mesh, model_matrix: Mat4) -> AABB {
    let mut bbox = AABB {
        min: vec3![f32::MAX, f32::MAX, f32::MAX],
        max: vec3![f32::MIN, f32::MIN, f32::MIN]
    };
    for v in &mesh.vertices {
        let mv = Vec3::transform_point(*v, model_matrix);
        for i in 0..3 {
            bbox.min[i] = f32::min(mv[i], bbox.min[i]);
            bbox.max[i] = f32::max(mv[i], bbox.max[i]);
        }
    }
    bbox
}

fn parse_obj_vec<const N: usize>(token_iter: &mut SplitAsciiWhitespace) -> Result<VecN<N>> {
    let mut v = VecN::<N>::new();
    let errmsg = "invalid vector";
    for i in 0..N {
        let tok = token_iter.next();
        if tok.is_none() { return Err(Error::new(ErrorKind::InvalidData, errmsg)); }
        let tok = tok.unwrap();
        let x = tok.parse::<f32>();
        if x.is_err() { return Err(Error::new(ErrorKind::InvalidData, errmsg)); }
        v[i] = x.unwrap();
    }
    Ok(v)
}

fn parse_obj_index(token: Option<&str>) -> Result<usize> {
    if token.is_none() { return Ok(INVALID_INDEX); }
    let token = token.unwrap();
    match token.parse::<usize>() {
        Err(_) => Err(Error::new(ErrorKind::InvalidData, "invalid index")),
        Ok(index) => {
            if index <= 0 {
                return Err(Error::new(ErrorKind::InvalidData, "index less than one"));
            }
            Ok(index)
        }
    }
}

pub fn load_wavefront_obj<P>(path: P) -> Result<Mesh> where P: AsRef<Path> + ToString {
    let content = std::fs::read_to_string(&path)?;
    let pathstr = path.to_string();
    let mut vertices = Vec::<Vec3>::new();
    let mut vertex_normals = Vec::<Vec3>::new();
    let mut vertex_uvs = Vec::<Vec2>::new();
    let mut faces = Vec::<Face>::new();
    let mut face_normals = Vec::<Vec3>::new();
    let mut line_number = 1usize;
    let mut missing_normal_indices = false;
    let mut missing_uv_indices = false;
    for line in content.lines() {
        let line = line.trim();
        if line.len() == 0 { continue; }
        let mut tokens = line.split_ascii_whitespace();
        let first_token = tokens.next();
        if first_token.is_none() {
            let errmsg = format!("{}:{}: malformed line", &pathstr, line_number);
            log::error!("{}", &errmsg);
            return Err(Error::new(ErrorKind::InvalidData, errmsg));
        }
        let first_token = first_token.unwrap();
        match first_token {
            "v"  => { vertices.push(parse_obj_vec::<3>(&mut tokens)?); },
            "vn" => { vertex_normals.push(parse_obj_vec::<3>(&mut tokens)?); },
            "vt" => { vertex_uvs.push(parse_obj_vec::<2>(&mut tokens)?); },
            "o"  => { log::info!("{}:{} object name ignored", &pathstr, line_number); },
            "#"  => { continue; }
            "f"  => {
                let mut face = [VertexIndex::default(); 3];
                for i in 0..3 {
                    let token = tokens.next();
                    if token.is_none() {
                        let errmsg = format!("{}:{}: invalid face", &pathstr, line_number);
                        log::error!("{}", &errmsg);
                        return Err(Error::new(ErrorKind::InvalidData, errmsg));
                    }
                    let mut index_iter = token.unwrap().split('/');
                    let vertex_index = parse_obj_index(index_iter.next())?;
                    let uv_index = parse_obj_index(index_iter.next())?;
                    let normal_index = parse_obj_index(index_iter.next())?;
                    if vertex_index == INVALID_INDEX {
                        let errmsg = format!("{}:{}: face is missing a vertex index", &pathstr, line_number);
                        log::error!("{}", &errmsg);
                        return Err(Error::new(ErrorKind::InvalidData, errmsg));
                    }
                    if uv_index == INVALID_INDEX { missing_uv_indices = true; }
                    if normal_index == INVALID_INDEX { missing_normal_indices = true; }
                    face[i].vertex = vertex_index - 1;
                    face[i].normal = normal_index - 1;
                    face[i].uv = uv_index - 1;
                }
                faces.push(face);
            },
            _ => {
                log::info!("{}:{}: unrecognized first token '{}'", &pathstr, line_number,
                    first_token);
            }
        }
        line_number += 1;
    }

    for face in &faces {
        face_normals.push(face_normal(
            vertices[face[0].vertex],
            vertices[face[1].vertex],
            vertices[face[2].vertex]
        ));
    }

    if missing_normal_indices { log::warn!("{}: missing vertex normal indices", &pathstr); }
    if missing_uv_indices { log::warn!("{}: missing uv indices", &pathstr); }

    Ok(Mesh{index: INVALID_INDEX, vertices, vertex_normals, vertex_uvs, faces, face_normals})
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            textures: Vec::<Texture>::new(),
            meshes: Vec::<Mesh>::new(),
            models: Vec::<Model>::new(),
            materials: Vec::<Material>::new(),
            point_lights: Vec::<PointLight>::new(),
            cameras: Vec::<Camera>::new(),
        }
    }

    pub fn get_texture(&self, index: usize)      -> &Texture    { &self.textures[index] }
    pub fn get_mesh(&self, index: usize)         -> &Mesh       { &self.meshes[index] }
    pub fn get_model(&self, index: usize)        -> &Model      { &self.models[index] }
    pub fn get_material(&self, index: usize)     -> &Material   { &self.materials[index] }
    pub fn get_point_lights(&self, index: usize) -> &PointLight { &self.point_lights[index] }
    pub fn get_camera(&self, index: usize)       -> &Camera     { &self.cameras[index] }

    pub fn create_texture(&mut self, width: u32, height: u32) -> usize {
        let index = self.textures.len();
        self.textures.push(Texture{
            index,
            image: RgbaImage::new(width, height),
            urange: vec2![0.0, 1.0],
            vrange: vec2![0.0, 1.0]
        });
        index
    }

    pub fn load_texture<P>(&mut self, path: P) -> ImageResult<usize> where P: AsRef<Path> {
        let image = image::open(path)?;
        let index = self.textures.len();
        self.textures.push(Texture{
            index,
            image: image.into(),
            urange: vec2![0.0, 1.0],
            vrange: vec2![0.0, 1.0]
        });
        Ok(index)
    }

    pub fn create_mesh(&mut self) -> usize {
        let index = self.meshes.len();
        self.meshes.push(Mesh{
            index,
            vertices: Vec::<Vec3>::new(),
            vertex_normals: Vec::<Vec3>::new(),
            vertex_uvs: Vec::<Vec2>::new(),
            faces: Vec::<Face>::new(),
            face_normals: Vec::<Vec3>::new()
        });
        index
    }

    pub fn load_wavefront_obj<P>(&mut self, path: P) -> Result<usize> where P: AsRef<Path> + ToString {
        let index = self.meshes.len();
        let mut mesh = load_wavefront_obj(path)?;
        mesh.index = index;
        self.meshes.push(mesh);
        Ok(index)
    }

    pub fn create_box_mesh(&mut self, width: f32, height: f32, depth: f32) -> usize {
        static VERTEX_COUNT: usize = 8;
        static VERTEX_UV_COUNT: usize = 14;
        static FACE_COUNT: usize = 12;

        static VERTICES: [Vec3; VERTEX_COUNT] = [
            vec3![0.5, 0.5, -0.5],
            vec3![0.5, -0.5, -0.5],
            vec3![0.5, 0.5, 0.5],
            vec3![0.5, -0.5, 0.5],
            vec3![-0.5, 0.5, -0.5],
            vec3![-0.5, -0.5, -0.5],
            vec3![-0.5, 0.5, 0.5],
            vec3![-0.5, -0.5, 0.5],
        ];

        static VERTEX_NORMALS: [Vec3; VERTEX_COUNT] = [
            vec3![-0.5774, 0.5774, -0.5774],
            vec3![0.5774, 0.5774, 0.5774],
            vec3![0.5774, 0.5774, -0.5774],
            vec3![-0.5774, -0.5774, 0.5774],
            vec3![0.5774, -0.5774, 0.5774],
            vec3![-0.5774, 0.5774, 0.5774],
            vec3![-0.5774, -0.5774, -0.5774],
            vec3![0.5774, -0.5774, -0.5774],
        ];

        static VERTEX_UVS: [Vec2; VERTEX_UV_COUNT] = [
            vec2![0.625, 0.50],
            vec2![0.375, 0.50],
            vec2![0.625, 0.75],
            vec2![0.375, 0.75],
            vec2![0.875, 0.50],
            vec2![0.625, 0.25],
            vec2![0.125, 0.50],
            vec2![0.375, 0.25],
            vec2![0.875, 0.75],
            vec2![0.625, 1.00],
            vec2![0.625, 0.00],
            vec2![0.375, 0.00],
            vec2![0.375, 1.00],
            vec2![0.125, 0.75],
        ];

        // these indices are 1-based so we must subtract one when adding them
        static FACES: [[usize; 9]; FACE_COUNT] = [
            [5,5,1,  3,3,2,  1,1,3],
            [3,3,2,  8,13,4, 4,4,5],
            [7,11,6, 6,8,7,  8,12,4],
            [2,2,8,  8,14,4, 6,7,7],
            [1,1,3,  4,4,5,  2,2,8],
            [5,6,1,  2,2,8,  6,8,7],
            [5,5,1,  7,9,6,  3,3,2],
            [3,3,2,  7,10,6, 8,13,4],
            [7,11,6, 5,6,1,  6,8,7],
            [2,2,8,  4,4,5,  8,14,4],
            [1,1,3,  3,3,2,  4,4,5],
            [5,6,1,  1,1,3,  2,2,8],
        ];

        let scaling_matrix = Mat4::scaling(vec3![width, height, depth]);
        let mut vertices = Vec::<Vec3>::new();
        let mut vertex_normals = Vec::<Vec3>::new();
        let mut vertex_uvs = Vec::<Vec2>::new();
        let mut faces = Vec::<Face>::new();
        let mut face_normals = Vec::<Vec3>::new();

        for i in 0..VERTEX_COUNT {
            vertices.push(Vec3::transform_point(VERTICES[i], scaling_matrix));
            vertex_normals.push(VERTEX_NORMALS[i]);
        }

        for i in 0..VERTEX_UV_COUNT {
            vertex_uvs.push(VERTEX_UVS[i]);
        }

        for i in 0..FACE_COUNT {
            faces.push([
                VertexIndex{vertex: FACES[i][0]-1, normal: FACES[i][2]-1, uv: FACES[i][1]-1},
                VertexIndex{vertex: FACES[i][3]-1, normal: FACES[i][5]-1, uv: FACES[i][4]-1},
                VertexIndex{vertex: FACES[i][6]-1, normal: FACES[i][8]-1, uv: FACES[i][7]-1}
            ]);
        }

        for face in &faces {
            face_normals.push(face_normal(
                vertices[face[0].vertex],
                vertices[face[1].vertex],
                vertices[face[2].vertex]
            ));
        }

        let index = self.meshes.len();
        self.meshes.push(Mesh{index, vertices, vertex_normals, vertex_uvs, faces, face_normals});
        index
    }

    pub fn create_plane_mesh(&mut self, width: f32, length: f32, xseg: usize, zseg: usize) -> usize {
        let xpoints = xseg + 1;
        let zpoints = xseg + 1;
        let fxseg = xseg as f32;
        let fzseg = zseg as f32;
        let normal = vec3![0.0, 1.0, 0.0];
        let mut vertices = Vec::<Vec3>::new();
        let mut vertex_normals = Vec::<Vec3>::new();
        let mut vertex_uvs = Vec::<Vec2>::new();
        let mut faces = Vec::<Face>::new();
        let mut face_normals = Vec::<Vec3>::new();

        vertex_normals.push(normal);
        for z in 0..zpoints {
            let fz = z as f32;
            let zpos = (fz / fzseg - 0.5) * length;
            for x in 0..xpoints {
                let fx = x as f32;
                let xpos = (fx / fxseg - 0.5) * width;
                vertices.push(vec3![xpos, 0.0, zpos]);
                vertex_uvs.push(vec2![fx/fxseg, fz/fzseg]);
            }
        }

        for quad in 0..xseg*zseg {
            let i = quad % xseg + (quad / zseg * xpoints);
            faces.push([
                VertexIndex{vertex: i+xpoints, normal: 0, uv: i+xpoints},
                VertexIndex{vertex: i+1,       normal: 0, uv: i+1},
                VertexIndex{vertex: i,         normal: 0, uv: i}
            ]);
            faces.push([
                VertexIndex{vertex: i+xpoints,   normal: 0, uv: i+xpoints},
                VertexIndex{vertex: i+xpoints+1, normal: 0, uv: i+xpoints+1},
                VertexIndex{vertex: i+1,         normal: 0, uv: i+1}
            ]);
            face_normals.push(normal);
            face_normals.push(normal);
        }

        let index = self.meshes.len();
        self.meshes.push(Mesh{index, vertices, vertex_normals, vertex_uvs, faces, face_normals});
        index
    }

    pub fn create_model(&mut self, mesh_index: usize, material_index: usize,
        model_matrix: Mat4) -> usize {
        let model_index = self.models.len();
        let mesh = self.get_mesh(mesh_index);
        self.models.push(Model{
            index: model_index,
            mesh_index: mesh_index,
            material_index: material_index,
            model_matrix: model_matrix,
            bounding_box: compute_aabb(mesh, model_matrix)
        });
        model_index
    }

    pub fn create_material(&mut self, material: Material) -> usize {
        let index = self.materials.len();
        self.materials.push(Material{
            index,
            diffuse_color: material.diffuse_color,
            specular_color: material.specular_color,
            ambient_coeff: material.ambient_coeff,
            diffuse_coeff: material.diffuse_coeff,
            specular_coeff: material.specular_coeff,
            specular_exp: material.specular_exp,
            diffuse_map_index: material.diffuse_map_index,
            specular_map_index: material.specular_map_index,
            normal_map_index: material.normal_map_index
        });
        index
    }

    pub fn create_point_light(&mut self, light: PointLight) -> usize {
        let index = self.point_lights.len();
        self.point_lights.push(PointLight{
            index,
            pos: light.pos,
            color: light.color,
            attenuation: light.attenuation
        });
        index
    }

    pub fn create_camera(&mut self, cam: Camera) -> usize {
        let index = self.cameras.len();
        self.cameras.push(Camera{
            index,
            pos: cam.pos,
            look: cam.look,
            up: cam.up,
            znear: cam.znear,
            zfar: cam.zfar,
            horizontal_fov: cam.horizontal_fov,
            aspect_ratio: cam.aspect_ratio
        });
        index
    }
}
