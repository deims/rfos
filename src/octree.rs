use std::collections::VecDeque;
use std::sync::Arc;
use crossbeam_channel::bounded;

use crate::math::*;
use crate::texture::*;
use crate::scene::*;
use crate::render::*;
use crate::rasterizer::*;

#[derive(Clone, Copy)]
struct OctreeObject {
    model_index: usize,
    face_index: usize
}

struct OctreeNode {
    aabb: AABB,
    child_indices: Option<[usize; 8]>,
    objects: Vec<OctreeObject>,
}

pub struct OctreeQueryResult {
    pub model_index: usize,
    pub face_index: usize,
    pub dist: f32,
    pub point: Vec3,
    pub barycentric: Vec3
}

impl OctreeQueryResult {
    pub fn new() -> Self {
        OctreeQueryResult {
            model_index: 0,
            face_index: 0,
            dist: f32::MAX,
            point: Vec3::zero(),
            barycentric: Vec3::zero()
        }
    }
}

pub struct Octree {
    scene: Arc<Scene>,
    leaf_capacity: usize,
    min_box_size: f32,
    root_index: usize,
    nodes: Vec<OctreeNode>,
}

impl Octree {
    pub fn new(scene: Arc<Scene>, leaf_capacity: usize, min_box_size: f32) -> Self {
        let aabb = scene.models.iter()
            .fold(AABB{min: Vec3::zero(), max: Vec3::zero()},
            |acc, m| AABB::merge(&acc, &m.bounding_box));
        let root = OctreeNode {aabb, child_indices: None, objects: Vec::new()};
        let mut octree = Self {
            scene: scene.clone(),
            leaf_capacity,
            min_box_size,
            root_index: 0,
            nodes: vec![root],
        };
        for model in &scene.models {
            let mesh = scene.get_mesh(model.mesh_index);
            for face_index in 0..mesh.faces.len() {
                octree.insert(model.index, face_index);
            }
        }
        octree
    }

    fn get_triangle(&self, obj: OctreeObject) -> Triangle {
        let model = self.scene.get_model(obj.model_index);
        let mesh = self.scene.get_mesh(model.mesh_index);
        let face = &mesh.faces[obj.face_index];
        Triangle::new(
            Vec3::transform_point(mesh.vertices[face[0].vertex], model.model_matrix),
            Vec3::transform_point(mesh.vertices[face[1].vertex], model.model_matrix),
            Vec3::transform_point(mesh.vertices[face[2].vertex], model.model_matrix)
        )
    }

    pub fn insert(&mut self, model_index: usize, face_index: usize) {
        self.insert_recursive(self.root_index, OctreeObject{model_index, face_index});
    }

    fn insert_recursive(&mut self, node_index: usize, obj: OctreeObject) {
        if let Some(child_indices) = self.nodes[node_index].child_indices {
            let tri = self.get_triangle(obj);
            for child_index in child_indices {
                let child_aabb = self.nodes[child_index].aabb;
                if tri.intersects_aabb(&child_aabb) {
                    self.insert_recursive(child_index, obj);
                }
            }
        } else {
            self.nodes[node_index].objects.push(obj);
            let node_aabb = self.nodes[node_index].aabb;
            let objcount = self.nodes[node_index].objects.len();
            if objcount > self.leaf_capacity && node_aabb.max_extent() > self.min_box_size {
                self.subdivide(node_index);
            }
        }
    }

    fn subdivide(&mut self, parent_index: usize) {
        let mut old_objects = Vec::<OctreeObject>::new();
        std::mem::swap(&mut self.nodes[parent_index].objects, &mut old_objects);

        let pc = self.nodes[parent_index].aabb.center();
        let pmin = self.nodes[parent_index].aabb.min;
        let pmax = self.nodes[parent_index].aabb.max;
        let mut child_indices = [0usize; 8];
        let child_aabbs = [
            AABB{min: pmin, max: pc},
            AABB{min: vec3![pc[0], pmin[1], pmin[2]], max: vec3![pmax[0], pc[1], pc[2]]},
            AABB{min: vec3![pmin[0], pc[1], pmin[2]], max: vec3![pc[0], pmax[1], pc[2]]},
            AABB{min: vec3![pc[0], pc[1], pmin[2]]  , max: vec3![pmax[0], pmax[1], pc[2]]},
            AABB{min: vec3![pmin[0], pmin[1], pc[2]], max: vec3![pc[0], pc[1], pmax[2]]},
            AABB{min: vec3![pc[0], pmin[1], pc[2]]  , max: vec3![pmax[0], pc[1], pmax[2]]},
            AABB{min: vec3![pmin[0], pc[1], pc[2]]  , max: vec3![pc[0], pmax[1], pmax[2]]},
            AABB{min: pc                            , max: pmax}
        ];
        for i in 0..8 {
            child_indices[i] = self.nodes.len();
            self.nodes.push(OctreeNode {
                aabb: child_aabbs[i],
                child_indices: None,
                objects: Vec::new()
            });
        }

        self.nodes[parent_index].child_indices = Some(child_indices);
        for obj in old_objects {
            let tri = self.get_triangle(obj);
            for i in 0..8 {
                let child_aabb = self.nodes[child_indices[i]].aabb;
                if tri.intersects_aabb(&child_aabb){
                    self.insert_recursive(child_indices[i], obj);
                }
            }
        }
    }

    fn closest_isect(&self, ray: &Ray, objects: &Vec<OctreeObject>) -> Option<OctreeQueryResult> {
        let mut isect_found = false;
        let mut qres = OctreeQueryResult::new();
        for obj in objects {
            let obj = *obj;
            let tri = self.get_triangle(obj);
            if let Some((tri_point, bc)) = ray.intersects_triangle(&tri) {
                isect_found = true;
                let d = Vec3::dist(ray.origin, tri_point);
                if d < qres.dist {
                    qres.model_index = obj.model_index;
                    qres.face_index = obj.face_index;
                    qres.dist = d;
                    qres.point = tri_point;
                    qres.barycentric = bc;
                }
            }
        }
        if isect_found {Some(qres)} else {None}
    }

    pub fn query(&self, ray: &Ray) -> Option<OctreeQueryResult> {
        let mut isect_found = false;
        let mut qres = OctreeQueryResult::new();
        let mut indexq = VecDeque::new();
        indexq.push_back(self.root_index);
        while !indexq.is_empty() {
            let node_index = indexq.pop_front().unwrap();
            if let Some(_) = ray.intersects_aabb(&self.nodes[node_index].aabb) {
                if let Some(child_indices) = self.nodes[node_index].child_indices {
                    child_indices.iter().for_each(|index| indexq.push_back(*index));
                } else {
                    let leaf = self.closest_isect(ray, &self.nodes[node_index].objects);
                    if let Some(lqres) = leaf {
                        isect_found = true;
                        if lqres.dist < qres.dist {
                            qres.model_index = lqres.model_index;
                            qres.face_index = lqres.face_index;
                            qres.dist = lqres.dist;
                            qres.point = lqres.point;
                            qres.barycentric = lqres.barycentric;
                        }
                    }
                }
            }
        }
        if isect_found {Some(qres)} else {None}
    }

    pub fn render_to_file(&self, config: RenderConfig, camera_index: usize, path: &str)
        -> std::result::Result<(), &str> {

        let (sender, receiver) = bounded(PRIM_QUEUE_CAP);
        let handle = FragmentProcessor::launch(None, config.clone(), camera_index, receiver);

        let camera = self.scene.get_camera(camera_index);
        let view_matrix = camera.view_matrix();
        let proj_matrix = camera.perspective_matrix();
        let view_proj_matrix = proj_matrix * view_matrix;
        let w = config.image_width as f32;
        let h = config.image_height as f32;
        let rasconfig = &config.rasterizer_config;
        let color = rgba_to_vec3(rasconfig.bounding_box_color);

        let mut indexq = VecDeque::<usize>::new();
        indexq.push_back(self.root_index);
        while !indexq.is_empty() {
            let node_index = indexq.pop_front().unwrap();
            let corners = self.nodes[node_index].aabb.corners_vec4();
            let mut bbox_frags = [Fragment::new(); 8];
            for i in 0..8 {
                bbox_frags[i].screen_pos = view_proj_matrix * corners[i];
                bbox_frags[i].screen_pos /= bbox_frags[i].screen_pos[3];
                bbox_frags[i].pixel_pos = vec2! [
                    lerp(0.0, w, -1.0, 1.0, bbox_frags[i].screen_pos[0]),
                    lerp(h, 0.0, -1.0, 1.0, bbox_frags[i].screen_pos[1])
                ];
                bbox_frags[i].color = color;
            }
            for i in 0..4 {
                let j = (i+1) % 4;
                _ = sender.send(RasterPrimitive::Line(bbox_frags[i], bbox_frags[j])).unwrap();
                _ = sender.send(RasterPrimitive::Line(bbox_frags[i+4], bbox_frags[j+4])).unwrap();
                _ = sender.send(RasterPrimitive::Line(bbox_frags[i], bbox_frags[i+4])).unwrap();
            }

            if let Some(child_indices) = self.nodes[node_index].child_indices {
                child_indices.into_iter().for_each(|index| indexq.push_back(index));
            }
        }
        drop(sender);
        let fragproc = handle.join().unwrap();
        _ = fragproc.color_buffer.save(path).unwrap();
        Ok(())
    }
}
