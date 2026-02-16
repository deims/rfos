# rfos: Hybrid Rust Renderer

A high-performance software renderer written from scratch in Rust, supporting both **rasterization** and **ray tracing**. The project is built without external graphics APIs (like OpenGL/Vulkan) to explore low-level graphics algorithms and multi-threaded performance.

## Key Features

* **Dual Mode:** Choose between a software-based rasterizer (`ras`) and a multi-threaded ray tracer (`ray`).
* **Octree Acceleration:** Hierarchical spatial partitioning for fast ray-object intersection and scene management.
* **Multithreading:** Parallelized rendering using thread pools and crossbeam channels for efficient workload distribution.
* **Phong Shading:** Implementation of the Phong reflection model with support for ambient, diffuse, and specular lighting.
* **Custom Math Library:** Built-in linear algebra for 3D transformations, matrix operations, and vector math.

## Quick Start

1. **Rasterize:** `cargo run --release -- ras`
2. **Ray trace:** `cargo run --release -- ray`

Outputs are saved as `render.png`.
