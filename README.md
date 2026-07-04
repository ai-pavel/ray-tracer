# Ray Tracer

[![CI](https://github.com/pavel-genai/ray-tracer/actions/workflows/ci.yml/badge.svg)](https://github.com/pavel-genai/ray-tracer/actions/workflows/ci.yml)

A path tracer written in Rust featuring BVH acceleration, multiple material types, and multi-threaded rendering via Rayon.

## Features

- Ray-sphere and ray-plane intersection
- Bounding Volume Hierarchy (BVH) for acceleration
- Materials: Lambertian (diffuse), Metal (reflective with fuzz), Dielectric (glass with refraction)
- Multi-threaded rendering using Rayon
- Output to PPM and PNG formats
- Configurable resolution, samples per pixel, and max bounce depth via CLI

## Usage

```bash
cargo run --release -- --width 800 --height 450 --samples 100 --depth 50 --output render.png
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--width` | `-W` | 800 | Image width in pixels |
| `--height` | `-H` | 450 | Image height in pixels |
| `--samples` | `-s` | 100 | Samples per pixel |
| `--depth` | `-d` | 50 | Maximum ray bounce depth |
| `--output` | `-o` | output.png | Output file path (.png or .ppm) |

## Project Structure

```
src/
  main.rs       - CLI entry point and rendering loop
  ray.rs        - Ray and Vec3 types
  hittable.rs   - Hit record, Sphere, Plane, HittableList
  bvh.rs        - Bounding Volume Hierarchy
  material.rs   - Lambertian, Metal, Dielectric materials
  camera.rs     - Camera with defocus blur
  scene.rs      - Demo scene construction
```

## Building

```bash
cargo build --release
```
