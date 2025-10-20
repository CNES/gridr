<div align="center">
<a target="_blank" href="https://gitlab.cnes.fr/gridr/gridr">
<picture>
  <img
    src="./doc/images/gridr_logo.svg"
    alt="GRIDR"
    width="40%"
  />
</picture>
</a>

<h4>Geometric and Radiometric Image Data Resampling</h4>


[![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![minimum rustc 1.80](https://img.shields.io/badge/rustc-1.80+-blue?logo=rust)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![minimum pyo3 0.21](https://img.shields.io/badge/pyo3-0.21+-green?logo=rust)](https://github.com/PyO3/pyo3)
[![Quality Gate Status](https://sonarqube.cnes.fr/api/project_badges/measure?project=cnes%3Agridr%3Agridr&metric=alert_status&token=sqb_9e3c65ce7f9784759937eea8a5aa1747b662948e)](https://sonarqube.cnes.fr/dashboard?id=cnes%3Agridr%3Agridr)
</div>


**GRIDR** is a library for resampling and filtering raster image data, designed for efficiency in both in-memory processing and I/O operations.

## Functional Scope & Features

### Core Capabilities
- **Grid-based Resampling**
    - Adapt raster data to a target geometry defined by a grid containing the coordinates of each target pixel in the source image geometry.
    - Supports both **full-resolution** and **under-sampled** resolution grids.
    - **Interpolation Methods** : Nearest neighbor, linear, cubic
    - **Mask Support**:
        - **Grid Masks**: Raster or sentinel values.
        - **Source Image Masks**: Raster, sentinel values, or vectorized geometry.
        - **Target Mask Production**: Generate masks for the target raster geometry.
- **Filtering**: Apply spatial filters in the frequency domain (e.g., low-pass filtering).
- **Mask Rasterization**: Convert vectorized geometry masks into a regular target raster geometry.
- **Optimized Workflows**: Reduce I/O overhead for large-scale processing.

### Function Types
1. **Elemental (Core) Functions**
    - Standalone operations for direct manipulation of in-memory data.
    - Ideal for custom processing pipelines and fine-grained control.
2. **Chained Functions**
    - Optimized sequences of operations to minimize I/O overhead.
    - Efficiently manage memory and CPU usage for large-scale processing.


## Technical Implementation

### Architecture
- **Python**: Core functionality and interface (not just for bindings).
- **Rust**: Performance-critical algorithms and heavy computations.
- **PyO3**: Used for seamless Python-Rust bindings.

### Key Technical Aspects
- **Rust Core Library**: Can be used independently in other Rust projects.
- **Python Integration**: Full-featured methods available in Python, not just bindings.
- **Optimized I/O**: Designed to handle large datasets efficiently.

## Getting Started
To install and use GRIDR, refer to the [online documentation](https://gridr.readthedocs.io/en/stable/)
