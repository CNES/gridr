<div align="center">
<a target="_blank" href="https://gitlab.cnes.fr/gridr/gridr">
<picture>
  <img
    src="./doc/images/gridr.png"
    alt="GRIDR"
    width="40%"
  />
</picture>
</a>

<h4>Geometric and Radiometric Image Data Resampling</h4>


[![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![minimum rustc 1.79](https://img.shields.io/badge/rustc-1.79+-blue?logo=rust)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![minimum pyo3 0.21](https://img.shields.io/badge/pyo3-0.21+-green?logo=rust)](https://github.com/PyO3/pyo3)
[![Quality Gate Status](https://sonarqube.cnes.fr/api/project_badges/measure?project=cnes%3Agridr%3Agridr&metric=alert_status&token=sqb_9e3c65ce7f9784759937eea8a5aa1747b662948e)](https://sonarqube.cnes.fr/dashboard?id=cnes%3Agridr%3Agridr)
</div>

- [Context](#context)
- [Installation](#installation) 
- [Usage](#usage) 
- [Links](#links)

## Context
GRIDR is a tool to resample and filter image data raster.

GRIDR is a python/rust project : rust is used to implement computational heavy algorithm.

GRIDR provides both elemental functions that can directly be used with in memory data and chain functions that aim to optimize the I/O operations.

Functionnalities :
- spatial resampling (not yet implemented) :
    - grid based :
        - linear oversampling of the grid coordinates
        - exact radiometric interpolation with internal functions (nearest neighbor, linear, cubic, spline 
        - radiometric interpolation with internal tabulated functions
        - radiometric interpolation using an external kernel
        - usage of mask and nodata values
    - zoom
    - unzoom
- frequential filtering
        
## Installation

```
pip install gridr
``` 

## User documentation 

The user documentation can be found [here](https://gridr.pages.cnes.fr/gridr/)



## Links

TO BE DONE
