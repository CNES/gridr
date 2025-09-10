// Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
//
// This file is part of GRIDR
// (see https://github.com/CNES/gridr).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![warn(missing_docs)]
//! Crate doc
use pyo3::prelude::*;

use crate::pyapi::py_array;
use crate::pyapi::py_array_utils;
use crate::pyapi::py_grid_resampling;
use crate::pyapi::py_grid_geometry;


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _libgridr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    
    // Add classes/structures from py_array
    m.add_class::<py_array::PyArrayWindow2>()?;
    
    // Add from py_array_utils
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_i8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f32_i8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f64_i8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_u8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f32_u8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f64_u8,m)?)?;
    
    // Add from py_grid_resampling
    m.add_class::<py_grid_resampling::PyInterpolatorType>()?;
    m.add_function(wrap_pyfunction!(py_grid_resampling::py_array1_grid_resampling_f64,m)?)?;
    
    // Add from py_grid_geometry
    m.add_class::<py_grid_geometry::PyGridTransitionMatrix>()?;
    m.add_class::<py_grid_geometry::PyGeometryBoundsUsize>()?;
    m.add_class::<py_grid_geometry::PyGeometryBoundsF64>()?;
    m.add_class::<py_grid_geometry::PyGridGeometriesMetricsF64>()?;
    m.add_function(wrap_pyfunction!(py_grid_geometry::py_array1_compute_resampling_grid_geometries_f64_f64,m)?)?;
    
    Ok(())

}