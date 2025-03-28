#![warn(missing_docs)]
//! Crate doc
use pyo3::prelude::*;

use crate::pyapi::py_array;
use crate::pyapi::py_array_utils;
use crate::pyapi::py_grid_resampling;


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _libgridr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    
    m.add_class::<py_array::PyArrayWindow2>()?;
    
    // Add from py_array_utils
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_i8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f32_i8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f64_i8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_u8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f32_u8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array_utils::py_array1_replace_f64_u8,m)?)?;
    
    // Add from py_grid_resampling
    m.add_function(wrap_pyfunction!(py_grid_resampling::py_array1_grid_resampling_f64_u8,m)?)?;
    
    Ok(())

}