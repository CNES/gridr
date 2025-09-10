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
use pyo3::exceptions::PyValueError;
//use ndarray;
//use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
//use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use numpy::{PyArray1, PyArrayMethods, Element};
/// We tell here what module/functions we use from the pure rust library (lib.rs)
use crate::core::gx_array_utils;
use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};

use super::py_array::{PyArrayWindow2};

/// Replaces values in a 1D NumPy array based on a condition and an optional window constraint.
///
/// This function operates on a mutable NumPy array (`array`) and replaces values based on a
/// comparison with `val_cond`. If a matching condition is met, the value is replaced by
/// `val_true`, otherwise it is replaced by `val_false`.
///
/// Optionally, a secondary condition can be applied using `array_cond` and `array_cond_val`.
/// If provided, values in `array` will be replaced based on whether the corresponding index
/// in `array_cond` matches `array_cond_val`.
///
/// Furthermore, an optional `window` parameter can be specified to restrict the operation
/// to a specific sub-region of the array.
///
/// This function is not added to the pymodule as it uses generic types.
///
/// # Parameters
/// - `array`: A mutable reference to a 1D NumPy array (`PyArray1<T>`) that will be modified.
/// - `val_cond`: The value to compare against elements in `array`.
/// - `val_true`: The value to assign when the condition is met.
/// - `val_false`: An optional value used when the condition is not met. If `None`, the existing array value is preserved.
/// - `array_cond`: An optional secondary condition array (`PyArray1<C>`).
/// - `array_cond_val`: An optional value to compare against `array_cond`.
/// - `window`: An optional `PyArrayWindow2` defining the sub-region of `array` to modify.
///
/// # Type Parameters
/// - `T`: The data type of the `array` elements (must implement `Copy` and `PartialEq`).
/// - `C`: The data type of the `array_cond` elements (must implement `Copy` and `PartialEq`).
///
/// # Errors
/// - Returns a `PyValueError` if any internal operation fails (e.g., array shape mismatch).
///
/// # Example (Python)
/// ```python
/// import numpy as np
/// from my_module import py_array1_replace, PyArrayWindow2
///
/// arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
/// cond_arr = np.array([0, 1, 0, 1, 0], dtype=np.int8)
///
/// # Replace values where arr == 3 with 10, otherwise set to 20
/// py_array1_replace(arr, 3.0, 10.0, 20.0, None, None, None)
///
/// # Apply a condition from cond_arr where cond_arr == 1
/// py_array1_replace(arr, 3.0, 10.0, 20.0, cond_arr, 1, None)
///
/// # Restrict the operation to a windowed region
/// window = PyArrayWindow2(0, 0, 4, 4, 2, 2)  # Define a sub-region
/// py_array1_replace(arr, 3.0, 10.0, 20.0, cond_arr, 1, window)
/// ```
fn py_array1_replace<T,C>(
    array: &Bound<'_, PyArray1<T>>,
    nrow: usize,
    ncol: usize,
    val_cond: T,
    val_true: T,
    val_false: Option<T>,
    array_cond: Option<&Bound<'_, PyArray1<C>>>,
    array_cond_val: Option<C>,
    window: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
where
    T: Element + Copy + PartialEq + Default,
    C: Element + Copy + PartialEq,
{
    // Create a safe mutable array_view in order to be able to read and write
    // from/to the input array
    let mut array_view_mut = array.readwrite();
    let array_slice = array_view_mut.as_slice_mut().expect("Failed to get slice");
    let mut gx_array_view_mut = GxArrayViewMut::new(array_slice, 1, nrow, ncol);
    
    // Handle optional array_cond
    let cond_array_view;
    let cond_view = match array_cond {
        Some(cond_array) => {
            cond_array_view = cond_array.readonly();
            let cond_slice = cond_array_view.as_slice()?; // Récupère le slice immuable
            Some(GxArrayView::new(cond_slice, 1, nrow, ncol)) // Adapter nvar, nrow, ncol si besoin
        }
        None => None, 
    };
    
    let rust_window = window.map(GxArrayWindow::from);
    
    if let Some(win) = rust_window.as_ref() {
        // Call the wrapped method
        match gx_array_utils::array1_replace_win2::<T, C>(&mut gx_array_view_mut, win,
                val_cond, val_true, val_false, cond_view.as_ref(), array_cond_val) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
    else {
        // Call the wrapped method
        match gx_array_utils::array1_replace::<T, C>(&mut gx_array_view_mut, val_cond, val_true, val_false, cond_view.as_ref(), array_cond_val) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }

}

/// Specialized function to replace values in a 1D NumPy array of type `i8`, with an optional condition array of type `i8`.
/// # Implementation Details
/// This function calls the generic [`py_array1_replace`] with `T = i8` and `C = i8`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_replace_i8(
    array: &Bound<'_, PyArray1<i8>>,
    nrow: usize,
    ncol: usize,
    val_cond: i8,
    val_true: i8,
    val_false: Option<i8>,
    array_cond: Option<&Bound<'_, PyArray1<i8>>>,
    array_cond_val: Option<i8>,
    window: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
{
    py_array1_replace::<i8, i8>(array, nrow, ncol, val_cond, val_true, val_false, array_cond, array_cond_val, window)
}

/// Specialized function to replace values in a 1D NumPy array of type `f32`, with an optional condition array of type `i8`.
/// # Implementation Details
/// This function calls the generic [`py_array1_replace`] with `T = f32` and `C = i8`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_replace_f32_i8(
    array: &Bound<'_, PyArray1<f32>>,
    nrow: usize,
    ncol: usize,
    val_cond: f32,
    val_true: f32,
    val_false: Option<f32>,
    array_cond: Option<&Bound<'_, PyArray1<i8>>>,
    array_cond_val: Option<i8>,
    window: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
{
    py_array1_replace::<f32, i8>(array, nrow, ncol, val_cond, val_true, val_false, array_cond, array_cond_val, window)
}

/// Specialized function to replace values in a 1D NumPy array of type `f64`, with an optional condition array of type `i8`.
/// # Implementation Details
/// This function calls the generic [`py_array1_replace`] with `T = f64` and `C = i8`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_replace_f64_i8(
    array: &Bound<'_, PyArray1<f64>>,
    nrow: usize,
    ncol: usize,
    val_cond: f64,
    val_true: f64,
    val_false: Option<f64>,
    array_cond: Option<&Bound<'_, PyArray1<i8>>>,
    array_cond_val: Option<i8>,
    window: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
{
    py_array1_replace::<f64, i8>(array, nrow, ncol, val_cond, val_true, val_false, array_cond, array_cond_val, window)
}

/// Specialized function to replace values in a 1D NumPy array of type `u8`, with an optional condition array of type `u8`.
/// # Implementation Details
/// This function calls the generic [`py_array1_replace`] with `T = u8` and `C = u8`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_replace_u8(
    array: &Bound<'_, PyArray1<u8>>,
    nrow: usize,
    ncol: usize,
    val_cond: u8,
    val_true: u8,
    val_false: Option<u8>,
    array_cond: Option<&Bound<'_, PyArray1<u8>>>,
    array_cond_val: Option<u8>,
    window: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
{
    py_array1_replace::<u8, u8>(array, nrow, ncol, val_cond, val_true, val_false, array_cond, array_cond_val, window)
}

/// Specialized function to replace values in a 1D NumPy array of type `f32`, with an optional condition array of type `u8`.
/// # Implementation Details
/// This function calls the generic [`py_array1_replace`] with `T = 32` and `C = u8`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_replace_f32_u8(
    array: &Bound<'_, PyArray1<f32>>,
    nrow: usize,
    ncol: usize,
    val_cond: f32,
    val_true: f32,
    val_false: Option<f32>,
    array_cond: Option<&Bound<'_, PyArray1<u8>>>,
    array_cond_val: Option<u8>,
    window: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
{    
    py_array1_replace::<f32, u8>(array, nrow, ncol, val_cond, val_true, val_false, array_cond, array_cond_val, window)
}

/// Specialized function to replace values in a 1D NumPy array of type `f64`, with an optional condition array of type `u8`.
/// # Implementation Details
/// This function calls the generic [`py_array1_replace`] with `T = 64` and `C = u8`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_replace_f64_u8(
    array: &Bound<'_, PyArray1<f64>>,
    nrow: usize,
    ncol: usize,
    val_cond: f64,
    val_true: f64,
    val_false: Option<f64>,
    array_cond: Option<&Bound<'_, PyArray1<u8>>>,
    array_cond_val: Option<u8>,
    window: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
{
    py_array1_replace::<f64, u8>(array, nrow, ncol, val_cond, val_true, val_false, array_cond, array_cond_val, window)
}


