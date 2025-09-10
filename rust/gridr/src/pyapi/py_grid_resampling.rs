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
use numpy::{PyArray1, PyReadwriteArray1, PyArrayMethods, Element};

use crate::{assert_options_exclusive};
use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};
use crate::core::gx_grid_resampling::{array1_grid_resampling, NoCheckGridMeshValidator, MaskGridMeshValidator, InvalidValueGridMeshValidator};
use crate::core::interp::gx_array_view_interp::GxArrayViewInterpolator;
use crate::core::interp::gx_optimized_bicubic_kernel::{GxOptimizedBicubicInterpolator};
use crate::core::interp::gx_nearest_kernel::{GxNearestInterpolator};
use crate::core::interp::gx_linear_kernel::{GxLinearInterpolator};
use crate::core::gx_const::{F64_TOLERANCE};

use super::py_array::{PyArrayWindow2};

/// A Python-exposed enum representing different interpolation methods for array operations.
///
/// # Variants
/// - `Nearest`: Nearest neighbor interpolation method
/// - `Linear`: Linear interpolation method
/// - `OptimizedBicubic`: Optimized bicubic interpolation method
#[pyclass]
#[derive(Clone, Copy)]
pub enum PyInterpolatorType {
    /// Nearest neighbor interpolation variant
    Nearest,
    /// Linear interpolation variant
    Linear,
    /// Optimized bicubic interpolation variant
    OptimizedBicubic,
}

/// Performs grid resampling operations on 1D arrays using configurable interpolation methods.
///
/// This function applies various interpolation techniques (nearest neighbor, linear, and optimized bicubic)
/// to resample data from an input array onto a specified grid. It supports optional input and output masks,
/// grid validation through masks or nodata values, and windowing operations to process only specific regions
/// of the arrays.
///
/// The function wraps the `core::gx_grid_resampling::array1_grid_resampling` implementation and handles 
/// the complexity of managing different interpolation strategies and validation mechanisms through a unified
/// Python interface.
///
/// # Parameters
/// - `interp`: The interpolation method to use (Nearest, Linear, or OptimizedBicubic)
/// - `array_in`: Bound immutable reference to the input 1D array containing source data
/// - `array_in_shape`: Tuple `(depth, rows, cols)` defining the shape of the input array
/// - `grid_row`: Bound immutable reference to 1D array of row coordinates for output grid
/// - `grid_col`: Bound immutable reference to 1D array of column coordinates for output grid
/// - `grid_shape`: Tuple `(rows, cols)` defining the shape of the grid arrays
/// - `grid_resolution`: Tuple `(row_resolution, col_resolution)` specifying oversampling factors
/// - `array_out`: Bound mutable reference to the output 1D array where results are stored
/// - `array_out_shape`: Tuple `(depth, rows, cols)` defining the shape of the output array
/// - `nodata_out`: Value to use for nodata pixels in the output array
/// - `array_in_origin`: Optional tuple `(row_bias, col_bias)` for input array coordinate origin
/// - `array_in_mask`: Optional bound immutable reference to input validity mask
/// - `grid_mask`: Optional bound immutable reference to grid validity mask
/// - `grid_mask_valid_value`: Valid value in grid mask indicating valid nodes (required if grid_mask provided)
/// - `grid_nodata`: Optional nodata value for grid validation (mutually exclusive with grid_mask)
/// - `array_out_mask`: Optional bound mutable reference to output validity mask
/// - `grid_win`: Optional window specifying region of interest in grid coordinates
/// - `out_win`: Optional window specifying output region that will hold the resampled data
/// - `check_boundaries`: Boolean flag to enable/disable boundary checking - to be set to false with caution !
///
/// # Type Parameters
/// - `T`: Input array element type implementing Element + Copy + PartialEq + Default + Mul<f64, Output=f64> + Into<f64>
/// - `V`: Output array element type implementing Element + Copy + PartialEq + Default + From<f64>
/// - `W`: Grid coordinate type implementing Element + Copy + PartialEq + Default + Mul<f64, Output=f64> + Into<f64>
///
/// # Returns
/// - `Ok(())` if resampling completes successfully
/// - `Err(PyErr)` if resampling fails due to invalid parameters, computation errors, or internal issues
///
/// # Constraints
/// The function requires that either `grid_mask` or `grid_nodata` is provided (but not both) when
/// grid validation is needed. When `grid_mask` is provided, `grid_mask_valid_value` must also be specified.
///
/// # Behavior
/// - Supports three interpolation methods
/// - Handles optional input/output masks for data validity tracking
/// - Implements windowing support for processing sub-regions
/// - Applies boundary checking when enabled
///
/// # Panics
/// - If both `grid_mask` and `grid_nodata` are provided simultaneously (mutually exclusive)
/// - If `grid_mask_valid_value` is not provided when `grid_mask` is specified
///
/// # Errors
/// Returns a Python `ValueError` (`PyValueError`) if:
/// - Invalid parameter combinations are detected
/// - Grid validation parameters are missing or inconsistent
/// - Internal computation failures occur
/// - Memory allocation or access errors happen during array operations
///
/// # Example
/// ```python
/// from my_module import py_array1_grid_resampling, PyInterpolatorType
///
/// # Resample using linear interpolation with grid validation
/// py_array1_grid_resampling(
///     interp=PyInterpolatorType.Linear,
///     array_in=input_array,
///     array_in_shape=(1, 100, 100),
///     grid_row=grid_rows,
///     grid_col=grid_cols,
///     grid_shape=(10, 10),
///     grid_resolution=(1, 1),
///     array_out=output_array,
///     array_out_shape=(1, 10, 10),
///     nodata_out=-9999.0,
///     array_in_origin=None,
///     array_in_mask=None,
///     grid_mask=None,
///     grid_mask_valid_value=None,
///     grid_nodata=None,
///     array_out_mask=None,
///     grid_win=None,
///     out_win=None,
///     check_boundaries=True
/// )
/// ```
fn py_array1_grid_resampling<T, V, W>(
    interp: PyInterpolatorType,
    array_in: &Bound<'_, PyArray1<T>>,
    array_in_shape: (usize, usize, usize),
    grid_row: &Bound<'_, PyArray1<W>>,
    grid_col: &Bound<'_, PyArray1<W>>,
    grid_shape: (usize, usize),
    grid_resolution: (usize, usize),
    array_out: &Bound<'_, PyArray1<V>>,
    array_out_shape: (usize, usize, usize),
    nodata_out: V,
    array_in_origin: Option<(f64, f64)>,
    array_in_mask: Option<&Bound<'_, PyArray1<u8>>>,
    //grid_origin: (W, W),
    grid_mask: Option<&Bound<'_, PyArray1<u8>>>,
    grid_mask_valid_value: Option<u8>,
    grid_nodata: Option<W>,
    array_out_mask: Option<&Bound<'_, PyArray1<u8>>>,
    //array_out_win : Option<PyArrayWindow2>,
    //array_out_origin,
    grid_win: Option<PyArrayWindow2>,
    out_win: Option<PyArrayWindow2>,
    check_boundaries: bool,
    ) -> Result<(), PyErr>
where
    T: Element + Copy + PartialEq + Default + std::ops::Mul<f64, Output=f64> + Into<f64>,
    V: Element + Copy + PartialEq + Default + From<f64>,
    W: Element + Copy + PartialEq + Default + std::ops::Mul<f64, Output=f64> + Into<f64>,
{
    // Create a safe mutable array_view in order to be able to read and write
    // from/to the output array
    let mut array_out_view_mut = array_out.readwrite();
    let array_out_slice = array_out_view_mut.as_slice_mut().expect("Failed to get slice");
    let mut array_out_arrayview = GxArrayViewMut::new(array_out_slice, array_out_shape.0, array_out_shape.1, array_out_shape.2);
    
    // Create a safe immuable array_view in order to read from thein input array
    let array_in_view = array_in.readonly();
    let array_in_slice = array_in_view.as_slice()?;
    let array_in_arrayview = GxArrayView::new(array_in_slice, array_in_shape.0, array_in_shape.1, array_in_shape.2);
    
    // Create a safe immuable array_view in order to read from the grid - row - array
    let grid_row_view = grid_row.readonly();
    let grid_row_slice = grid_row_view.as_slice()?;
    let grid_row_arrayview = GxArrayView::new(grid_row_slice, 1, grid_shape.0, grid_shape.1);
    
    // Create a safe immuable array_view in order to read from the grid - col - array
    let grid_col_view = grid_col.readonly();
    let grid_col_slice = grid_col_view.as_slice()?;
    let grid_col_arrayview = GxArrayView::new(grid_col_slice, 1, grid_shape.0, grid_shape.1);
    
    // Manage the optional production window (in full resolution grid coordinates system)
    let rs_grid_win = grid_win.map(GxArrayWindow::from);
    
    // Manage the optional output window
    let rs_out_win = out_win.map(GxArrayWindow::from);
    
    // Manage array in origin
    let (origin_row_bias, origin_col_bias) = match array_in_origin {
        Some((row_bias, col_bias)) => (Some(row_bias), Some(col_bias)),
        None => (None, None),
    };
    
    // Prepare optional input validity mask to pass to the `array1_grid_resampling`
    // method.
    let mask_in_array_view: Option<GxArrayView<u8>>;
    let mask_in_view;
    mask_in_array_view = match array_in_mask {
        Some(a_mask_in_array_view) => {
            mask_in_view = a_mask_in_array_view.readonly();
            let mask_in_slice = mask_in_view.as_slice()?;
            Some(GxArrayView::new(mask_in_slice, 1, array_in_shape.1, array_in_shape.2))
        },
        None => {
            None
        }
    };
    
    // Prepare optional output validity mask to pass to the `array1_grid_resampling`
    // method.
    let mut mask_out_array_view: Option<GxArrayViewMut<u8>> = None;
    let mut mask_out_view: PyReadwriteArray1<u8>;
    //let mask_out_slice: &mut [u8];

    if let Some(bound_mask) = array_out_mask {
        mask_out_view = bound_mask.readwrite();
        let mask_out_slice = mask_out_view.as_slice_mut().expect("Failed to get slice");
        mask_out_array_view = Some(GxArrayViewMut::new(mask_out_slice, 1, array_out_shape.1, array_out_shape.2));
    }
    
    // Manage the grid validator mode through a the `grid_validator_flag` variable.
    // - 0 : that value corresponds to the use of a NoCheckGridMeshValidator, ie
    //       no mask has been provided by the caller
    // - 1 : that value corresponds to the use of a MaskGridMeshValidator, ie
    //       a raster mask has been provided.
    // - 2 : that value corresponds to the use of a InvalidValueGridMeshValidator, ie
    //       a grid nodata value has been provided.
    let mut grid_validator_flag : u8 = 0;
    //let grid_validity_checker = gx_grid_resampling::
    
    // Check exclusive parameters
    assert_options_exclusive!(grid_mask, grid_nodata, PyErr::new::<PyValueError, _>(
        "Only one of `grid_mask` or `grid_nodata` may be provided, not both."));
    
    let grid_mask_view;
    let grid_mask_array_view = match grid_mask {
        Some(a_mask_grid) => {
            grid_validator_flag += 1;
            grid_mask_view = a_mask_grid.readonly();
            let grid_mask_slice = grid_mask_view.as_slice()?;
            Some(GxArrayView::new(grid_mask_slice, 1, grid_shape.0, grid_shape.1))
        }
        None => None, 
    };
    // Get grid_nodata_value ; warning : if None a default value will be given
    let grid_nodata_value = grid_nodata.map(|val| {
        grid_validator_flag += 2;
        val.into()
    });
    
    match (interp, grid_validator_flag) {
        // ---- nearest ----
        (PyInterpolatorType::Nearest, 0) => {
            // No validator parameter has been passed ; we set the grid_checker to the always
            // positiv NoCheckGridMeshValidator
            let grid_checker = NoCheckGridMeshValidator{};
            let interp = GxNearestInterpolator::new();
            match array1_grid_resampling::<T, V, W, GxNearestInterpolator, NoCheckGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        (PyInterpolatorType::Nearest, 1) => {
            // A grid mask parameter has been passed ; we intialize a MaskGridMeshValidator
            let mask_view = grid_mask_array_view.unwrap();
            let mask_valid_value = grid_mask_valid_value.ok_or_else(|| PyValueError::new_err(
                    "The argument `grid_mask_valid_value` is mandatory when using `grid_mask`"
                ))?;
            let grid_checker = MaskGridMeshValidator{ mask_view: &mask_view, valid_value: mask_valid_value };
            let interp = GxNearestInterpolator::new();
            
            match array1_grid_resampling::<T, V, W, GxNearestInterpolator, MaskGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        (PyInterpolatorType::Nearest, 2) => {
            // A grid nodata value parameter has been passed ; we intialize an InvalidValueGridMeshValidator
            let grid_checker = InvalidValueGridMeshValidator{
                invalid_value: grid_nodata_value.expect("grid_nodata was None, but a value was expected"),
                epsilon: F64_TOLERANCE
            };
            let interp = GxNearestInterpolator::new();
            
            match array1_grid_resampling::<T, V, W, GxNearestInterpolator, InvalidValueGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        
        // ---- linear ----
        (PyInterpolatorType::Linear, 0) => {
            // No validator parameter has been passed ; we set the grid_checker to the always
            // positiv NoCheckGridMeshValidator
            let grid_checker = NoCheckGridMeshValidator{};
            let interp = GxLinearInterpolator::new();
            match array1_grid_resampling::<T, V, W, GxLinearInterpolator, NoCheckGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        (PyInterpolatorType::Linear, 1) => {
            // A grid mask parameter has been passed ; we intialize a MaskGridMeshValidator
            let mask_view = grid_mask_array_view.unwrap();
            let mask_valid_value = grid_mask_valid_value.ok_or_else(|| PyValueError::new_err(
                    "The argument `grid_mask_valid_value` is mandatory when using `grid_mask`"
                ))?;
            let grid_checker = MaskGridMeshValidator{ mask_view: &mask_view, valid_value: mask_valid_value };
            let interp = GxLinearInterpolator::new();
            
            match array1_grid_resampling::<T, V, W, GxLinearInterpolator, MaskGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        (PyInterpolatorType::Linear, 2) => {
            // A grid nodata value parameter has been passed ; we intialize an InvalidValueGridMeshValidator
            let grid_checker = InvalidValueGridMeshValidator{
                invalid_value: grid_nodata_value.expect("grid_nodata was None, but a value was expected"),
                epsilon: F64_TOLERANCE
            };
            let interp = GxLinearInterpolator::new();
            
            match array1_grid_resampling::<T, V, W, GxLinearInterpolator, InvalidValueGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        
        // ----- optimized bicubic -----
        (PyInterpolatorType::OptimizedBicubic, 0) => {
            // No validator parameter has been passed ; we set the grid_checker to the always
            // positiv NoCheckGridMeshValidator
            let grid_checker = NoCheckGridMeshValidator{};
            let interp = GxOptimizedBicubicInterpolator::new();
            match array1_grid_resampling::<T, V, W, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        (PyInterpolatorType::OptimizedBicubic, 1) => {
            // A grid mask parameter has been passed ; we intialize a MaskGridMeshValidator
            let mask_view = grid_mask_array_view.unwrap();
            let mask_valid_value = grid_mask_valid_value.ok_or_else(|| PyValueError::new_err(
                    "The argument `grid_mask_valid_value` is mandatory when using `grid_mask`"
                ))?;
            let grid_checker = MaskGridMeshValidator{ mask_view: &mask_view, valid_value: mask_valid_value };
            let interp = GxOptimizedBicubicInterpolator::new();
            
            match array1_grid_resampling::<T, V, W, GxOptimizedBicubicInterpolator, MaskGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        (PyInterpolatorType::OptimizedBicubic, 2) => {
            // A grid nodata value parameter has been passed ; we intialize an InvalidValueGridMeshValidator
            let grid_checker = InvalidValueGridMeshValidator{
                invalid_value: grid_nodata_value.expect("grid_nodata was None, but a value was expected"),
                epsilon: F64_TOLERANCE
            };
            let interp = GxOptimizedBicubicInterpolator::new();
            
            match array1_grid_resampling::<T, V, W, GxOptimizedBicubicInterpolator, InvalidValueGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    mask_in_array_view.as_ref(), //ima_mask_in
                    mask_out_array_view.as_mut(), //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                    rs_out_win.as_ref(), //out_win
                    origin_row_bias, // ima_in_origin_row
                    origin_col_bias, // ima_in_origin_col
                    check_boundaries, // check_boundaries
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        (_, _) => Err(PyValueError::new_err("Interpolator / Grid validator mode not implemented")),
    }
}

/// This function calls the generic [`py_array1_grid_resampling`] with `T = f64`, `V = f64` and `W = f64`.
#[pyfunction]
#[pyo3(signature = (interp, array_in, array_in_shape, grid_row, grid_col, grid_shape, grid_resolution, array_out, array_out_shape, nodata_out, array_in_origin=None, array_in_mask=None, grid_mask=None, grid_mask_valid_value=None, grid_nodata=None, array_out_mask=None, grid_win=None, out_win=None, check_boundaries=true))]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_grid_resampling_f64(
    interp: PyInterpolatorType,
    array_in: &Bound<'_, PyArray1<f64>>,
    array_in_shape: (usize, usize, usize),
    grid_row: &Bound<'_, PyArray1<f64>>,
    grid_col: &Bound<'_, PyArray1<f64>>,
    grid_shape: (usize, usize),
    grid_resolution: (usize, usize),
    array_out: &Bound<'_, PyArray1<f64>>,
    array_out_shape: (usize, usize, usize),
    nodata_out: f64,
    array_in_origin: Option<(f64, f64)>,
    array_in_mask: Option<&Bound<'_, PyArray1<u8>>>,
    //grid_origin: (W, W),
    grid_mask: Option<&Bound<'_, PyArray1<u8>>>,
    grid_mask_valid_value: Option<u8>,
    grid_nodata: Option<f64>,
    array_out_mask: Option<&Bound<'_, PyArray1<u8>>>,
    //array_out_win : Option<PyArrayWindow2>,
    //array_out_origin,
    grid_win: Option<PyArrayWindow2>,
    out_win: Option<PyArrayWindow2>,
    check_boundaries: bool,
    ) -> Result<(), PyErr>
{
    py_array1_grid_resampling::<f64, f64, f64>(
            interp, // PyInterpolatorType
            array_in, //: &Bound<'_, PyArray1<f64>>,
            array_in_shape, //: (usize, usize, usize),
            grid_row, //: &Bound<'_, PyArray1<f64>>,
            grid_col, //: &Bound<'_, PyArray1<f64>>,
            grid_shape, //: (usize, usize),
            grid_resolution, //: (usize, usize),
            array_out, //: &Bound<'_, PyArray1<f64>>,
            array_out_shape, //: (usize, usize, usize),
            nodata_out, //: f64,
            array_in_origin, //: Option<(f64, f64)>,
            array_in_mask, //: Option<&Bound<'_, PyArray1<u8>>>,
            //grid_origin: (W, W),
            grid_mask, //: Option<&Bound<'_, PyArray1<u8>>>,
            grid_mask_valid_value, //: Option<u8>,
            grid_nodata, //: Option<W>,
            array_out_mask, //: Option<&Bound<'_, PyArray1<u8>>>,
            //array_out_win : Option<PyArrayWindow2>,
            //array_out_origin,
            grid_win, // grid_win: Option<PyArrayWindow2>,
            out_win, // win_out: Option<PyArrayWindow2>,
            check_boundaries, //check_boundaries
            )
}
