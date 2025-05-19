#![warn(missing_docs)]
//! Crate doc
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
//use ndarray;
//use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
//use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use numpy::{PyArray1, PyArrayMethods, Element};
/// We tell here what module/functions we use from the pure rust library (lib.rs)
use crate::{assert_options_match, assert_options_exclusive};
use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};
use crate::core::gx_grid_resampling::{array1_grid_resampling, GridMeshValidator, NoCheckGridMeshValidator, MaskGridMeshValidator, InvalidValueGridMeshValidator};
use crate::core::interp::gx_array_view_interp::GxArrayViewInterpolator;
use crate::core::interp::gx_optimized_bicubic_kernel::{GxOptimizedBicubicInterpolator};

use super::py_array::{PyArrayWindow2};

pub const F64_TOLERANCE: f64 = 1e-5;

fn py_array1_grid_resampling<T, U, V, W>(
    array_in: &Bound<'_, PyArray1<T>>,
    array_in_shape: (usize, usize, usize),
    grid_row: &Bound<'_, PyArray1<W>>,
    grid_col: &Bound<'_, PyArray1<W>>,
    grid_shape: (usize, usize),
    grid_resolution: (usize, usize),
    array_out: &Bound<'_, PyArray1<V>>,
    array_out_shape: (usize, usize, usize),
    nodata_out: V,
    //array_in_origin: (usize, usize), ==> take care of that by the caller in the grid ?
    array_in_mask: Option<&Bound<'_, PyArray1<U>>>,
    //grid_origin: (W, W),
    grid_mask: Option<&Bound<'_, PyArray1<u8>>>,
    grid_mask_valid_value: Option<u8>,
    grid_nodata: Option<W>,
    array_out_mask: Option<&Bound<'_, PyArray1<i8>>>,
    //array_out_win : Option<PyArrayWindow2>,
    //array_out_origin,
    grid_win: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
where
    T: Element + Copy + PartialEq + Default + std::ops::Mul<f64, Output=f64> + Into<f64>,
    U: Element + Copy + PartialEq + Default + Into<f64>,
    V: Element + Copy + PartialEq + Default + From<f64>,
    W: Element + Copy + PartialEq + Default + std::ops::Mul<f64, Output=f64> + Into<f64>,
{
    // Create the interpolator (bicubic forced here for now)
    let interp = GxOptimizedBicubicInterpolator::new();
    
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
    
    match grid_validator_flag {
        0 => {
            // No validator parameter has been passed ; we set the grid_checker to the always
            // positiv NoCheckGridMeshValidator
            let grid_checker = NoCheckGridMeshValidator{};
            match array1_grid_resampling::<T, U, V, W, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    None, //ima_mask_in
                    None, //grid_mask_array
                    &mut None, //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        1 => {
            // A grid mask parameter has been passed ; we intialize a MaskGridMeshValidator
            let mask_view = grid_mask_array_view.unwrap();
            let mask_valid_value = grid_mask_valid_value.ok_or_else(|| PyValueError::new_err(
                    "The argument `grid_mask_valid_value` is mandatory when using `grid_mask`"
                ))?;
            let grid_checker = MaskGridMeshValidator{ mask_view: &mask_view, valid_value: mask_valid_value };
            
            match array1_grid_resampling::<T, U, V, W, GxOptimizedBicubicInterpolator, MaskGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    None, //ima_mask_in
                    None, //grid_mask_array
                    &mut None, //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        2 => {
            // A grid nodata value parameter has been passed ; we intialize an InvalidValueGridMeshValidator
            let grid_checker = InvalidValueGridMeshValidator{
                invalid_value: grid_nodata_value.expect("grid_nodata was None, but a value was expected"),
                epsilon: F64_TOLERANCE
            };
            
            match array1_grid_resampling::<T, U, V, W, GxOptimizedBicubicInterpolator, InvalidValueGridMeshValidator>(
                    &interp, // interp
                    &grid_checker, //
                    &array_in_arrayview, //ima_in
                    &grid_row_arrayview, //grid_row_array
                    &grid_col_arrayview, //grid_col_array
                    grid_resolution.0, //grid_row_oversampling
                    grid_resolution.1, //grid_col_oversampling
                    &mut array_out_arrayview, //ima_out
                    nodata_out, //nodata_val_out
                    None, //ima_mask_in
                    None, //grid_mask_array
                    &mut None, //ima_mask_out
                    rs_grid_win.as_ref(), //grid_win
                ) {
                Ok(_) => Ok(()),
                Err(e) => Err(PyValueError::new_err(e.to_string())),
            }
        },
        _ => Err(PyValueError::new_err("Grid validator mode not implemented")),
    }
}

/// This function calls the generic [`py_array1_grid_resampling`] with `T = f64`, `U = u8`, `V = f64` and `W = f64`.
#[pyfunction]
#[pyo3(signature = (array_in, array_in_shape, grid_row, grid_col, grid_shape, grid_resolution, array_out, array_out_shape, nodata_out, array_in_mask=None, grid_mask=None, grid_mask_valid_value=None, grid_nodata=None, array_out_mask=None, grid_win=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_grid_resampling_f64_u8(
    array_in: &Bound<'_, PyArray1<f64>>,
    array_in_shape: (usize, usize, usize),
    grid_row: &Bound<'_, PyArray1<f64>>,
    grid_col: &Bound<'_, PyArray1<f64>>,
    grid_shape: (usize, usize),
    grid_resolution: (usize, usize),
    array_out: &Bound<'_, PyArray1<f64>>,
    array_out_shape: (usize, usize, usize),
    nodata_out: f64,
    //array_in_origin: (usize, usize), ==> take care of that by the caller in the grid ?
    array_in_mask: Option<&Bound<'_, PyArray1<u8>>>,
    //grid_origin: (W, W),
    grid_mask: Option<&Bound<'_, PyArray1<u8>>>,
    grid_mask_valid_value: Option<u8>,
    grid_nodata: Option<f64>,
    array_out_mask: Option<&Bound<'_, PyArray1<i8>>>,
    //array_out_win : Option<PyArrayWindow2>,
    //array_out_origin,
    grid_win: Option<PyArrayWindow2>,
    ) -> Result<(), PyErr>
{
    py_array1_grid_resampling::<f64, u8, f64, f64>(
            array_in, //: &Bound<'_, PyArray1<f64>>,
            array_in_shape, //: (usize, usize, usize),
            grid_row, //: &Bound<'_, PyArray1<f64>>,
            grid_col, //: &Bound<'_, PyArray1<f64>>,
            grid_shape, //: (usize, usize),
            grid_resolution, //: (usize, usize),
            array_out, //: &Bound<'_, PyArray1<f64>>,
            array_out_shape, //: (usize, usize, usize),
            nodata_out, //: f64,
            //array_in_origin: (usize, usize), ==> take care of that by the caller in the grid ?
            array_in_mask, //: Option<&Bound<'_, PyArray1<u8>>>,
            //grid_origin: (W, W),
            grid_mask, //: Option<&Bound<'_, PyArray1<u8>>>,
            grid_mask_valid_value, //: Option<u8>,
            grid_nodata, //: Option<W>,
            array_out_mask, //: Option<&Bound<'_, PyArray1<i8>>>,
            //array_out_win : Option<PyArrayWindow2>,
            //array_out_origin,
            grid_win, // grid_win: Option<PyArrayWindow2>,
            )
}
