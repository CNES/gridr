#![warn(missing_docs)]
//! Crate doc
use pyo3::prelude::*;
//use ndarray;
//use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods};
//use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use numpy::{PyArray1, PyArrayMethods};
/// We tell here what module/functions we use from the pure rust library (lib.rs)
use crate::core::array_utils;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _libgridr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_array1_replace_i8,m)?)?;
    m.add_function(wrap_pyfunction!(py_array1_replace_u8,m)?)?;
    Ok(())
}

/// Function py_array1_replace_i8
/// Wrapper to the core pure rust method array1_replace - i8 typed
/// This methods replaces values in a 1d array by applying the following rule:
/// - if an element equals to 'val_cond' then the element is set to 'val_true'
/// - otherwise the element is set to 'val_false'
/// This methods has been implemented to respond to a lack of python's numpy
/// methods (where, putmask copyto) that allocate temporary memory.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn py_array1_replace_i8(
    array: &Bound<'_, PyArray1<i8>>,
    val_cond: i8,
    val_true: i8,
    val_false: i8) -> Result<(), PyErr>
{
    // Create a safe mutable array_view in order to be able to read and write
    // from/to the input array
    let mut array_view = array.readwrite();
    // The wrapped method use rust standard mutable slice, so we have to first
    // get it here frow the array_view.
    let slice = array_view.as_slice_mut()?;
    // Call the wrapped method
    array_utils::array1_replace(slice, val_cond, val_true, val_false);
    Ok(())
}


/// Function py_array1_replace_u8
/// Wrapper to the core pure rust method array1_replace - u8 typed
/// This methods replaces values in a 1d array by applying the following rule:
/// - if an element equals to 'val_cond' then the element is set to 'val_true'
/// - otherwise the element is set to 'val_false'
/// This methods has been implemented to respond to a lack of python's numpy
/// methods (where, putmask copyto) that allocate temporary memory.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn py_array1_replace_u8(
    array: &Bound<'_, PyArray1<u8>>,
    val_cond: u8,
    val_true: u8,
    val_false: u8) -> Result<(), PyErr>
{
    // Create a safe mutable array_view in order to be able to read and write
    // from/to the input array
    let mut array_view = array.readwrite();
    // The wrapped method use rust standard mutable slice, so we have to first
    // get it here frow the array_view.
    let slice = array_view.as_slice_mut()?;
    // Call the wrapped method
    array_utils::array1_replace(slice, val_cond, val_true, val_false);
    Ok(())
}
