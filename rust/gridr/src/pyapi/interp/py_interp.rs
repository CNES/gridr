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
//! # Common Python-exposed Module
//!
//! This module provides a common interface for various interpolation methods.
//! It bridges Rust-based interpolation implementations with Python bindings.
//!
//! The module supports multiple interpolation techniques including nearest
//! neighbor, linear, bicubic, and B-spline interpolation with various orders.
//! These implementations are optimized for performance and thread safety.
//!
//! ## Key Features
//! - **Unified Interface**: The `AnyInterpolator` enum provides a type-safe way
//!   to handle different interpolation methods through a single interface.
//! - **Python Bindings**: The module includes Python-compatible classes and
//!   conversion utilities for seamless integration with Python code.
//! - **Thread Safety**: All interpolator implementations use `Arc` (Atomic
//!   Reference Counting) for thread-safe shared ownership.
//! - **Performance Optimized**: The Rust-based implementations are designed for
//!   efficient array operations.
//!
//! ## Usage
//!
//! The module is mainly designed to be used both in Rust code and through Python
//! bindings, but it can also be used in pure Rust code. The `FromPyObject` 
//! implementation allows easy conversion from Python objects to Rust 
//! interpolator instances.
//!
//! ## Dependencies
//!
//! - `pyo3`: For Python bindings and interoperability
//! - `std::sync::Arc`: For thread-safe reference counting
//!
//! ## Compatibility
//!
//! This implementation requires `pyo3` version 0.27 or higher.
//!
//! ## Examples
//!
//! ```rust
//! // Example of creating and using an interpolator in Rust
//! use std::sync::Arc;
//! use interpolation_module::{AnyInterpolator, GxNearestInterpolator};
//!
//! let nearest_interp = AnyInterpolator::Nearest(Arc::new(GxNearestInterpolator::new()));
//! ```
//!
//! ```python
//! # Example of using an interpolator through Python bindings
//! from gridr.cdylib import PyNearestInterpolator
//!
//! interp = PyNearestInterpolator()
//! ```
use std::sync::{Arc, RwLock};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use crate::core::interp::gx_optimized_bicubic_kernel::GxOptimizedBicubicInterpolator;
use crate::core::interp::gx_bspline_kernel::GxBSplineInterpolator;
use crate::core::interp::gx_linear_kernel::GxLinearInterpolator;
use crate::core::interp::gx_nearest_kernel::GxNearestInterpolator;

use crate::pyapi::interp::py_bspline_kernel::{
    PyBSpline3Interpolator,
    PyBSpline5Interpolator,
    PyBSpline7Interpolator,
    PyBSpline9Interpolator,
    PyBSpline11Interpolator
};
use crate::pyapi::interp::py_nearest_kernel::PyNearestInterpolator;
use crate::pyapi::interp::py_linear_kernel::PyLinearInterpolator;
use crate::pyapi::interp::py_optimized_bicubic_kernel::PyOptimizedBicubicInterpolator;

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
    /// Cubic Cardinal BSpline
    BSpline3,
    /// Quintic Cardinal BSpline
    BSpline5,
    /// Septic Cardinal BSpline
    BSpline7,
    /// Nonic Cardinal BSpline
    BSpline9,
    /// Eleventh order Cardinal BSpline
    BSpline11,
}


/// Enum representing different types of interpolators in a unified interface.
///
/// This enum provides a type-safe way to handle various interpolation methods
/// through a single interface. It supports different interpolation techniques
/// including nearest neighbor, linear, bicubic, and B-spline interpolation
/// with various orders.
///
/// The enum variants wrap different interpolator implementations using Arc
/// (Atomic Reference Counting) for thread-safe shared ownership.
///
/// This enum implements the `FromPyObject` trait to enable creation of Rust
/// interpolators from Python-side interpolators.
pub enum AnyInterpolator {
    /// Nearest neighbor interpolation variant ARC wrapper
    Nearest(Arc<RwLock<GxNearestInterpolator>>),
    /// Linear interpolation variant ARC wrapper
    Linear(Arc<RwLock<GxLinearInterpolator>>),
    /// Optimized bicubic interpolation variant ARC wrapper
    OptimizedBicubic(Arc<RwLock<GxOptimizedBicubicInterpolator>>),
    /// Cubic Cardinal BSpline variant ARC wrapper
    BSpline3(Arc<RwLock<GxBSplineInterpolator<3>>>),
    /// Quintic Cardinal BSpline variant ARC wrapper
    BSpline5(Arc<RwLock<GxBSplineInterpolator<5>>>),
    /// Septic Cardinal BSpline variant ARC wrapper
    BSpline7(Arc<RwLock<GxBSplineInterpolator<7>>>),
    /// Nonic Cardinal BSpline variant ARC wrapper
    BSpline9(Arc<RwLock<GxBSplineInterpolator<9>>>),
    /// Eleventh order Cardinal BSpline variant ARC wrapper
    BSpline11(Arc<RwLock<GxBSplineInterpolator<11>>>),
}


/// Implementation of FromPyObject trait for AnyInterpolator.
/// This allows conversion from Python objects to Rust AnyInterpolator instances.
/// The implementation handles all supported interpolators and converts them
/// to their corresponding Rust representations.
///
/// # Notes
/// This trait implementation is aimed to be used in the implementations of 
/// pyfunctions.
/// This implementation requires pyo3>=0.27
impl<'a, 'py> FromPyObject<'a, 'py> for AnyInterpolator {
    type Error = PyErr;
    
    /// Extracts an AnyInterpolator from a Python object.
    ///
    /// This method attempts to convert a Python object into an AnyInterpolator
    /// by checking against various known interpolator types. It supports:
    /// - Nearest neighbor interpolation
    /// - Linear interpolation
    /// - Optimized bicubic interpolation
    /// - B-spline interpolation of orders 3, 5, 7, 9, and 11
    ///
    /// # Arguments
    /// * `ob` - The Python object to convert
    ///
    /// # Returns
    /// * `Ok(AnyInterpolator)` if conversion succeeds
    /// * `PyErr` if the object is not a recognized interpolator type
    ///
    /// # Errors
    /// Returns a PyTypeError if the input object is not one of the supported
    /// interpolator types.
    fn extract(ob: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(py_interp) = ob.extract::<PyRef<PyNearestInterpolator>>() {
            return Ok(AnyInterpolator::Nearest(Arc::clone(&py_interp.inner)));
        }
        if let Ok(py_interp) = ob.extract::<PyRef<PyLinearInterpolator>>() {
            return Ok(AnyInterpolator::Linear(Arc::clone(&py_interp.inner)));
        }
        if let Ok(py_interp) = ob.extract::<PyRef<PyOptimizedBicubicInterpolator>>() {
            return Ok(AnyInterpolator::OptimizedBicubic(Arc::clone(&py_interp.inner)));
        }
        if let Ok(py_interp) = ob.extract::<PyRef<PyBSpline3Interpolator>>() {
            return Ok(AnyInterpolator::BSpline3(Arc::clone(&py_interp.inner)));
        }
        if let Ok(py_interp) = ob.extract::<PyRef<PyBSpline5Interpolator>>() {
            return Ok(AnyInterpolator::BSpline5(Arc::clone(&py_interp.inner)));
        }
        if let Ok(py_interp) = ob.extract::<PyRef<PyBSpline7Interpolator>>() {
            return Ok(AnyInterpolator::BSpline7(Arc::clone(&py_interp.inner)));
        }
        if let Ok(py_interp) = ob.extract::<PyRef<PyBSpline9Interpolator>>() {
            return Ok(AnyInterpolator::BSpline9(Arc::clone(&py_interp.inner)));
        }
        if let Ok(py_interp) = ob.extract::<PyRef<PyBSpline11Interpolator>>() {
            return Ok(AnyInterpolator::BSpline11(Arc::clone(&py_interp.inner)));
        }
        Err(pyo3::exceptions::PyTypeError::new_err("Expected an interpolator"))
    }
}
