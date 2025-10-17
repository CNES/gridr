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
//! # B-Spline Python-exposed Interpolation Module
//!
//! This module provides Python bindings for various B-spline interpolation methods
//! with different orders (3, 5, 7, 9, and 11).
//!
//! ## Key Features
//!
//! - **Multiple B-spline orders**: Supports cubic (order 3), quintic (order 5),
//!   septic (order 7), nonic (order 9), and undecic (order 11) B-splines.
//! - **Array pre-filtering**: Provides optimized methods for in-place array
//!   pre-filtering using B-spline interpolation.
//! - **Thread safety**: Uses `Arc` (Atomic Reference Counting) for thread-safe
//!   shared ownership of interpolator instances.
//! - **Python integration**: Exposes B-spline interpolators as Python classes
//!   with comprehensive method bindings.
//! - **Code reuse**: Uses a macro to implement common functionality across
//!   different B-spline orders, reducing code duplication.
//!
//! ## Interpolation Methods
//!
//! The module provides the following B-spline interpolator classes:
//!
//! - `BSpline3Interpolator`: Cubic B-spline interpolation (order 3)
//! - `BSpline5Interpolator`: Quintic B-spline interpolation (order 5)
//! - `BSpline7Interpolator`: Septic B-spline interpolation (order 7)
//! - `BSpline9Interpolator`: Nonic B-spline interpolation (order 9)
//! - `BSpline11Interpolator`: Eleventh-order B-spline interpolation (order 11)
//!
//! ## Core Functionality
//!
//! Each interpolator class provides the following methods:
//!
//! - `new(epsilon)`: Creates a new interpolator instance
//! - `shortname()` : Gets the short name of the interpolator
//! - `initialize()`: Initializes the interpolator
//! - `order()`: Gets the order of the B-spline
//! - `npoles()`: Gets the number of poles in the B-spline
//! - `bspline(x)`: Evaluates the B-spline at a given point
//! - `array1_bspline_prefiltering_f64()`: Pre-filters a 1D array using B-spline interpolation
//!
//! ## Technical Details
//!
//! - The module uses the `pyo3` crate for Python bindings
//! - The `GxBSplineInterpolator` trait provides the core interpolation functionality
//! - A macro (`impl_pybspline!`) is used to implement common functionality across
//!   different B-spline orders
//!
//! ## Usage Examples
//!
//! ```python
//! # Example of using a B-spline interpolator in Python
//! from gridr.cdylib import BSpline3Interpolator
//!
//! # Create a cubic B-spline interpolator with a 6 decimal precision
//! interp = BSpline3Interpolator(epsilon=1e-6)
//!
//! # Initialize the interpolator
//! interp.initialize()
//!
//! # Get the order of the B-spline
//! order = interp.order
//!
//! # Evaluate the B-spline at x=1.5
//! value = interp.bspline(1.5)
//! ```
//!
//! ## Performance Considerations
//!
//! - Higher order B-splines provide smoother results but are more computationally
//!   intensive
//! - The `array1_bspline_prefiltering_f64` method is optimized for performance
//!   and operates in-place on the input array
//! - The `epsilon` parameter controls the precision of the approximation and
//!   affects both accuracy and performance
//!
//! ## Error Handling
//!
//! Methods that can fail (like `initialize()` and `array1_bspline_prefiltering_f64()`)
//! return `PyErr` to propagate errors to Python code.
use std::sync::{Arc, RwLock};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArrayMethods};

use crate::core::gx_array::GxArrayViewMut;
use crate::core::interp::gx_bspline_kernel::{GxBSplineInterpolator, GxBSplineInterpolatorTrait, GxBSplineInterpolatorArgs};
use crate::core::interp::gx_array_view_interp::GxArrayViewInterpolator;

// Implements BSpline classes

/// A Python-exposed class representing a GxBSplineInterpolator<3> interpolator
///
/// This class provides a Python interface to the order 3 B-Spline interpolation functionality
/// implemented in the Rust `GxBSplineInterpolator<3>` struct. It enables seamless integration
/// between Python and Rust interpolation systems through a thread-safe interface.
///
/// The implementation uses thread-safe reference counting (`Arc`) combined with read-write
/// locking (`RwLock`) to allow safe concurrent access and sharing between threads.
#[pyclass(name = "BSpline3Interpolator")]
#[derive(Clone, Debug)]
pub struct PyBSpline3Interpolator {
    /// Inner interpolator wrapped in Arc<RwLock> for thread-safe shared ownership
    pub inner: Arc<RwLock<GxBSplineInterpolator<3>>>,
}

/// A Python-exposed class representing a GxBSplineInterpolator<5> interpolator
///
/// This class provides a Python interface to the order 5 B-Spline interpolation functionality
/// implemented in the Rust `GxBSplineInterpolator<5>` struct. It enables seamless integration
/// between Python and Rust interpolation systems through a thread-safe interface.
///
/// The implementation uses thread-safe reference counting (`Arc`) combined with read-write
/// locking (`RwLock`) to allow safe concurrent access and sharing between threads.
#[pyclass(name = "BSpline5Interpolator")]
#[derive(Clone, Debug)]
pub struct PyBSpline5Interpolator {
    /// Inner interpolator wrapped in Arc<RwLock> for thread-safe shared ownership
    pub inner: Arc<RwLock<GxBSplineInterpolator<5>>>,
}

/// A Python-exposed class representing a GxBSplineInterpolator<7> interpolator
///
/// This class provides a Python interface to the order 7 B-Spline interpolation functionality
/// implemented in the Rust `GxBSplineInterpolator<7>` struct. It enables seamless integration
/// between Python and Rust interpolation systems through a thread-safe interface.
///
/// The implementation uses thread-safe reference counting (`Arc`) combined with read-write
/// locking (`RwLock`) to allow safe concurrent access and sharing between threads.
#[pyclass(name = "BSpline7Interpolator")]
#[derive(Clone, Debug)]
pub struct PyBSpline7Interpolator {
    /// Inner interpolator wrapped in Arc<RwLock> for thread-safe shared ownership
    pub inner: Arc<RwLock<GxBSplineInterpolator<7>>>,
}

/// A Python-exposed class representing a GxBSplineInterpolator<9> interpolator
///
/// This class provides a Python interface to the order 9 B-Spline interpolation functionality
/// implemented in the Rust `GxBSplineInterpolator<9>` struct. It enables seamless integration
/// between Python and Rust interpolation systems through a thread-safe interface.
///
/// The implementation uses thread-safe reference counting (`Arc`) combined with read-write
/// locking (`RwLock`) to allow safe concurrent access and sharing between threads.
#[pyclass(name = "BSpline9Interpolator")]
#[derive(Clone, Debug)]
pub struct PyBSpline9Interpolator {
    /// Inner interpolator wrapped in Arc<RwLock> for thread-safe shared ownership
    pub inner: Arc<RwLock<GxBSplineInterpolator<9>>>,
}

/// A Python-exposed class representing a GxBSplineInterpolator<11> interpolator
///
/// This class provides a Python interface to the order 11 B-Spline interpolation functionality
/// implemented in the Rust `GxBSplineInterpolator<11>` struct. It enables seamless integration
/// between Python and Rust interpolation systems through a thread-safe interface.
///
/// The implementation uses thread-safe reference counting (`Arc`) combined with read-write
/// locking (`RwLock`) to allow safe concurrent access and sharing between threads.
#[pyclass(name = "BSpline11Interpolator")]
#[derive(Clone, Debug)]
pub struct PyBSpline11Interpolator {
    /// Inner interpolator wrapped in Arc<RwLock> for thread-safe shared ownership
    pub inner: Arc<RwLock<GxBSplineInterpolator<11>>>,
}

/// Macro to expose GxBSplineInterpolator<N> methods without code duplication
/// This macro generates Python method implementations for different dimensional
/// B-Spline interpolators.
macro_rules! impl_pybspline {
    ($name:ident, $n:expr) => {
        #[pymethods]
        impl $name {
            /// Creates a new GxBSplineInterpolator<N> instance.
            ///
            /// # Arguments
            /// * `epsilon` - The precision error used to defining the tolerance for the approximation of the infinite
            ///               sums during prefiltering.
            /// * `mask_influence_threshold` - The accepted circular influence threshold of invalid data in order to 
            ///                                update the image mask during preprocessing.
            ///
            /// # Returns
            /// A new instance of the GxBSplineInterpolator wrapper
            #[new]
            fn new(epsilon: f64, mask_influence_threshold: f64) -> Self {
                Self { inner: Arc::new(RwLock::new(GxBSplineInterpolator::<$n>::new(&GxBSplineInterpolatorArgs{ epsilon: epsilon, mask_influence_threshold: mask_influence_threshold }))) }
            }
    
            /// Gets the shortname of the interpolator
            ///
            /// This methods returns the short name of the interpolator
            ///
            /// # Returns
            /// The shortname of the interpolator
            /// ```
            pub fn shortname(&self) -> String {
                let guard = self.inner.read().unwrap(); // Read lock
                guard.shortname()
            }
            
            /// Computes and returns the total margins required on each side for the entire interpolation process.
            ///
            /// The margins are provided as a 4-element `usize` array representing the top, bottom, left, and right sides respectively.
            ///
            /// # Returns
            /// A 4-element `usize` array where each element corresponds to:
            /// - Index 0: Top margin
            /// - Index 1: Bottom margin
            /// - Index 2: Left margin
            /// - Index 3: Right margin
            pub fn total_margins(&self) -> PyResult<[usize; 4]>
            {
                let guard = self.inner.read().unwrap(); // Read lock
                guard.total_margins().map_err(|e| PyValueError::new_err(e.to_string()))
            }
            
            /// Initializes the interpolator.
            ///
            /// # Returns
            /// * `Ok(())` on successful initialization
            /// * `PyErr` if initialization fails
            fn initialize(&mut self) -> Result<(), PyErr> {
                let mut guard = self.inner.write().unwrap(); // Write lock
                guard.initialize().map_err(|e| PyValueError::new_err(e.to_string()))
            }
            
            /// Gets the order of the B-spline.
            ///
            /// # Returns
            /// The order of the B-spline as a usize
            #[getter]
            fn order(&self) -> usize {
                let guard = self.inner.read().unwrap(); // Read lock
                guard.order
            }
            
            /// Gets the number of poles in the B-spline.
            ///
            /// # Returns
            /// The number of poles as a usize
            #[getter]
            fn npoles(&self) -> usize {
                let guard = self.inner.read().unwrap(); // Read lock
                guard.npoles
            }
            
            /// Evaluates the B-spline at a given point.
            ///
            /// # Arguments
            /// * `x` - The point at which to evaluate the spline
            ///
            /// # Returns
            /// The value of the B-spline at point x
            fn bspline(&self, x: f64) -> f64 {
                let guard = self.inner.read().unwrap(); // Read lock
                guard.bspline(x)
            }
            
            /// Inplace pre-filters a flattened array using B-spline interpolation.
            /// It also updates the associated mask if provided.
            ///
            /// # Arguments
            /// * `array_in` - Input array to be filtered
            /// * `array_in_shape` - Shape of the input array as (x, y, z)
            /// * `array_in_mask` - Optional mask array (None by default)
            ///
            /// # Returns
            /// * `Ok(())` on successful filtering
            /// * `PyErr` if filtering fails
            ///
            /// # Note
            /// This method handles floating-point (f64) arrays specifically.
            #[pyo3(signature = (array_in, array_in_shape, array_in_mask=None))]
            #[allow(clippy::too_many_arguments)]
            fn array1_bspline_prefiltering_ext_f64(
                &self,
                array_in: &Bound<'_, PyArray1<f64>>,
                array_in_shape: (usize, usize, usize),
                array_in_mask: Option<&Bound<'_, PyArray1<u8>>>,
                ) -> Result<(), PyErr>
            {
                // Create a safe mutable array_view in order to be able to read and write
                // from/to the input array
                let mut array_in_view_mut = array_in.readwrite();
                let array_in_slice = array_in_view_mut.as_slice_mut().expect("Failed to get slice");
                let mut array_in_arrayview = GxArrayViewMut::new(array_in_slice, array_in_shape.0, array_in_shape.1, array_in_shape.2);
                
                // Prepare optional input validity mask to pass to the wrapped core function
                let mut mask_in_view = array_in_mask.map(|b| b.readwrite());
                let mut mask_in_array_view: Option<GxArrayViewMut<u8>> = mask_in_view.as_mut().map(|view| {
                    GxArrayViewMut::new(view.as_slice_mut().expect("Failed to get slice"), 1, array_in_shape.1, array_in_shape.2)
                });

                let guard = self.inner.read().unwrap(); // Read lock
                guard.array1_bspline_prefiltering_ext(
                    &mut array_in_arrayview,
                    mask_in_array_view.as_mut(),
                ).map_err(|e| PyValueError::new_err(e.to_string()))
            }
        }
    };
}

impl_pybspline!(PyBSpline3Interpolator, 3);
impl_pybspline!(PyBSpline5Interpolator, 5);
impl_pybspline!(PyBSpline7Interpolator, 7);
impl_pybspline!(PyBSpline9Interpolator, 9);
impl_pybspline!(PyBSpline11Interpolator, 11);
