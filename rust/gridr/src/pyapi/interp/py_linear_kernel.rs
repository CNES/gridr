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
//! # Linear Python-exposed Interpolation Module
//!
//! This module provides Python bindings for linear interpolation functionality.
//!
//! ## Key Features
//!
//! - **Python-exposed interpolator**: Provides a Python class for linear
//!   interpolation with thread-safe implementation.
//! - **Thread safety**: Uses `Arc` (Atomic Reference Counting) for thread-safe
//!   shared ownership of interpolator instances.
//! - **Error handling**: Proper error propagation to Python code.
//!
//! ## Core Functionality
//!
//! The module provides a single Python-exposed class with the following methods:
//!
//! - `new()`: Creates a new interpolator instance
//! - `shortname()` : Gets the short name of the interpolator
//! - `initialize()`: Prepares the interpolator for use
//!
//! ## Technical Implementation
//!
//! - **Python bindings**: Uses the `pyo3` crate for seamless Python integration
//! - **Core functionality**: Based on the `GxLinearInterpolator` struct
//!
//! ## Usage Examples
//!
//! ```python
//! # Example of using an interpolator through Python bindings
//! from gridr.cdylib import LinearInterpolator
//!
//! # Create a linear interpolator
//! interp = LinearInterpolator()
//!
//! # Initialize the interpolator
//! interp.initialize()
//! ```
//!
//! ## Error Handling
//!
//! Implemented methods are unlikely to fail, but any errors would be propagated
//! as Python exceptions.
//!
//! ## Thread Safety
//!
//! The implementation uses `Arc` (Atomic Reference Counting) to ensure thread-safe
//! shared ownership of interpolator instances. This allows the interpolator to be
//! safely used across multiple threads in Python applications.
use std::sync::{Arc, RwLock};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::core::interp::gx_linear_kernel::GxLinearInterpolator;
use crate::core::interp::gx_array_view_interp::{GxArrayViewInterpolator, GxArrayViewInterpolatorNoArgs};

/// A Python-exposed class representing a GxLinearInterpolator interpolator
///
/// This class provides a Python interface to the linear interpolation functionality
/// implemented in the Rust `GxLinearInterpolator` struct. It enables seamless integration
/// between Python and Rust interpolation systems through a thread-safe interface.
///
/// The implementation uses thread-safe reference counting (`Arc`) combined with read-write
/// locking (`RwLock`) to allow safe concurrent access and sharing between threads.
#[pyclass(name = "LinearInterpolator")]
#[derive(Clone, Debug)]
pub struct PyLinearInterpolator {
    /// Inner interpolator wrapped in Arc<RwLock> for thread-safe shared ownership
    pub inner: Arc<RwLock<GxLinearInterpolator>>, // Using Arc for sharing between threads
}

#[pymethods]
impl PyLinearInterpolator {
    /// Creates a new linear interpolator instance.
    ///
    /// This constructor initializes a new linear interpolator with default
    /// parameters. No arguments are required as linear interpolation doesn't
    /// need any configuration parameters.
    ///
    /// # Returns
    /// A new instance of the `PyLinearInterpolator` class
    ///
    /// # Examples
    /// ```python
    /// # Create a new linear interpolator
    /// interp = LinearInterpolator()
    /// ```
    #[new]
    pub fn new() -> Self {
        Self { inner: Arc::new(RwLock::new(GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs{}))) }
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
       
    /// Initializes the linear interpolator.
    ///
    /// This method prepares the interpolator for use. For linear interpolation,
    /// this does nothing.
    ///
    /// # Returns
    /// * `Ok(())` on successful initialization
    /// * `PyErr` if initialization fails
    fn initialize(&mut self) -> Result<(), PyErr> {
        let mut guard = self.inner.write().unwrap(); // Write lock
        guard.initialize().map_err(|e| PyValueError::new_err(e.to_string()))
    }
}