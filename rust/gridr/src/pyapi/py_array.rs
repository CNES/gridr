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

use crate::core::gx_array::{GxArrayWindow};

/// A Python-exposed class representing a rectangular window for 2D array operations.
///
/// This class defines a sub-region within a 2D array, allowing operations to be applied
/// selectively to a specific portion of the data.
///
/// # Attributes
/// - `start_row` (usize): The starting row index (inclusive) of the window.
/// - `end_row` (usize): The ending row index (inclusive) of the window.
/// - `start_col` (usize): The starting column index (inclusive) of the window.
/// - `end_col` (usize): The ending column index (inclusive) of the window.
///
/// Methods:
/// - `new(start_row: usize, end_row: usize, start_col: usize, end_col: usize)`: 
///   Creates a new `PyArrayWindow2` instance with the specified row and column indices.
///
/// # Example (Python)
/// ```python
/// from my_module import PyArrayWindow2
///
/// # Define a window for a sub-region in a 10x10 array
/// window = PyArrayWindow2(start_row=2, end_row=5, start_col=3, end_col=8)
///
/// # Access attributes
/// print(window.start_row, window.end_row)  # 2, 5
/// print(window.start_col, window.end_col)  # 3, 8
///
/// # Modify attributes
/// print(window.start_row)  # 1
/// ```
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyArrayWindow2 {
    #[pyo3(get, set)]
    start_row: usize,
    #[pyo3(get, set)]
    end_row: usize,
    #[pyo3(get, set)]
    start_col: usize,
    #[pyo3(get, set)]
    end_col: usize,
}

#[pymethods]
impl PyArrayWindow2 {
    /// Creates a new `PyArrayWindow2` instance with the given start and end indices for rows and columns.
    /// 
    /// # Arguments
    /// * `start_row` (usize): The starting row index of the window (inclusive).
    /// * `end_row` (usize): The ending row index of the window (inclusive).
    /// * `start_col` (usize): The starting column index of the window (inclusive).
    /// * `end_col` (usize): The ending column index of the window (inclusive).
    /// 
    /// # Returns
    /// A new `PyArrayWindow2` instance representing the specified window.
    #[new]
    pub fn new(start_row: usize, end_row: usize, start_col: usize, end_col: usize) -> Self {
        PyArrayWindow2 {
            start_row,
            end_row,
            start_col,
            end_col,
        }
    }
}

/// Implements conversion from `PyArrayWindow2` (Python-exposed window) to `gx_array_utils::GxArrayWindow` (Rust-internal window).
///
/// This `From` implementation allows seamless conversion of a `PyArrayWindow2` instance into
/// a `GxArrayWindow` structure used internally in Rust for array processing.
impl From<PyArrayWindow2> for GxArrayWindow {
    fn from(py_win: PyArrayWindow2) -> Self {
        GxArrayWindow {
            start_row: py_win.start_row,
            end_row: py_win.end_row,
            start_col: py_win.start_col,
            end_col: py_win.end_col,
        }
    }
}

