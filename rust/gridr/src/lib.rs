// Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
//
// This file is part of GRIDR
// (see https://github.com/CNES/gridr).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![warn(missing_docs)]
//! # gridr
//!
//! Rust core of [GridR](https://github.com/CNES/gridr), a raster resampling
//! and filtering library developed at CNES. No PyO3 or NumPy dependency.
//!
//! If you work in Python, use the `gridr` package from PyPI instead. It wraps
//! this crate and adds I/O, tiling and parallel processing.
//!
//! The API is not stable. Breaking changes can happen in any 0.x release.
//! See the [changelog](https://github.com/CNES/gridr/blob/main/CHANGELOG.md).
//!
//! ## Contents
//!
//! - [`gx_grid_resampling`]: resampling through a grid of source coordinates,
//! with oversampling, masks and nodata.
//! - [`interp`]: nearest, linear, bicubic and B-spline (order 3 to 11)
//! interpolators.
//! - [`gx_grid_geometry`]: source and destination footprints of a grid.
//! - [`gx_array`]: array views used by all functions.
//!
//! Arrays are flat row-major slices. A view adds the shape
//! `(nvar, nrow, ncol)` and copies nothing.
//!
//! Functions run on a single thread. Parallelism is up to the caller.
//!
//! ## Example
//!
//! Half-pixel shift of a 4x4 image with linear interpolation. Integer
//! coordinates are pixel centers.
//!
//! ```
//! use gridr::gx_array::{GxArrayView, GxArrayViewMut};
//! use gridr::gx_grid_resampling::{array1_grid_resampling, NoCheckGridMeshValidator};
//! use gridr::interp::gx_array_view_interp::{
//! GxArrayViewInterpolator, GxArrayViewInterpolatorNoArgs, SupportPolicy,
//! };
//! use gridr::interp::gx_linear_kernel::GxLinearInterpolator;
//!
//! // Input image: value = 4 * row + col.
//! let (nrow_in, ncol_in) = (4, 4);
//! let data_in: Vec<f64> = (0..nrow_in * ncol_in).map(|i| i as f64).collect();
//! let ima_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
//!
//! // Resampling grid: output pixel (r, c) reads input at (r + 0.5, c + 0.5).
//! let (nrow_out, ncol_out) = (3, 3);
//! let mut grid_row = Vec::with_capacity(nrow_out * ncol_out);
//! let mut grid_col = Vec::with_capacity(nrow_out * ncol_out);
//! for r in 0..nrow_out {
//! for c in 0..ncol_out {
//! grid_row.push(r as f64 + 0.5);
//! grid_col.push(c as f64 + 0.5);
//! }
//! }
//! let grid_row = GxArrayView::new(&grid_row, 1, nrow_out, ncol_out);
//! let grid_col = GxArrayView::new(&grid_col, 1, nrow_out, ncol_out);
//!
//! let mut data_out = vec![0.0_f64; nrow_out * ncol_out];
//! let mut ima_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
//!
//! let interp = GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs);
//! array1_grid_resampling(
//! &interp,
//! &NoCheckGridMeshValidator {},
//! &ima_in,
//! &grid_row,
//! &grid_col,
//! 1, // grid row oversampling
//! 1, // grid column oversampling
//! &mut ima_out,
//! f64::NAN, // output nodata
//! None, // input mask
//! None, // input mask safe window
//! None, // output mask
//! None, // grid window
//! None, // output window
//! None, // input origin row
//! None, // input origin column
//! true, // check boundaries
//! SupportPolicy::Effective,
//! )
//! .unwrap();
//!
//! assert_eq!(data_out[0], 2.5); // 4 * 0.5 + 0.5
//! assert_eq!(data_out[8], 12.5); // 4 * 2.5 + 2.5
//! ```
//!
//! ## Documentation
//!
//! Algorithms, conventions and tutorials are in the
//! [GridR documentation](https://gridr.readthedocs.io). Issues go to the
//! [repository](https://github.com/CNES/gridr).
pub mod gx_errors;
pub mod gx_macros;
pub mod gx_const;
pub mod gx_utils;
pub mod gx_array;
pub mod gx_array_utils;
pub mod gx_grid;
pub mod gx_grid_geometry;
pub mod gx_grid_resampling;
pub mod interp;

/// This crate's own version (`CARGO_PKG_VERSION`)
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
