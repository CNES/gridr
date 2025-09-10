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
//! # Grid Geometry Utilities
//!
//! This module provides tools to analyze and describe the geometric structure of non-uniform grids.
//! It defines key geometric constructs such as transition matrices and bounding boxes, and includes
//! algorithms to derive destination-to-source mappings for arbitrary grid transformations.
//!
//! ## Core Concepts
//!
//! - [`GridTransitionMatrix`]: Represents a 2D linear transformation from a canonical grid (e.g. 
//!   uniform indexing) to a custom geometry grid via a change-of-basis matrix. It is expressed
//!   using generator vectors for rows and columns.
//!
//! - [`GeometryBounds<T>`]: Encapsulates rectangular bounds in 2D space, used to define extents in 
//!   either grid or spatial coordinates.
//!
//! - [`GridGeometriesMetrics<T>`]: Collects detailed metrics for a transformed grid, including:
//!     - Source and destination bounding boxes,
//!     - Edge definitions for rows and columns,
//!     - The associated linear transformation.
//!
//! - [`array1_compute_resampling_grid_geometries`]: Main entry point for computing all geometric 
//!   descriptors of a resampling grid based on input arrays of source coordinates.
//!
//!
//! ## Type Constraints
//!
//! The generic types used allow high flexibility:
//!
//! - `W`: The raw input grid coordinate type (e.g., `f32`, `f64`, `i16`, etc.).
//! - `T`: A floating-point type used for calculations (e.g., `f64`), derived from `W`.
//!
//! ## Design Notes
//!
//! - The algorithm emphasizes memory efficiency, with per-row/column streaming-style valid edge
//!   detection.
//! - It handles completely invalid input grids gracefully by returning `Ok(None)`.
//! - Partial ordering support (`min_partial`, `max_partial`) allows robust bound estimation
//!   even when dealing with NaNs or undefined behavior in edge cases.
use crate::core::gx_array::{GxArrayView, GxArrayWindow};
use crate::core::gx_grid::{GridNodeValidator};
use crate::core::gx_errors::GxError;
use crate::core::gx_utils::{min_partial, max_partial, approx_eq_tuple2, GxToF64};

use approx::{abs_diff_eq};


/// A transition matrix that defines a linear transformation between a canonical grid
/// (e.g. regular array indexing) and a transformed geometry grid.
///
/// This structure models a regular grid transformation as a linear isomorphism:
/// such a transformation defines a new basis formed by the image of the canonical basis
/// under the transformation. The associated change-of-basis matrix (also known as the
/// transition matrix) is represented as a 2×2 matrix `W` such that:
///
/// ```text
/// | x |   | W1_x   W2_x |   | x_new |
/// |   | = |              | × |       |
/// | y |   | W1_y   W2_y |   | y_new |
/// ```
///
/// In this representation:
/// - `W1` corresponds to the column generator vector (i.e., the image of the unit vector along x).
/// - `W2` corresponds to the row generator vector (i.e., the image of the unit vector along y).
///
/// The vectors are stored as optional pairs of coordinates `(x, y)` in the destination space.
/// If either vector is missing (`None`), the transformation is considered undefined.
#[derive(Debug, Clone, Copy)]
pub struct GridTransitionMatrix {
    /// Column generator vector (x: col, y: row).
    pub w1: Option<(f64, f64)>,
    /// Row generator vector (x: col, y: row).
    pub w2: Option<(f64, f64)>,
}

impl GridTransitionMatrix {
    /// Creates a new [`GridTransitionMatrix`] from optional column and row generator vectors.
    ///
    /// # Parameters
    ///
    /// - `w1`: Column generator vector as `(x, y)` or `None` if undefined.
    /// - `w2`: Row generator vector as `(x, y)` or `None` if undefined.
    ///
    /// # Returns
    ///
    /// A new [`GridTransitionMatrix`] instance.
    pub fn new(w1: Option<(f64, f64)>, w2: Option<(f64, f64)>) -> Self {
        GridTransitionMatrix { w1, w2 }
    }

    /// Returns the determinant of the transformation matrix `W`, if both vectors are defined.
    ///
    /// # Returns
    ///
    /// - `Some(det)` if both `w1` and `w2` are `Some`, where `det = w1_x * w2_y - w2_x * w1_y`.
    /// - `None` if either `w1` or `w2` is `None`.
    pub fn det_w(&self) -> Option<f64> {
        match (self.w1, self.w2) {
            (Some((w1_x, w1_y)), Some((w2_x, w2_y))) => {
                Some(w1_x * w2_y - w2_x * w1_y)
            }
            _ => None,
        }
    }

    /// Returns the flattened 2×2 transition matrix as `[w1_x, w2_x, w1_y, w2_y]`, if defined.
    ///
    /// # Returns
    ///
    /// - `Some([w1_x, w2_x, w1_y, w2_y])` if both `w1` and `w2` are `Some`.
    /// - `None` if either vector is missing.
    pub fn w(&self) -> Option<[f64; 4]> {
        match (self.w1, self.w2) {
            (Some((w1_x, w1_y)), Some((w2_x, w2_y))) => {
                Some([w1_x, w2_x, w1_y, w2_y])
            }
            _ => None,
        }
    }
}


/// Represents a rectangular bounding box in 2D space.
///
/// The `GeometryBounds<T>` struct defines the minimum and maximum X and Y
/// coordinates that bound a geometry in a 2D coordinate system. It is generic
/// over type `T`, allowing it to work with integers, floating-point numbers,
/// or any numeric type.
///
/// # Fields
///
/// - `xmin`: Minimum coordinate along the X axis.
/// - `xmax`: Maximum coordinate along the X axis.
/// - `ymin`: Minimum coordinate along the Y axis.
/// - `ymax`: Maximum coordinate along the Y axis.
///
/// # Example
///
/// ```rust
/// let bounds = GeometryBounds {
///     xmin: 10.0,
///     xmax: 20.0,
///     ymin: 5.0,
///     ymax: 15.0,
/// };
/// assert!(bounds.xmin < bounds.xmax);
/// ```
///
/// # Constraints
///
/// This struct does not enforce that `xmin <= xmax` or `ymin <= ymax` at
/// construction time. It is the caller’s responsibility to ensure the bounds
/// are valid.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryBounds<T> {
    /// Minimum coordinate along the X axis.
    pub xmin: T,
    /// Maximum coordinate along the X axis.
    pub xmax: T,
    /// Minimum coordinate along the Y axis.
    pub ymin: T,
    /// Maximum coordinate along the Y axis.
    pub ymax: T,
}

impl<T> GeometryBounds<T>
where
    T: Copy + GxToF64,
{
    /// Compares two `GeometryBounds<T>` instances for approximate equality.
    ///
    /// All four components (`xmin`, `xmax`, `ymin`, `ymax`) are compared using
    /// absolute difference within the specified `epsilon`.
    ///
    /// # Type Parameters
    /// - `T`: A type that can be converted into `f64` (typically `f32` or `f64`).
    ///
    /// # Parameters
    /// - `a`: First bounds.
    /// - `b`: Second bounds.
    /// - `epsilon`: Tolerance for approximate equality.
    ///
    /// # Returns
    /// - `true` if all components differ by less than or equal to `epsilon`.
    /// - `false` otherwise.
    pub fn approx_eq(a: &GeometryBounds<T>, b: &GeometryBounds<T>, epsilon: f64) -> bool {
        abs_diff_eq!(a.xmin.to_f64(), b.xmin.to_f64(), epsilon = epsilon) &&
        abs_diff_eq!(a.xmax.to_f64(), b.xmax.to_f64(), epsilon = epsilon) &&
        abs_diff_eq!(a.ymin.to_f64(), b.ymin.to_f64(), epsilon = epsilon) &&
        abs_diff_eq!(a.ymax.to_f64(), b.ymax.to_f64(), epsilon = epsilon)
    }
}

/// A list of optional 1D edges or intervals.
///
/// Each element is either:
/// - `Some((start, end))`: a valid interval from `start` to `end`
/// - `None`: an undefined or missing edge
///
/// Useful for representing sparse or partial geometries (e.g., scanline spans).
///
/// # Type Parameters
/// - `T`: Coordinate type (e.g., `f64`, `i32`)
///
/// # Example
/// ```rust
/// let edges: EdgeVec<f64> = vec![Some((0.0, 1.0)), None, Some((2.5, 3.0))];
/// ```
type EdgeVec<T> = Vec<Option<(T, T)>>;

/// Compares two vectors of optional `(start, end)` intervals for approximate equality.
///
/// Each pair of intervals is compared:
/// - If both are `None`, they are equal.
/// - If both are `Some((a0, a1))` and `Some((b0, b1))`, their components must be
///   within the given `epsilon` difference (or exactly equal if `epsilon` is not provided).
/// - If one is `None` and the other is `Some`, they are not equal.
///
/// # Parameters
/// - `a`: First edge vector.
/// - `b`: Second edge vector.
/// - `epsilon`: Optional tolerance value for floating-point comparisons (default: 0.0).
///
/// # Returns
/// - `true` if vectors have the same shape and all corresponding values are approximately equal.
/// - `false` otherwise.
fn approx_eq_edgevec<T>(
    a: &EdgeVec<T>,
    b: &EdgeVec<T>,
    epsilon: f64
) -> bool
where
    T: Copy + GxToF64,
{
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b.iter()).all(|(a_opt, b_opt)| match (a_opt, b_opt) {
        (Some((a0, a1)), Some((b0, b1))) => {
            abs_diff_eq!((*a0).to_f64(), (*b0).to_f64(), epsilon = epsilon)
                && abs_diff_eq!((*a1).to_f64(), (*b1).to_f64(), epsilon = epsilon)
        }
        (None, None) => true,
        _ => false,
    })
}


/// Stores geometric and resampling metrics between a source and destination grid.
///
/// Captures the bounding boxes and edge intervals for both grids, along with
/// the interpolation transitions needed to map between them.
///
/// # Fields
/// - `dst_bounds`: Bounding box of the destination grid (in pixel indices).
/// - `src_bounds`: Bounding box of the source grid (in coordinate units).
/// - `dst_row_edges`: For each **row** in the destination grid, stores the interval of valid **column indices**.
/// - `dst_col_edges`: For each **column** in the destination grid, stores the interval of valid **row indices**.
/// - `src_row_edges`: For each **row** in the destination grid, stores the **source coordinate segment** `((y0, x0), (y1, x1))` that contributes to it.
/// - `src_col_edges`: For each **column** in the destination grid, stores the **source coordinate segment** `((y0, x0), (y1, x1))` that contributes to it.
/// - `transition_matrix`: The grid transition matrix
///
/// # Type Parameters
/// - `T`: Coordinate type (e.g., `f64`).
#[derive(Debug, Clone)]
pub struct GridGeometriesMetrics<T> {
    /// Bounding box of the destination grid (in pixel indices).
    pub dst_bounds: GeometryBounds<usize>, 
    /// Bounding box of the source grid (in coordinate units).
    pub src_bounds: GeometryBounds<T>,
    /// Interval of valid **column indices** for each **row** in the destination grid.
    pub dst_row_edges: EdgeVec<usize>,
    /// Interval of valid **row indices** for each **column** in the destination grid.
    pub dst_col_edges: EdgeVec<usize>,
    /// **source coordinate segment** `((y0, x0), (y1, x1))` that contributes to each **row** in the destination grid.
    pub src_row_edges: (EdgeVec<T>, EdgeVec<T>),
    /// **source coordinate segment** `((y0, x0), (y1, x1))` that contributes to each **column** in the destination grid.
    pub src_col_edges: (EdgeVec<T>, EdgeVec<T>),
    /// The grid transition matrix
    pub transition_matrix: GridTransitionMatrix,
}

impl GridGeometriesMetrics<f64> {
    /// Compares two [`GridGeometriesMetrics<f64>`] instances approximately.
    ///
    /// This method checks if two metric objects are approximately equal within a
    /// given absolute tolerance `epsilon`. It is useful for testing or validating
    /// numerical computations where floating-point rounding errors may occur.
    ///
    /// The comparison includes:
    /// - Transition matrix vectors `w1` and `w2`, compared using tuple-wise approximate equality.
    /// - Source bounds (`src_bounds`) compared approximately.
    /// - Destination bounds (`dst_bounds`) compared exactly (integers).
    /// - All edge vectors (`dst_row_edges`, `dst_col_edges`, `src_row_edges`, `src_col_edges`)
    ///   compared either exactly or using approximate equality for floating-point coordinates.
    ///
    /// # Parameters
    /// - `other`: The reference metrics to compare against.
    /// - `epsilon`: The allowed absolute difference when comparing floating-point values.
    ///
    /// # Returns
    /// - `true` if all fields are equal within tolerance (or exactly for non-float fields).
    /// - `false` otherwise.
    ///
    /// # Panics
    /// This function does not panic.
    ///
    /// # Examples
    /// ```rust
    /// let equal = metrics_a.approx_eq(&metrics_b, 1e-6);
    /// assert!(equal);
    /// ```
    pub fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
        // Compare matrices
        match (self.transition_matrix.w1, other.transition_matrix.w1) {
            (Some(a), Some(b)) => if !approx_eq_tuple2::<f64>(a, b, epsilon) {
                println!("transition_matrix.w1 left : {:#?}", self.transition_matrix.w1);
                println!("transition_matrix.w1 right : {:#?}", other.transition_matrix.w1);
                return false;
                },
            (None, None) => {},
            _ => {
                println!("transition_matrix.w1 left : {:#?}", self.transition_matrix.w1);
                println!("transition_matrix.w1 right : {:#?}", other.transition_matrix.w1);
                return false;
                },
        }
        match (self.transition_matrix.w2, other.transition_matrix.w2) {
            (Some(a), Some(b)) => if !approx_eq_tuple2::<f64>(a, b, epsilon) {
                println!("transition_matrix.w2 left : {:#?}", self.transition_matrix.w2);
                println!("transition_matrix.w2 right : {:#?}", other.transition_matrix.w2);
                return false;
                },
            (None, None) => {},
            _ => {
                println!("transition_matrix.w2 left : {:#?}", self.transition_matrix.w2);
                println!(" transition_matrix.w2right : {:#?}", other.transition_matrix.w2);
                return false;
                },
        }

        // src_bounds and dst_bounds
        if self.dst_bounds != other.dst_bounds {
            println!("dst_bounds left : {:#?}", self.dst_bounds);
            println!("dst_bounds right : {:#?}", other.dst_bounds);
            return false;
        }
        if !GeometryBounds::<f64>::approx_eq(&self.src_bounds, &other.src_bounds, epsilon) {
            println!("src_bounds left : {:#?}", self.src_bounds);
            println!("src_bounds right : {:#?}", other.src_bounds);
            return false;
        }

        // Compare EdgeVec variables
        if !approx_eq_edgevec(&self.dst_row_edges, &other.dst_row_edges, epsilon) {
            println!("dst_row_edges left : {:#?}", self.dst_row_edges);
            println!("dst_row_edges right : {:#?}", other.dst_row_edges);
            return false;
        }
        if !approx_eq_edgevec(&self.dst_col_edges, &other.dst_col_edges, epsilon) {
            println!("dst_col_edges left : {:#?}", self.dst_col_edges);
            println!("dst_col_edges right : {:#?}", other.dst_col_edges);
            return false;
        }
        if !approx_eq_edgevec(&self.src_row_edges.0, &other.src_row_edges.0, epsilon) {
            println!("src_row_edges.0 left : {:#?}", self.src_row_edges.0);
            println!("src_row_edges.0 right : {:#?}", other.src_row_edges.0);
            return false;
        }
        if !approx_eq_edgevec(&self.src_row_edges.1, &other.src_row_edges.1, epsilon) {
            println!("src_row_edges.1 left : {:#?}", self.src_row_edges.1);
            println!("src_row_edges.1 right : {:#?}", other.src_row_edges.1);
            return false;
        }
        if !approx_eq_edgevec(&self.src_col_edges.0, &other.src_col_edges.0, epsilon) {
            println!("src_col_edges.0 left : {:#?}", self.src_col_edges.0);
            println!("src_col_edges.0 right : {:#?}", other.src_col_edges.0);
            return false;
        }
        if !approx_eq_edgevec(&self.src_col_edges.1, &other.src_col_edges.1, epsilon) {
            println!("src_col_edges.1 left : {:#?}", self.src_col_edges.1);
            println!("src_col_edges.1 right : {:#?}", other.src_col_edges.1);
            return false;
        }

        true
    }
}

/// Computes the resampling grid geometries and metrics for a given 2D grid representation.
///
/// This function analyzes the validity of grid points (nodes) along rows and columns to determine
/// bounding boxes and transition matrix for resampling operations on grids. It returns detailed
/// information about the source and destination grid bounds, as well as the transition matrix
/// describing the linear transformation between source and destination grids.
///
/// If provided, the computation is limited to the region defined by `win`.
/// Otherwise, the entire arrays are processed.
///
/// # Type Parameters
/// - `W`: Type of the underlying grid data elements (must implement `Copy`, `PartialEq`, and `Into<T>`).
/// - `T`: Numeric type used for computations and geometry bounds (must implement `Copy`, `PartialEq`, `PartialOrd` and
///   `GxToF64`).
/// - `C`: Type implementing the `GridNodeValidator` trait, used to validate grid nodes.
///
/// # Arguments
/// - `grid_row_array`: A view of the grid data structured by rows (type `GxArrayView<W>`).
/// - `grid_col_array`: A view of the grid data structured by columns (type `GxArrayView<W>`).
/// - `grid_row_oversampling`: Oversampling factor applied along grid rows (usize).
/// - `grid_col_oversampling`: Oversampling factor applied along grid columns (usize).
/// - `grid_validity_checker`: Reference to a validator implementing `GridNodeValidator<W>`
///    that checks whether a given grid node is valid.
/// - `win`: A `GxArrayWindow` defining the sub-region of grid's arrays where computation will be performed.
///
/// # Returns
/// Returns a `Result` wrapping an optional `GridGeometriesMetrics<T>`:
/// - `Ok(Some(...))` with detailed grid geometry metrics if valid nodes are found.
/// - `Ok(None)` if the grid contains no valid nodes (fully invalid grid).
/// - `Err(GxError)` on unexpected internal errors (e.g., missing expected data).
///
/// # Description
/// The function performs the following main steps:
/// 1. **Row-wise iteration:**  
///    For each row, two sweeps are performed:
///    - Left-to-right to find the first valid grid node (minimum column index).
///    - Right-to-left to find the last valid grid node (maximum column index).
///    This determines the valid horizontal bounds per row in both source and destination coordinates.
///
/// 2. **Column-wise iteration:**  
///    For each column, two sweeps are performed similarly:
///    - Top-to-bottom to find the first valid grid node (minimum row index).
///    - Bottom-to-top to find the last valid grid node (maximum row index).
///    This determines the valid vertical bounds per column.
///
/// 3. **Bounding box calculation:**  
///    Global bounding boxes are computed both in source and destination coordinate systems.
///
/// 4. **Transition matrix computation:**  
///    Two transition vectors (`w1`, `w2`) are computed representing the average scaling factors 
///    between source and destination along rows and columns, respectively. They are calculated 
///    as the ratio of coordinate distances over the corresponding index spans multiplied by
///    oversampling factors. For each axis, the longest index span is used in the computation.
///
/// # Edge Cases and Limits
/// - **Fully invalid grid:**  
///   If no valid nodes are found anywhere in the grid (e.g., all masked or invalid), the function
///   returns `Ok(None)`.
///
/// - **Irregular grid with all zero coordinates:**  
///   When the grid data contains identical coordinates (e.g., all zeros), the function returns
///   valid geometry metrics, but with zero-sized bounding boxes and transition matrix weights set to zero.
///   This corresponds to grids violating the regularity assumption, but still processed without error.
///
/// - **Partially masked grid:**  
///   When only some nodes are valid (e.g., a single valid cell), the function returns geometry metrics
///   reflecting only those valid nodes. The bounding boxes and destination edges reflect the limited
///   valid area, and transition matrix components (`w1`, `w2`) may be `None` if insufficient points
///   exist to compute meaningful vectors.
///
/// # Notes
/// - The grid is assumed to be regular, and the transition matrix computations rely on this assumption.
/// - The function uses a two-pass sweeping approach for row and column scanning to improve efficiency 
///   when multiple valid nodes exist.
/// - If no valid nodes are detected, the function returns `Ok(None)` early.
///
/// # Errors
/// The function may return an error if expected bounding box components are missing due to logic inconsistencies.
///
/// # Example
/// ```
/// let result = array1_compute_resampling_grid_geometries(
///     &grid_row_view,
///     &grid_col_view,
///     row_oversampling,
///     col_oversampling,
///     &validator
/// )?;
/// if let Some(metrics) = result {
///     // Use metrics.dst_bounds, metrics.src_bounds, etc.
/// } else {
///     // Handle fully invalid grid case
/// }
/// ```
pub fn array1_compute_resampling_grid_geometries<W, T, C>(
        grid_row_array: &GxArrayView<'_, W>,
        grid_col_array: &GxArrayView<'_, W>,
        grid_row_oversampling: usize,
        grid_col_oversampling: usize,
        grid_validity_checker: &C,
        win: Option<GxArrayWindow>,
        ) -> Result<Option<GridGeometriesMetrics<T>>, GxError>
where
    W: Copy + PartialEq + Into<T>,
    T: Copy + PartialOrd + PartialEq + GxToF64 + Default,
    C: GridNodeValidator<W>,
{
    // Retrieve the optional window or full if win is None.
    // This call checks the window against the array dimension and may return
    // an error if the window is out of bounds of the array.
    let array_win = GxArrayWindow::resolve_or_full(win, grid_row_array, true)?;
    
    // Variables definition
    // - `scan_row_dst` holds the range of valid destination indices (inclusive) for each row.
    //     These indices are defined in the coordinate system of the destination grid,
    //     i.e., they refer to the native indexing of the grid arrays.
    // - `scan_col_dst` holds the range of valid destination indices (inclusive) for each column.
    //     These indices are defined in the coordinate system of the destination grid,
    //     i.e., they refer to the native indexing of the grid arrays.
    let mut scan_row_dst: Vec<Option<(usize, usize)>> = vec![None; array_win.height()];
    let mut scan_row_src_left: Vec<Option<(T, T)>> = vec![None; array_win.height()];
    let mut scan_row_src_right: Vec<Option<(T, T)>> = vec![None; array_win.height()];
    let mut dst_max_col_length: usize = 0;
    let mut dst_max_col_src_left: [T; 2] = [T::default(), T::default()];
    let mut dst_max_col_src_right: [T; 2] = [T::default(), T::default()];
    
    let mut scan_col_dst: Vec<Option<(usize, usize)>> = vec![None; array_win.width()];
    let mut scan_col_src_top: Vec<Option<(T, T)>> = vec![None; array_win.width()];
    let mut scan_col_src_bottom: Vec<Option<(T, T)>> = vec![None; array_win.width()];
    let mut dst_max_row_length: usize = 0;
    let mut dst_max_row_src_top: [T; 2] = [T::default(), T::default()];
    let mut dst_max_row_src_bottom: [T; 2] = [T::default(), T::default()];
    
    // Init `cidx_line` ; it stores the current absolute row index used in the 
    // first loop on row.
    let mut cidx_line = array_win.start_row * grid_row_array.ncol;
    
    // Init the dst bounding box to full size
    let mut glb_col_min_dst: Option<usize> = None;
    let mut glb_col_max_dst: Option<usize> = None;
    let mut glb_row_min_dst: Option<usize> = None;
    let mut glb_row_max_dst: Option<usize> = None;
    
    // Init the src bounding box - please note we lose here the orientation
    let mut glb_col_min_src: Option<T> = None;
    let mut glb_col_max_src: Option<T> = None;
    let mut glb_row_min_src: Option<T> = None;
    let mut glb_row_max_src: Option<T> = None;
    
    // This variable is used in order to detect a full invalid Grid
    // In such a case, only the loop on row is performed
    let mut full_invalid = true;
    
    // First loop on row in order to determine left and right limits
    // On each row the processing consists of two passes:
    // 1. A left-to-right sweep to find the minimum valid value.
    // 2. A right-to-left sweep to find the maximum valid value.
    //
    // Note: In the worst-case scenario where the entire row contains only invalid data,
    // this results in two full passes over the data, effectively doubling the iteration cost
    // compared to a single-pass approach.
    //
    // However, when there is only one valid value in the row, the two passes meet at that point,
    // so the total number of iterations is effectively the same as a single pass (plus one pixel).
    //
    // As soon as multiple valid values exist, this two-pass approach improves efficiency by limiting
    // the total number of pixels processed, since each pass can stop once the relevant boundary is found.
    for idx_row in 0..array_win.height() {
        let mut min_col_idx = None;
        let mut max_col_idx = None;
        let mut left_grid_row_value = None;
        let mut right_grid_row_value = None;
        let mut left_grid_col_value = None;
        let mut right_grid_col_value = None;
        //let abs_idx_row = idx_row + array_win.start_row;
        
        
        // Sweep from left to right in order to find min
        for idx_col in array_win.start_col..=array_win.end_col {
            let cidx = cidx_line + idx_col;
            //let cidx = abs_idx_row * grid_row_array.ncol + idx_col;
            
            // Check data is valid
            if grid_validity_checker.validate(cidx, &grid_row_array) {
                min_col_idx = Some(idx_col);
                left_grid_row_value = Some(grid_row_array.data[cidx]);
                left_grid_col_value = Some(grid_col_array.data[cidx]);
                break;
            }
        }
        
        // Sweep from right to left in order to find max
        for idx_col in (array_win.start_col..=array_win.end_col).rev() {
            let cidx = cidx_line + idx_col;
            //let cidx = abs_idx_row * grid_row_array.ncol + idx_col;
            
            // Check data is valid
            if grid_validity_checker.validate(cidx, &grid_row_array) {
                max_col_idx = Some(idx_col);
                right_grid_row_value = Some(grid_row_array.data[cidx]);
                right_grid_col_value = Some(grid_col_array.data[cidx]);
                break;
            }
        }
        
        if let (Some(a), Some(b)) = (min_col_idx, max_col_idx) {
            scan_row_dst[idx_row] = Some((a, b));

            
            // Unwrap safely or assume they are always Some
            let xl: T = left_grid_col_value.expect("left_grid_col_value should be Some").into();
            let yl: T = left_grid_row_value.expect("left_grid_row_value should be Some").into();
            let xr: T = right_grid_col_value.expect("right_grid_col_value should be Some").into();
            let yr: T = right_grid_row_value.expect("right_grid_row_value should be Some").into();
            
            scan_row_src_left[idx_row] = Some((yl, xl));
            scan_row_src_right[idx_row] = Some((yr, xr));
                        
            full_invalid = false;
            
            glb_col_min_dst = Some(glb_col_min_dst.map_or(a, |c| c.min(a)));
            glb_col_max_dst = Some(glb_col_max_dst.map_or(b, |c| c.max(b)));
            
            let candidate_xmin: T = min_partial(xl, xr);
            let candidate_xmax: T = max_partial(xl, xr);
            let candidate_ymin: T = min_partial(yl, yr);
            let candidate_ymax: T = max_partial(yl, yr);

            glb_col_min_src = Some(glb_col_min_src.map_or(candidate_xmin, |c| min_partial(c, candidate_xmin)));
            glb_col_max_src = Some(glb_col_max_src.map_or(candidate_xmax, |c| max_partial(c, candidate_xmax)));
            glb_row_min_src = Some(glb_row_min_src.map_or(candidate_ymin, |c| min_partial(c, candidate_ymin)));
            glb_row_max_src = Some(glb_row_max_src.map_or(candidate_ymax, |c| max_partial(c, candidate_ymax)));
            
            // Determine the row with the longest destination span (b - a).
            // This is done inline during iteration to avoid allocating an intermediate array
            // and to keep memory usage minimal with a single pass.
            let dst_length = b - a;
            if dst_length > dst_max_col_length {
                dst_max_col_length = dst_length;
                dst_max_col_src_left = [yl, xl];
                dst_max_col_src_right = [yr, xr];
                
            }
        }
        // Increment row index
        cidx_line += grid_row_array.ncol;
    }
    
    // Proceed with a column-wise iteration to determine the top and bottom boundaries.
    // The approach follows the same logic as described for the row iteration.
    //
    // Note: This loop is skipped if the previous row-wise pass concluded
    // that no valid nodes were present.
    if !full_invalid {
        for idx_col in 0..array_win.width() {
            let mut min_row_idx = None;
            let mut max_row_idx = None;
            let mut top_grid_row_value = None;
            let mut bottom_grid_row_value = None;
            let mut top_grid_col_value = None;
            let mut bottom_grid_col_value = None;
            let abs_idx_col = idx_col + array_win.start_col;
            
            
            // Sweep from top to bottom in order to find min
            for idx_row in array_win.start_row..=array_win.end_row {
                let cidx = idx_row * grid_row_array.ncol + abs_idx_col;
                
                // Check data is valid
                if grid_validity_checker.validate(cidx, &grid_row_array) {
                    min_row_idx = Some(idx_row);
                    top_grid_row_value = Some(grid_row_array.data[cidx]);
                    top_grid_col_value = Some(grid_col_array.data[cidx]);
                    break;
                }
            }
            
            // Sweep from bottom to top in order to find max
            for idx_row in (array_win.start_row..=array_win.end_row).rev() {
                let cidx = idx_row * grid_row_array.ncol + abs_idx_col;
                
                // Check data is valid
                if grid_validity_checker.validate(cidx, &grid_row_array) {
                    max_row_idx = Some(idx_row);
                    bottom_grid_row_value = Some(grid_row_array.data[cidx]);
                    bottom_grid_col_value = Some(grid_col_array.data[cidx]);
                    break;
                }
            }
            
            if let (Some(a), Some(b)) = (min_row_idx, max_row_idx) {
                scan_col_dst[idx_col] = Some((a, b));
                
                // Unwrap safely or assume they are always Some
                let xt: T = top_grid_col_value.expect("top_grid_col_value should be Some").into();
                let yt: T = top_grid_row_value.expect("top_grid_row_value should be Some").into();
                let xb: T = bottom_grid_col_value.expect("bottom_grid_col_value should be Some").into();
                let yb: T = bottom_grid_row_value.expect("bottom_grid_row_value should be Some").into();
                
                scan_col_src_top[idx_col] = Some((yt, xt));
                scan_col_src_bottom[idx_col] = Some((yb, xb));
                
                glb_row_min_dst = Some(glb_row_min_dst.map_or(a, |c| c.min(a)));
                glb_row_max_dst = Some(glb_row_max_dst.map_or(b, |c| c.max(b)));
                
                let candidate_xmin: T = min_partial(xb, xt);
                let candidate_xmax: T = max_partial(xb, xt);
                let candidate_ymin: T = min_partial(yb, yt);
                let candidate_ymax: T = max_partial(yb, yt);

                glb_col_min_src = Some(glb_col_min_src.map_or(candidate_xmin, |c| min_partial(c, candidate_xmin)));
                glb_col_max_src = Some(glb_col_max_src.map_or(candidate_xmax, |c| max_partial(c, candidate_xmax)));
                glb_row_min_src = Some(glb_row_min_src.map_or(candidate_ymin, |c| min_partial(c, candidate_ymin)));
                glb_row_max_src = Some(glb_row_max_src.map_or(candidate_ymax, |c| max_partial(c, candidate_ymax)));               
                
                // Determine the col with the longest destination span (b - a).
                // This is done inline during iteration to avoid allocating an intermediate array
                // and to keep memory usage minimal with a single pass.
                let dst_length = b - a;
                if dst_length > dst_max_row_length {
                    dst_max_row_length = dst_length;
                    dst_max_row_src_top = [yt, xt];
                    dst_max_row_src_bottom = [yb, xb];
                }
            }
        }
    }
    
    if full_invalid {
        return Ok(None);
    }
    
    // This function computes the transition matrix (W)with a major assumptions : the regularity of the grid.
    // Therefore :
    //  - W1 is computed from the grid by taking the ratio of :
    //       1) the distance between the farthest two valid distincts points of the grid along a line 
    //       2) with the distance of corresponding indices (ie. number of columns).
    //  - W2 is computed from the grid by taking the ratio of :
    //       1) the distance between the farthest two valid distincts points of the grid along a column 
    //       2) with the distance of corresponding indices (ie. number of lines).
    // Compute w1 if column length is valid
    let w1: Option<(f64, f64)> = (dst_max_col_length > 0).then(|| {
        let len: f64 = (dst_max_col_length as f64) * (grid_col_oversampling as f64);
        let x: f64 = ((dst_max_col_src_right[1]).to_f64() - (dst_max_col_src_left[1]).to_f64()) / len;
        let y: f64 = ((dst_max_col_src_right[0]).to_f64() - (dst_max_col_src_left[0]).to_f64()) / len;
        (x, y)
    });

    // Compute w2 if row length is valid
    let w2: Option<(f64, f64)> = (dst_max_row_length > 0).then(|| {
        let len: f64 = (dst_max_row_length as f64) * (grid_row_oversampling as f64);
        let x: f64 = ((dst_max_row_src_bottom[1]).to_f64() - (dst_max_row_src_top[1]).to_f64()) / len;
        let y: f64 = ((dst_max_row_src_bottom[0]).to_f64() - (dst_max_row_src_top[0]).to_f64()) / len;
        (x, y)
    });

    let dst_bounds = GeometryBounds {
        xmin: glb_col_min_dst.ok_or(GxError::UnexpectedNone{ field1: "dst xmin" })?,
        xmax: glb_col_max_dst.ok_or(GxError::UnexpectedNone{ field1: "dst xmax" })?,
        ymin: glb_row_min_dst.ok_or(GxError::UnexpectedNone{ field1: "dst ymin" })?,
        ymax: glb_row_max_dst.ok_or(GxError::UnexpectedNone{ field1: "dst ymax" })?,
    };

    let src_bounds = GeometryBounds {
        xmin: glb_col_min_src.ok_or(GxError::UnexpectedNone{ field1: "src xmin" })?,
        xmax: glb_col_max_src.ok_or(GxError::UnexpectedNone{ field1: "src xmax" })?,
        ymin: glb_row_min_src.ok_or(GxError::UnexpectedNone{ field1: "src ymin" })?,
        ymax: glb_row_max_src.ok_or(GxError::UnexpectedNone{ field1: "src ymax" })?,
    };

    Ok(Some(GridGeometriesMetrics {
        dst_bounds: dst_bounds,
        src_bounds: src_bounds,
        dst_row_edges: scan_row_dst,
        dst_col_edges: scan_col_dst,
        src_row_edges: (scan_row_src_left, scan_row_src_right),
        src_col_edges: (scan_col_src_top, scan_col_src_bottom),
        transition_matrix: GridTransitionMatrix::new(w1, w2),
    }))
}


/// Unit tests module for grid geometry.
///
/// This module contains utility functions and test cases
/// aimed at validating the behavior of grid geometry metrics
/// computations.
#[cfg(test)]
mod gx_grid_geometry_test {
    use super::*;
    //use crate::core::gx_array::{gx_array_data_approx_eq_window};
    use crate::core::gx_grid::{NoCheckGridNodeValidator, InvalidValueGridNodeValidator, MaskGridNodeValidator};

    /// Runs a test case for computing grid geometry metrics.
    ///
    /// This function computes the geometry metrics for a given grid,
    /// then compares the result against an optional expected value.
    ///
    /// The test passes if:
    /// - Both actual and expected metrics are `None`,
    /// - Or both are `Some` and approximately equal.
    ///
    /// # Type parameter
    /// - `V`: The grid node validator type implementing `GridNodeValidator<f64>`.
    ///
    /// # Arguments
    /// - `array_grid_row_in`: View of grid row data (`GxArrayView<f64>`).
    /// - `array_grid_col_in`: View of grid column data (`GxArrayView<f64>`).
    /// - `validator`: Instance of the grid node validator.
    /// - `row_oversampling`: Oversampling factor in rows.
    /// - `col_oversampling`: Oversampling factor in columns.
    /// - `expected`: Optional expected grid geometry metrics, or `None` if no expected result.
    ///
    /// # Returns
    /// - `Ok(())` if the test passes,
    /// - `Err` if the presence of actual and expected results differ,
    ///   or if the comparison fails.
    ///
    /// # Possible errors
    /// Propagates errors from the grid metrics computation function (`array1_compute_resampling_grid_geometries`).
    ///
    /// # Example
    /// ```rust,no_run
    /// let result = run_grid_test_case(&grid_row_view, &grid_col_view, &validator, 2, 2, Some(&expected_metrics));
    /// assert!(result.is_ok());
    /// ```
    fn run_grid_test_case<V: GridNodeValidator<f64>>(
        array_grid_row_in: &GxArrayView<f64>,
        array_grid_col_in: &GxArrayView<f64>,
        validator: &V,
        row_oversampling: usize,
        col_oversampling: usize,
        win: Option<GxArrayWindow>,
        expected: Option<&GridGeometriesMetrics<f64>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        
        let grid_metrics_opt = array1_compute_resampling_grid_geometries::<f64, f64, V>(
            array_grid_row_in,
            array_grid_col_in,
            row_oversampling,
            col_oversampling,
            validator,
            win,
        )?;
        match (grid_metrics_opt, expected) {
            (None, None) => {
                // Both are None => test OK
                Ok(())
            }
            (Some(grid_metrics), Some(expected_metrics)) => {
                // Both are Some => lets compare
                assert!(GridGeometriesMetrics::<f64>::approx_eq(&grid_metrics, expected_metrics, 1e-7));
                Ok(())
            }
            _ => {
                // One is Some the other is None => failure !
                Err("Mismatch between expected and actual GridGeometriesMetrics presence".into())
            }
        }
    }
    
    /// Tests that a 2x2 identity grid produces the expected geometry metrics.
    ///
    /// The grid is initialized as an identity mapping from indices to coordinates.
    /// The function validates that the computed metrics match the expected values,
    /// given oversampling factors in rows and columns.
    ///
    /// # Arguments
    /// - `row_oversampling`: Oversampling factor in the row direction.
    /// - `col_oversampling`: Oversampling factor in the column direction.
    ///
    /// # Returns
    /// Returns `Ok(())` if the test passes, or propagates any errors encountered.
    ///
    /// # Panics
    /// Panics if the computed metrics do not approximately equal the expected metrics.
    fn run_test_array1_compute_resampling_grid_geometries_identity_2x2(
            row_oversampling: usize,
            col_oversampling: usize
    ) -> Result<(), Box<dyn std::error::Error>> {
        let nrow_in = 2;
        let ncol_in = 2;
        let mut grid_row = vec![0.0; nrow_in * ncol_in];
        let mut grid_col = vec![0.0; nrow_in * ncol_in];

        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                grid_row[irow * ncol_in + icol] = irow as f64;
                grid_col[irow * ncol_in + icol] = icol as f64;
            }
        }
        
        // Init input structures
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_in, ncol_in);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_in, ncol_in);
                
        let expected = GridGeometriesMetrics {
            transition_matrix: GridTransitionMatrix {
                w1: Some((1.0 / col_oversampling as f64, 0.0)),
                w2: Some((0.0, 1.0 / row_oversampling as f64)),
            },
            dst_bounds: GeometryBounds {
                xmin: 0,
                xmax: 1,
                ymin: 0,
                ymax: 1,
            },
            src_bounds: GeometryBounds {
                xmin: 0.0,
                xmax: 1.0,
                ymin: 0.0,
                ymax: 1.0,
            },
            dst_row_edges: vec![Some((0, 1)); 2],
            dst_col_edges: vec![Some((0, 1)); 2],
            src_row_edges: (
                vec![Some((0.0, 0.0)), Some((1.0, 0.0))],
                vec![Some((0.0, 1.0)), Some((1.0, 1.0))],
            ),
            src_col_edges: (
                vec![Some((0.0, 0.0)), Some((0.0, 1.0))],
                vec![Some((1.0, 0.0)), Some((1.0, 1.0))],
            ),
        };
        
        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &NoCheckGridNodeValidator {},
            row_oversampling,
            col_oversampling,
            None,
            Some(&expected),
        )?;
        
        Ok(())
    }
    
    
    /// Tests the resampling grid geometry computation on a regular grid.
    ///
    /// Constructs a grid where each point is computed as a linear combination of 
    /// row and column vectors starting from an origin, according to the equations:
    /// 
    /// ```text
    /// grid_row[i, j] = origin_y + vec_gen_row_y * i + vec_gen_col_y * j
    /// grid_col[i, j] = origin_x + vec_gen_row_x * i + vec_gen_col_x * j
    /// ```
    ///
    /// where `i` and `j` are the row and column indices respectively.
    ///
    /// The function then verifies that the computed geometry metrics match the expected values.
    ///
    /// # Arguments
    /// - `nrow_in`: Number of rows in the input grid.
    /// - `ncol_in`: Number of columns in the input grid.
    /// - `vec_gen_col`: Column vector (x, y) for grid generation.
    /// - `vec_gen_row`: Row vector (x, y) for grid generation.
    /// - `origin_gen`: Origin point (x, y) for the grid.
    /// - `row_oversampling`: Oversampling factor in the row direction.
    /// - `col_oversampling`: Oversampling factor in the column direction.
    ///
    /// # Returns
    /// Returns `Ok(())` if the computed metrics match expectations, otherwise propagates errors.
    ///
    /// # Panics
    /// Panics if the computed metrics do not approximately equal the expected metrics.
    fn run_test_array1_compute_resampling_grid_geometries_regular_grid(
            nrow_in: usize,
            ncol_in: usize,
            vec_gen_col: (f64, f64), //(x, y)
            vec_gen_row: (f64, f64), //(x, y)
            origin_gen: (f64, f64), // (x, y)
            row_oversampling: usize,
            col_oversampling: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut grid_row = vec![0.0; nrow_in * ncol_in];
        let mut grid_col = vec![0.0; nrow_in * ncol_in];

        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                grid_row[irow * ncol_in + icol] = origin_gen.1 + vec_gen_row.1 * (irow as f64) + vec_gen_col.1 * (icol as f64);
                grid_col[irow * ncol_in + icol] = origin_gen.0 + vec_gen_row.0 * (irow as f64) + vec_gen_col.0 * (icol as f64);
            }
        }
        let n = ncol_in * nrow_in;
        let mut xmin = grid_col[0];
        let mut xmax = grid_col[0];
        let mut ymin = grid_row[0];
        let mut ymax = grid_row[0];
        for i in 1..n {
            xmin = min_partial(xmin, grid_col[i]);
            xmax = max_partial(xmax, grid_col[i]);
            ymin = min_partial(ymin, grid_row[i]);
            ymax = max_partial(ymax, grid_row[i]);
        }
        
        // Init input structures
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_in, ncol_in);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_in, ncol_in);
        
        // Define expected
        let expected_dst_row_edges = vec![Some((0, ncol_in-1)); nrow_in];
        let expected_dst_col_edges = vec![Some((0, nrow_in-1)); ncol_in];
        
        let expected_src_row_edges_0: Vec<Option<(f64, f64)>> = (0..nrow_in).map(|irow| {
            let icol_start = expected_dst_row_edges[irow].unwrap().0;
            let idx0 = irow * ncol_in + icol_start;
            Some((grid_row[idx0], grid_col[idx0]))
        }).collect();
        
        let expected_src_row_edges_1: Vec<Option<(f64, f64)>> = (0..nrow_in).map(|irow| {
            let icol_stop = expected_dst_row_edges[irow].unwrap().1;
            let idx1 = irow * ncol_in + icol_stop;
            Some((grid_row[idx1], grid_col[idx1]))
        }).collect();
        
        let expected_src_col_edges_0: Vec<Option<(f64, f64)>> = (0..ncol_in).map(|icol| {
            let irow_start = expected_dst_col_edges[icol].unwrap().0;
            let idx0 = irow_start * ncol_in + icol;
            Some((grid_row[idx0], grid_col[idx0]))
        }).collect();

        let expected_src_col_edges_1: Vec<Option<(f64, f64)>> = (0..ncol_in).map(|icol| {
            let irow_stop = expected_dst_col_edges[icol].unwrap().1;
            let idx1 = irow_stop * ncol_in + icol;
            Some((grid_row[idx1], grid_col[idx1]))
        }).collect();
        
        let expected = GridGeometriesMetrics {
            transition_matrix: GridTransitionMatrix {
                w1: Some((vec_gen_col.0 / col_oversampling as f64, vec_gen_col.1 / col_oversampling as f64)),
                w2: Some((vec_gen_row.0 / row_oversampling as f64, vec_gen_row.1 / row_oversampling as f64)),
            },
            dst_bounds: GeometryBounds {
                xmin: 0,
                xmax: ncol_in - 1,
                ymin: 0,
                ymax: nrow_in - 1,
            },
            src_bounds: GeometryBounds {
                xmin: xmin,
                xmax: xmax,
                ymin: ymin,
                ymax: ymax,
            },
            dst_row_edges: expected_dst_row_edges,
            dst_col_edges: expected_dst_col_edges,
            src_row_edges: (
                expected_src_row_edges_0,
                expected_src_row_edges_1,
            ),
            src_col_edges: (
                expected_src_col_edges_0,
                expected_src_col_edges_1,
            ),
        };
        
        // Run and compare
        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &NoCheckGridNodeValidator {},
            row_oversampling,
            col_oversampling,
            None,
            Some(&expected),
        )?;
        
        Ok(())
    }
    
    /// Tests resampling grid geometry computation on an irregular grid with masking support.
    ///
    /// This test verifies correct handling of grid metrics under three scenarios:
    ///
    /// ## Case 1: No mask
    /// Computes the column and row generation vectors (`vcol`, `vrow`) based on the full grid bounds:
    ///
    /// ```text
    /// vcol = [ (col_max - col_min) / (ncol - 1),
    ///          (row_max - row_min) / (ncol - 1) ]
    ///
    /// vrow = [ (col_max - col_min) / (nrow - 1),
    ///          (row_max - row_min) / (nrow - 1) ]
    /// ```
    ///
    /// Here, the entire grid is considered, without masking.
    ///
    /// ## Case 2: Raster mask applied
    /// The grid nodes are filtered by the mask, restricting computations only to valid nodes:
    ///
    /// ```text
    /// vcol_masked = [ (col_max_masked - col_min_masked) / (valid_cols - 1),
    ///                 (row_max_masked - row_min_masked) / (valid_cols - 1) ]
    ///
    /// vrow_masked = [ (col_max_masked - col_min_masked) / (valid_rows - 1),
    ///                 (row_max_masked - row_min_masked) / (valid_rows - 1) ]
    /// ```
    ///
    /// Only masked nodes marked as valid (e.g. mask value = 1) are used.
    ///
    /// ## Case 3: Invalid value validator
    /// Similar to the mask case, nodes with invalid values (e.g. -999) are excluded from calculations.
    ///
    /// # Returns
    /// Returns `Ok(())` if the computed geometry metrics match expected results for each case,
    /// otherwise returns an error.
    ///
    /// # Panics
    /// Panics if computed metrics differ from expectations.
    fn run_test_array1_compute_resampling_grid_geometries_grid_w_mask() -> Result<(), Box<dyn std::error::Error>> {
        let row_oversampling = 1;
        let col_oversampling = 1;
        let nrow_in = 3;
        let ncol_in = 4;

        // Irregular test grid
        // Build a grid that intentionally violates the main assumption.
        // Irregular spacing is used to validate proper masking functionality.    
        let grid_row = [0.0, 1.0, 2.0, 4.0,
                8.0, 16.0, 32.0, 64.0,
                128.0, 256.0, 512.0, 1024.0];
        let grid_col = [0.0, 1.0, 2.0, 3.0,
                0.0, 10.0, 20.0, 30.0,
                0.0, 100.0, 200.0, 300.0];
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_in, ncol_in);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_in, ncol_in);

        // Case 1: No mask
        //----------------------------------------------------------------------
        // Without masking, the col_gen_vec should use the first and last item
        // on 1st row, ie. :
        //
        //        ┌                    ┐   ┌           ┐
        // vcol = │ (3.0 - 0.0) / 3.0  │ = │    1.0    │
        //        │ (4.0 - 0.0) / 3.0  │   │ 4.0 / 3.0 │
        //        └                    ┘   └           ┘ 
        //
        //        ┌                      ┐   ┌      ┐
        // vrow = │   (0.0 - 0.0) / 2.0  │ = │  0.0 │
        //        │ (128.0 - 0.0) / 2.0  │   │ 64.0 │
        //        └                      ┘   └      ┘ 
        let expected_no_mask = GridGeometriesMetrics {
            transition_matrix: GridTransitionMatrix {
                w1: Some((1.0, 4.0 / 3.0)),
                w2: Some((0.0, 64.0)),
            },
            dst_bounds: GeometryBounds {
                xmin: 0,
                xmax: 3,
                ymin: 0,
                ymax: 2,
            },
            src_bounds: GeometryBounds {
                xmin: 0.0,
                xmax: 300.0,
                ymin: 0.0,
                ymax: 1024.0,
            },
            dst_row_edges: vec![Some((0, 3)); 3],
            dst_col_edges: vec![Some((0, 2)); 4],
            src_row_edges: (
                vec![Some((0.0, 0.0)), Some((8.0, 0.0)), Some((128.0, 0.0))],
                vec![Some((4.0, 3.0)), Some((64.0, 30.0)), Some((1024.0, 300.0))],
            ),
            src_col_edges: (
                vec![Some((0.0, 0.0)), Some((1.0, 1.0)), Some((2.0, 2.0)), Some((4.0, 3.0))],
                vec![Some((128.0, 0.0)), Some((256.0, 100.0)), Some((512.0, 200.0)), Some((1024.0, 300.0))],
            ),
        };

        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &NoCheckGridNodeValidator {},
            row_oversampling,
            col_oversampling,
            None,
            Some(&expected_no_mask),
        )?;

        // Case 2: Raster mask
        // ----------------------------------------------------------------------
        // With the specified mask, we should get
        //
        //        ┌                        ┐   ┌         ┐
        // vcol = │ (300.0 - 100.0) / 2.0  │ = │  100.0  │
        //        │ (1024.0 - 256.0) / 2.0 │   │  384.0  │
        //        └                        ┘   └         ┘ 
        //
        //        ┌                       ┐   ┌       ┐
        // vrow = │ (200.0 - 20.0) / 1.0  │ = │ 180.0 │
        //        │ (512.0 - 32.0) / 1.0  │   │ 480.0 │
        //        └                       ┘   └       ┘ 
        let mask = [
            0, 0, 0, 0,
            0, 0, 1, 1,
            0, 1, 1, 1,
        ];
        let mask_view = GxArrayView::new(&mask, 1, nrow_in, ncol_in);
        let expected_metrics_mask = GridGeometriesMetrics {
            transition_matrix: GridTransitionMatrix {
                w1: Some((100.0, 384.0)),
                w2: Some((180.0, 480.0)),
            },
            dst_bounds: GeometryBounds {
                xmin: 1,
                xmax: 3,
                ymin: 1,
                ymax: 2,
            },
            src_bounds: GeometryBounds {
                xmin: 20.0,
                xmax: 300.0,
                ymin: 32.0,
                ymax: 1024.0,
            },
            dst_row_edges: vec![None, Some((2, 3)), Some((1, 3))],
            dst_col_edges: vec![None, Some((2, 2)), Some((1, 2)), Some((1, 2))],
            src_row_edges: (
                vec![None, Some((32.0, 20.0)), Some((256.0, 100.0))],
                vec![None, Some((64.0, 30.0)), Some((1024.0, 300.0))],
            ),
            src_col_edges: (
                vec![None, Some((256.0, 100.0)), Some((32.0, 20.0)), Some((64.0, 30.0))],
                vec![None, Some((256.0, 100.0)), Some((512.0, 200.0)), Some((1024.0, 300.0))],
            ),
        };

        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &MaskGridNodeValidator { mask_view: &mask_view, valid_value: 1 },
            row_oversampling,
            col_oversampling,
            None,
            Some(&expected_metrics_mask),
        )?;

        // Case 3: Invalid value
        let grid_row_iv = [
            -999., -999., -999., -999.,
            -999., -999., 32.0, 64.0,
            -999., 256.0, 512.0, 1024.0,
        ];
        let grid_col_iv = [
            -999., -999., -999., -999.,
            -999., -999., 20.0, 30.0,
            -999., 100.0, 200.0, 300.0,
        ];
        let array_grid_row_iv = GxArrayView::new(&grid_row_iv, 1, nrow_in, ncol_in);
        let array_grid_col_iv = GxArrayView::new(&grid_col_iv, 1, nrow_in, ncol_in);
        run_grid_test_case(
            &array_grid_row_iv,
            &array_grid_col_iv,
            &InvalidValueGridNodeValidator { invalid_value: -999., epsilon: 1e-7 },
            row_oversampling,
            col_oversampling,
            None,
            Some(&expected_metrics_mask), // Same expectation as mask case
        )?;

        Ok(())
    }
    
    /// Tests resampling grid geometry metrics on edge cases with minimal and masked grids.
    ///
    /// This test validates correct behavior of grid metrics computation in three limit cases:
    ///
    /// ## Case 1: All zero grid coordinates
    /// A grid where all coordinates are zero results in zero transition vectors:
    ///
    /// ```text
    /// w1 = (0, 0), w2 = (0, 0)
    /// src_bounds = { xmin = xmax = ymin = ymax = 0 }
    /// dst_bounds = full index range (0..1)
    /// ```
    ///
    /// Metrics are valid but reflect zero spatial extent.
    ///
    /// ## Case 2: Fully masked grid
    /// When all grid nodes are masked out (invalid), no valid geometry can be computed,
    /// thus metrics must be `None`.
    ///
    /// ## Case 3: Single valid masked node
    /// When only one node is valid (unmasked), transition vectors cannot be computed (None),
    /// but source and destination bounds and edges reflect this single valid cell.
    ///
    /// # Returns
    /// Returns `Ok(())` if all cases produce expected results; otherwise returns an error.
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_limit_cases() -> Result<(), Box<dyn std::error::Error>> {
        let row_oversampling = 1;
        let col_oversampling = 1;
        let nrow_in = 2;
        let ncol_in = 2;

        // Test 1 : Irregular test grid
        // Build a grid that intentionally violates the main assumption.
        // Irregular spacing is used to validate proper masking functionality.    
        let grid_row = [0.0, 0.0, 0.0, 0.0];
        let grid_col = [0.0, 0.0, 0.0, 0.0];
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_in, ncol_in);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_in, ncol_in);
        
        // In that case the grid_metrics is Ok but have zero matrix and vectors.
        let expected_zero_array = GridGeometriesMetrics {
            transition_matrix: GridTransitionMatrix {
                w1: Some((0., 0.)),
                w2: Some((0., 0.)),
            },
            dst_bounds: GeometryBounds {
                xmin: 0,
                xmax: 1,
                ymin: 0,
                ymax: 1,
            },
            src_bounds: GeometryBounds {
                xmin: 0.,
                xmax: 0.,
                ymin: 0.,
                ymax: 0.,
            },
            dst_row_edges: vec![Some((0, 1)); 2],
            dst_col_edges: vec![Some((0, 1)); 2],
            src_row_edges: (
                vec![Some((0.0, 0.0)), Some((0.0, 0.0))],
                vec![Some((0.0, 0.0)), Some((0.0, 0.0))],
            ),
            src_col_edges: (
                vec![Some((0.0, 0.0)), Some((0.0, 0.0))],
                vec![Some((0.0, 0.0)), Some((0.0, 0.0))],
            ),
        };
        
        // Run and compare
        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &NoCheckGridNodeValidator {},
            row_oversampling,
            col_oversampling,
            None,
            Some(&expected_zero_array),
        )?;
        
        // Test 2 : full masked data        
        // In that case the grid_metrics must be None => no valid cell for computation.
        let mask = [
            0, 0, 0, 0,
        ];
        let mask_view = GxArrayView::new(&mask, 1, nrow_in, ncol_in);
        
        // Run and compare
        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &MaskGridNodeValidator { mask_view: &mask_view, valid_value: 1 },
            row_oversampling,
            col_oversampling,
            None,
            None,
        )?;

        // Test 3 : Irregular test grid
        // In that case the grid_metrics must be None => no valid cell for computation.
        let mask_2 = [
            0, 0, 1, 0,
        ];
        let mask_view_2 = GxArrayView::new(&mask_2, 1, nrow_in, ncol_in);
        
        let expected_one_valid = GridGeometriesMetrics::<f64> {
            transition_matrix: GridTransitionMatrix {
                w1: None,
                w2: None,
            },
            dst_bounds: GeometryBounds {
                xmin: 0,
                xmax: 0,
                ymin: 1,
                ymax: 1,
            },
            src_bounds: GeometryBounds {
                xmin: 0.0,
                xmax: 0.0,
                ymin: 0.0,
                ymax: 0.0,
            },
            dst_row_edges: vec![None, Some((0, 0))],
            dst_col_edges: vec![Some((1, 1)), None],
            src_row_edges: (
                vec![None, Some((0., 0.))],
                vec![None, Some((0., 0.))],
            ),
            src_col_edges: (
                vec![Some((0., 0.)), None],
                vec![Some((0., 0.)), None],
            ),
        };
        // Run and compare
        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &MaskGridNodeValidator { mask_view: &mask_view_2, valid_value: 1 },
            row_oversampling,
            col_oversampling,
            None,
            Some(&expected_one_valid),
        )?;

        Ok(())
    }
    
    /// Tests resampling grid geometry metrics with window
    fn run_test_array1_compute_resampling_grid_geometries_5x7_win_2_4_1_2(
            row_oversampling: usize,
            col_oversampling: usize
    ) -> Result<(), Box<dyn std::error::Error>> {
        let nrow_in = 5;
        let ncol_in = 7;
        let mut grid_row = vec![0.0; nrow_in * ncol_in];
        let mut grid_col = vec![0.0; nrow_in * ncol_in];

        let win = GxArrayWindow{ start_col: 2, end_col: 4, start_row:1, end_row:2 };

        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {                
                if icol < win.start_col || icol > win.end_col || irow < win.start_row || irow > win.end_row {
                    grid_row[irow * ncol_in + icol] = 100. * irow as f64;
                    grid_col[irow * ncol_in + icol] = 100. * icol as f64;
                }
                else {
                    grid_row[irow * ncol_in + icol] = irow as f64;
                    grid_col[irow * ncol_in + icol] = icol as f64;
                }
            }
        }

        // Init input structures
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_in, ncol_in);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_in, ncol_in);
        
        // Define expected
        let expected_dst_row_edges = vec![Some((win.start_col, win.end_col)); win.height()];
        let expected_dst_col_edges = vec![Some((win.start_row, win.end_row)); win.width()];
        
        let expected_src_row_edges_0: Vec<Option<(f64, f64)>> = (0..win.height()).map(|irow| {
            let icol_start = expected_dst_row_edges[irow].unwrap().0;
            let idx0 = (win.start_row + irow) * ncol_in + icol_start;
            Some((grid_row[idx0], grid_col[idx0]))
        }).collect();
        
        let expected_src_row_edges_1: Vec<Option<(f64, f64)>> = (0..win.height()).map(|irow| {
            let icol_stop = expected_dst_row_edges[irow].unwrap().1;
            let idx1 = (win.start_row + irow) * ncol_in + icol_stop;
            Some((grid_row[idx1], grid_col[idx1]))
        }).collect();
        
        let expected_src_col_edges_0: Vec<Option<(f64, f64)>> = (0..win.width()).map(|icol| {
            let irow_start = expected_dst_col_edges[icol].unwrap().0;
            let idx0 = irow_start * ncol_in + icol + win.start_col;
            Some((grid_row[idx0], grid_col[idx0]))
        }).collect();

        let expected_src_col_edges_1: Vec<Option<(f64, f64)>> = (0..win.width()).map(|icol| {
            let irow_stop = expected_dst_col_edges[icol].unwrap().1;
            let idx1 = irow_stop * ncol_in + icol + win.start_col;
            Some((grid_row[idx1], grid_col[idx1]))
        }).collect();
        
        
        let expected = GridGeometriesMetrics {
            transition_matrix: GridTransitionMatrix {
                w1: Some((1.0 / col_oversampling as f64, 0.0)),
                w2: Some((0.0, 1.0 / row_oversampling as f64)),
            },
            dst_bounds: GeometryBounds {
                xmin: win.start_col,
                xmax: win.end_col,
                ymin: win.start_row,
                ymax: win.end_row,
            },
            src_bounds: GeometryBounds {
                xmin: win.start_col as f64,
                xmax: win.end_col as f64,
                ymin: win.start_row as f64,
                ymax: win.end_row as f64,
            },
            dst_row_edges: expected_dst_row_edges,
            dst_col_edges: expected_dst_col_edges,
            src_row_edges: (
                expected_src_row_edges_0,
                expected_src_row_edges_1,
            ),
            src_col_edges: (
                expected_src_col_edges_0,
                expected_src_col_edges_1,
            ),
        };
        
        run_grid_test_case(
            &array_grid_row_in,
            &array_grid_col_in,
            &NoCheckGridNodeValidator {},
            row_oversampling,
            col_oversampling,
            Some(win),
            Some(&expected),
        )?;
        
        Ok(())
    }
    
    
    #[test]
    fn test_array1_compute_resampling_grid_geometries_identity_2x2_no_oversampling() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_identity_2x2(1, 1)
    }

    #[test]
    fn test_array1_compute_resampling_grid_geometries_identity_2x2_with_oversampling() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_identity_2x2(5, 4)
    }
    
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_regular_grid_idendity_2x2() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_regular_grid(2, 2, (0., 1.), (1., 0.), (0., 0.), 1, 1)
    }
    
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_regular_grid_2x2() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_regular_grid(2, 2, (0.2, 0.7), (1., 0.3), (0., 0.), 1, 1)
    }
    
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_regular_grid_2x2_shift_origin() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_regular_grid(2, 2, (0.2, 0.7), (1., 0.3), (10.2, -3.5), 1, 1)
    }
    
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_regular_grid_2x2_shift_origin_with_oversampling() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_regular_grid(2, 2, (0.2, 0.7), (1., 0.3), (10.2, -3.5), 2, 3)
    }
    
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_regular_grid_3x4() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_regular_grid(3, 4, (0., 1.), (1., 0.), (0., 10.), 1, 1)
    }
    
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_grid_w_mask_() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_grid_w_mask()
    }
    
    #[test]
    fn run_test_array1_compute_resampling_grid_geometries_5x7_win_2_4_1_2_res_1_1() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_5x7_win_2_4_1_2(1, 1)
    }
    
    /*#[test]
    fn run_test_array1_compute_resampling_grid_geometries_5x7_win_2_4_1_2_res_3_2() -> Result<(), Box<dyn std::error::Error>> {
        run_test_array1_compute_resampling_grid_geometries_5x7_win_2_4_1_2(3, 2)
    }*/
}