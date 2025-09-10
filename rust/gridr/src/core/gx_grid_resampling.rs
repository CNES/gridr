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
//! # Grid Resampling Utilities and Validation Framework
//!
//! This module provides a complete testing and validation setup for resampling routines based
//! on grid mesh definitions. It includes key components used in grid-based interpolation tasks,
//! such as `GridMesh`, various implementations of the `GridMeshValidator` trait, and utility
//! methods for accuracy testing.
//!
//! ## Key Components
//!
//! - [`GridMesh`]: Defines a quadrilateral cell in a 2D source grid and is used as the interpolation
//!   primitive. It supports efficient iteration over rows and columns.
//!
//! - [`GridMeshValidator`]: A trait that abstracts the logic for determining whether a mesh is valid
//!   based on value content or an external mask. Includes multiple implementations:
//!   - `NoCheckGridMeshValidator`: Always returns valid.
//!   - `InvalidValueGridMeshValidator`: Flags invalid cells based on a reference "no data" value.
//!   - `MaskGridMeshValidator`: Uses a mask array to determine validity per cell.
//!
//! - [`array1_grid_resampling`]: The core function tested in this module. It performs resampling
//!   using a provided interpolation kernel and optionally applies mesh validation.
//!
//! ## Interpolation Context Abstraction
//!
//! At the heart of the interpolation machinery lies the [`GxArrayViewInterpolationContextTrait`],
//! which abstracts how the interpolator interacts with:
//!
//! - **Input validity**: Whether each neighborhood point should be included in the computation,
//!   based on either a binary mask, value filtering, or no filtering at all.
//! - **Output validity**: Whether to write a result validity mask or not.
//! - **Bounds checking**: Whether accesses outside the input array bounds should be allowed, clamped,
//!   or skipped entirely.
//!
//! These strategies are encoded in types implementing `GxArrayViewInterpolationContextTrait`, and passed
//! generically to the interpolator. This enables:
//!
//! - **Compile-time selection** of behavior with zero runtime overhead (monomorphization).
//! - **Composability**: strategies can be combined or swapped without changing the core algorithm.
//! - **Safety**: unchecked code is only used when the context guarantees its validity.
//!
//! ## Generic Design and Performance
//!
//! The framework relies heavily on Rust's trait system and generics to provide high-performance,
//! flexible and reusable resampling routines:
//!
//! - Interpolators are generic over:
//!   - The input and output scalar types (`T`, `V`), typically `f64`, `f32`, etc.
//!   - The interpolation strategy (e.g., bilinear, B-spline).
//!   - The context type (`IC`), which controls validity and boundary behavior.
//!
//! - This design supports both safe and unsafe (optimized) execution paths, with clear contracts
//!   for when each may be used.
use crate::core::gx_array::{GxArrayWindow, GxArrayView, GxArrayViewMut};
use crate::core::interp::gx_array_view_interp::{GxArrayViewInterpolator, GxArrayViewInterpolationContextTrait, GxArrayViewInterpolationContext, GxArrayViewInterpolatorOutputMaskStrategy, NoInputMask, BinaryInputMask, NoOutputMask, BinaryOutputMask, BoundsCheck, NoBoundsCheck};
use crate::core::gx_errors::GxError;

/// A trait that standardizes grid cell validation logic within a mesh-based computation.
///
/// Implementors of this trait define a validation method used to determine whether
/// a grid cell (represented by a `GridMesh`) is valid for further computation. 
/// This allows for pluggable and reusable validation strategies, such as checking
/// for invalid values or mask-based exclusion.
///
/// # Type Parameters
///
/// * `W` - The type of data stored in the grid array (e.g., `f64`, `u8`, etc.).
pub trait GridMeshValidator<W>
{
    /// Validates whether the current grid position is suitable for computation.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The current grid mesh element to be validated.
    /// * `out_idx` - A mutable reference to an output index (can be used or modified during validation).
    /// * `grid_view` - A view into the grid data being validated.
    ///
    /// # Returns
    ///
    /// Returns `true` if the grid cell is valid and can be processed, `false` otherwise.
    fn validate<'a>(&self, mesh: &'a mut GridMesh, out_idx: &'a mut usize, grid_view: &GxArrayView<'a, W>) -> bool;
}

/// A validator implementation that unconditionally accepts all grid positions.
///
/// This is the simplest implementation of `GridMeshValidator`, which always returns `true`,
/// effectively disabling any validation logic.
///
/// Useful as a default or placeholder when no filtering is required.
#[derive(Debug)]
pub struct NoCheckGridMeshValidator {
}

impl<W> GridMeshValidator<W> for NoCheckGridMeshValidator{
    
    #[inline]
    fn validate<'a>(&self, _mesh: &'a mut GridMesh, _out_idx: &'a mut usize, _grid_view: &GxArrayView<'a, W>) -> bool
    {
        true
    }
}

/// A validator that excludes grid positions based on a specific invalid value.
///
/// This implementation of `GridMeshValidator` considers a cell invalid if any of the four
/// nodes in the mesh are within a small threshold (`epsilon`) of a specified `invalid_value`.
///
/// This is typically used to ignore missing or masked data encoded as sentinel values
/// (e.g., -9999.0).
///
/// # Fields
///
/// * `invalid_value` - The sentinel value that represents invalid or missing data.
/// * `epsilon` - The tolerance used to compare against `invalid_value`.
#[derive(Debug)]
pub struct InvalidValueGridMeshValidator {
    /// The sentinel value that represents invalid or missing data.
    pub invalid_value: f64,
    /// The tolerance used to compare against `invalid_value`.
    pub epsilon: f64,
}


impl InvalidValueGridMeshValidator {
    
    /// Checks whether the value at a given node in the grid view is considered invalid,
    /// based on a predefined invalid value and an epsilon tolerance.
    ///
    /// This method converts the value at the specified node to `f64`, then compares it
    /// to the `invalid_value` defined in the validator. If the absolute difference between
    /// the two is less than `epsilon`, the value is considered invalid.
    ///
    /// # Type Parameters
    /// - `W`: The data type stored in the grid view. It must be copyable and convertible to `f64`.
    ///
    /// # Parameters
    /// - `node`: The index of the node in the grid view to validate.
    /// - `grid_view`: A reference to the grid data structure containing the value to check.
    ///
    /// # Returns
    /// `true` if the value at the given node is within `epsilon` of the `invalid_value`,  
    /// otherwise `false`.
    #[inline] 
    fn is_invalid<W>(&self, node: usize, grid_view: &GxArrayView<W>) -> bool
    where
        W: Into<f64> + Copy,
    {
        (grid_view.data[node].into() - self.invalid_value).abs() < self.epsilon
    }
}

impl<W> GridMeshValidator<W> for InvalidValueGridMeshValidator
where
    W: Into<f64> + Copy,
{
    /// Validates whether a mesh cell should be processed based on the validity of its nodes
    /// in the associated grid view.
    ///
    /// The validation logic depends on the grid oversampling factors. Depending on whether
    /// rows, columns, or both are oversampled, a subset of the cell's nodes are checked.
    /// A node is considered invalid if its value is sufficiently close (within `epsilon`)
    /// to a predefined `invalid_value`. If any of the relevant nodes are invalid,
    /// the cell is rejected (returns `false`).
    ///
    /// A cell's node must be actively involved in the targeted interpolation, meaning its associated weight must be
    /// non-zero, to be included in the validity check.
    ///
    /// This method delegates node-level checks to `is_invalid`.
    ///
    /// # Type Parameters
    /// - `W`: The scalar type of the grid data values, which must be convertible to `f64` and `Copy`.
    ///
    /// # Parameters
    /// - `mesh`: Mutable reference to the mesh structure containing node indices and oversampling metadata.
    /// - `_out_idx`: An unused mutable reference to an output index (retained for interface compatibility).
    /// - `grid_view`: A view of the grid data from which values are retrieved for validation.
    ///
    /// # Returns
    /// `true` if all relevant nodes are considered valid; `false` otherwise.
    ///
    /// # Inlined
    /// This method is marked `#[inline]` to encourage inlining during performance-critical loops.
    #[inline]
    fn validate<'a>(&self, mesh: &'a mut GridMesh, _out_idx: &'a mut usize, grid_view: &GxArrayView<'a, W>) -> bool {
        
        match (mesh.grid_row_oversampling, mesh.grid_col_oversampling) {
            // Both are different from 1
            (r, c) if r != 1 && c != 1 => {
                if ((mesh.gmi_w1 > 0) && self.is_invalid(mesh.node1, &grid_view)) ||
                        ((mesh.gmi_w2 > 0) && self.is_invalid(mesh.node2, &grid_view)) ||
                        ((mesh.gmi_w3 > 0) && self.is_invalid(mesh.node3, &grid_view)) ||
                        ((mesh.gmi_w4 > 0) && self.is_invalid(mesh.node4, &grid_view)) {
                    return false;
                }
                true
            },                    
            // Only rows are oversampled (note : changed node3 to node4)
            (r, 1) if r != 1 => {
                if ((mesh.gmi_w1 > 0) && self.is_invalid(mesh.node1, &grid_view)) || 
                        ((mesh.gmi_w4 > 0) && self.is_invalid(mesh.node4, &grid_view)) {
                    return false;
                }
                true
            },

            // Only columns are oversampled
            (1, c) if c != 1 => {
                if ((mesh.gmi_w1 > 0) && self.is_invalid(mesh.node1, &grid_view)) ||
                        ((mesh.gmi_w2 > 0) && self.is_invalid(mesh.node2, &grid_view)) {
                    return false;
                }
                true
            },

            // Default (1, 1)
            _ => !self.is_invalid(mesh.node1, &grid_view),
        }
    }
}

/// A validator that uses a binary mask array to determine validity.
///
/// This implementation of `GridMeshValidator` checks an auxiliary mask array to determine
/// whether a grid cell is valid. If any node in the mesh does not correspond to the `valid_value`
/// value, the cell is considered invalid.
///
///
/// A cell's node must be actively involved in the targeted interpolation, meaning its associated weight must be
/// non-zero, to be included in the validity check.
///
/// This is commonly used for excluding regions via precomputed masks (e.g., land/sea masks).
///
/// # Fields
///
/// * `mask_view` - A reference to a `GxArrayView` containing `u8` mask values.
/// * `valid_value` - That value indicates valid, and any different value indicates invalid.
#[derive(Debug)]
pub struct MaskGridMeshValidator<'a> {
    /// A reference to a `GxArrayView` containing `u8` mask values.
    pub mask_view: &'a GxArrayView<'a, u8>,
    /// That value indicates valid, and any different value indicates invalid.
    pub valid_value: u8,
}

impl<'a, W> GridMeshValidator<W> for MaskGridMeshValidator<'a>
where
    W: Copy,
{
    #[inline]
    fn validate(&self, mesh: &mut GridMesh, _out_idx: &mut usize, _grid_view: &GxArrayView<W>) -> bool
    {
        let mask = &self.mask_view.data;
        
        let node1_valid = (mesh.gmi_w1 == 0) || (mask[mesh.node1] == self.valid_value);
        let node2_valid = (mesh.gmi_w2 == 0) || (mask[mesh.node2] == self.valid_value);
        let node3_valid = (mesh.gmi_w3 == 0) || (mask[mesh.node3] == self.valid_value);
        let node4_valid = (mesh.gmi_w4 == 0) || (mask[mesh.node4] == self.valid_value);

        match (mesh.grid_row_oversampling, mesh.grid_col_oversampling) {
            // Both are different from 1
            (r, c) if r != 1 && c != 1 => node1_valid && node2_valid && node3_valid && node4_valid,

            // Only rows are oversampled (note : changed node3 to node4)
            (r, 1) if r != 1 => node1_valid && node4_valid,

            // Only columns are oversampled
            (1, c) if c != 1 => node1_valid && node2_valid,

            // Default (1, 1)
            _ => node1_valid,
        }
    }
}


/// Represents a 2D quadrilateral mesh used for bilinear interpolation over a grid.
///
/// `GridMesh` defines a rectangular cell within a low-resolution source grid. It is primarily used
/// in interpolation routines to compute target values at higher resolutions. The mesh is defined by
/// four corner nodes, indexed into a flat 1D representation of the 2D source grid.
///
/// The corner nodes are laid out in a clockwise order, starting from the top-left corner:
///
/// ```text
///     (node1: upper left)  +--------------+  (node2: upper right)
///                          |              |
///                          |              |
///                          |              |
///                          |              |
///                          |              |
///     (node4: bottom left) +--------------+  (node3: bottom right)
/// ```
///
/// When reaching the last row or column of the grid, the mesh is adjusted to become a degenerate
/// "vertical" or "horizontal" mesh with zero width (i.e., some corners are collapsed) to safely
/// handle grid boundaries.
///
/// # Fields
///
/// - `node1`: Index of the top-left corner of the mesh.
/// - `node2`: Index of the top-right corner of the mesh.
/// - `node3`: Index of the bottom-right corner of the mesh.
/// - `node4`: Index of the bottom-left corner of the mesh.
/// - `grid_nrow`: Total number of rows in the parent source grid.
/// - `grid_ncol`: Total number of columns in the parent source grid.
/// - `grid_row_oversampling`: The grid's oversampling for rows.
/// - `grid_col_oversampling`: The grid's oversampling for columns.
/// - `window_src`: The window applied on the native source grid to restrict the region of interpolation.
/// - `window_rel`: The window applied on the `window_src` restricted grid to define the oversampled production window. It is mainly used to set the starting positions.
/// - `gmi_col_idx` : The current column index within a mesh
/// - `gmi_col_idx_t` : The current column index complement within a mesh
/// - `gmi_row_idx` : The current row index within a mesh
/// - `gmi_row_idx_t` : The current row index complement within a mesh
/// - `gmi_w1` : The current non-normalized integer weight associated to `node1`
/// - `gmi_w2` : The current non-normalized integer weight associated to `node2`
/// - `gmi_w3` : The current non-normalized integer weight associated to `node3`
/// - `gmi_w4` : The current non-normalized integer weight associated to `node4`
///
/// # Usage
///
/// Typically, a `GridMesh` is iterated column-wise and row-wise over a `GxArrayWindow`, updating
/// its internal corner nodes using [`next_src_col`] and [`next_src_row`] as needed.
///
/// [`next_src_col`]: Self::next_src_col
/// [`next_src_row`]: Self::next_src_row
#[derive(Debug)]
pub struct GridMesh<'a> {
    node1: usize,
    node2: usize,
    node3: usize,
    node4: usize,
    grid_nrow: usize,
    grid_ncol: usize,
    grid_row_oversampling: usize,
    grid_col_oversampling: usize,
    window_src: &'a GxArrayWindow, 
    window_rel: &'a GxArrayWindow,
    gmi_col_idx: usize,
    gmi_col_idx_t: usize,
    gmi_row_idx: usize,
    gmi_row_idx_t: usize,
    gmi_w1: usize,
    gmi_w2: usize,
    gmi_w3: usize,
    gmi_w4: usize,
}

impl<'a> GridMesh<'a> {
    
    /// Creates a new `GridMesh` from the dimensions of a source grid and a source window.
    ///
    /// The mesh is initially positioned at the top-left cell of the provided window.
    ///
    /// # Arguments
    ///
    /// - `grid_nrow`: Number of rows in the source grid.
    /// - `grid_ncol`: Number of columns in the source grid.
    /// - `grid_row_oversampling`: The grid's oversampling for rows.
    /// - `grid_col_oversampling`: The grid's oversampling for columns.
    /// - `window_src`: A reference to a window on the source grid defining the region to iterate over.
    /// - `window_rel`: A reference to a window on the oversampled grid defining the region to iterate over.
    ///
    /// # Returns
    ///
    /// A new `GridMesh` positioned at the top-left corner of `window_src`.
    #[inline]
    pub fn new(
            grid_nrow: usize,
            grid_ncol: usize,
            grid_row_oversampling: usize,
            grid_col_oversampling: usize,
            window_src: &'a GxArrayWindow,
            window_rel: &'a GxArrayWindow,
        ) -> Self
    {
        let node1 = window_src.start_row * grid_ncol + window_src.start_col;
        
        // The init idx is given by the relative position in the source window.
        let gmi_row_idx: usize = window_rel.start_row;
        let gmi_col_idx: usize = window_rel.start_col;
        
        // Complement of the relative position in current grid mesh 
        let gmi_col_idx_t: usize = grid_col_oversampling - gmi_col_idx;
        let gmi_row_idx_t: usize = grid_row_oversampling - gmi_row_idx;
        
        
        let gmi_w1: usize = gmi_col_idx_t * gmi_row_idx_t;
        let gmi_w2: usize = gmi_col_idx * gmi_row_idx_t;
        let gmi_w3: usize = gmi_col_idx * gmi_row_idx;
        let gmi_w4: usize = gmi_col_idx_t * gmi_row_idx;
        
        Self {
            node1: node1,
            node2: node1 + 1,
            node3: node1 + grid_ncol + 1,
            node4: node1 + grid_ncol,
            grid_nrow: grid_nrow,
            grid_ncol: grid_ncol,
            grid_row_oversampling: grid_row_oversampling,
            grid_col_oversampling: grid_col_oversampling,
            window_src: window_src,
            window_rel: window_rel,
            gmi_col_idx: gmi_col_idx,
            gmi_col_idx_t: gmi_col_idx_t,
            gmi_row_idx: gmi_row_idx,
            gmi_row_idx_t: gmi_row_idx_t,
            gmi_w1: gmi_w1,
            gmi_w2: gmi_w2,
            gmi_w3: gmi_w3,
            gmi_w4: gmi_w4,
            }
    }
    
    /// Update current weights - aimed to be called at the start of each output position iteration.
    #[inline]
    pub fn update_weights(&mut self) {
        self.gmi_w1 = self.gmi_col_idx_t * self.gmi_row_idx_t;
        self.gmi_w2 = self.gmi_col_idx * self.gmi_row_idx_t;
        self.gmi_w3 = self.gmi_col_idx * self.gmi_row_idx;
        self.gmi_w4 = self.gmi_col_idx_t * self.gmi_row_idx;
    }
    
    /// Advances the mesh one column to the right within the current row of the grid.
    ///
    /// This updates the four corner node indices to match the new column position.
    /// If the current column is the last column in the grid, the mesh is collapsed into a
    /// vertical line (zero-width), effectively duplicating the right corners to match the left ones.
    ///
    /// # Arguments
    ///
    /// - `grid_col_idx`: The current column index in the iteration.
    #[inline]
    pub fn next_src_col(&mut self, grid_col_idx: usize) {
        self.node1 += 1;
        self.node2 += 1;
        self.node3 += 1;
        self.node4 += 1;
        
        // If on the last column, collapse the mesh to a vertical line
        if grid_col_idx == self.grid_ncol - 1 {
            self.node2 = self.node1;
            self.node3 = self.node4;
        }
    }
    
    /// Advances the mesh one row down within the grid.
    ///
    /// Updates the corner node indices to match the new row at the same column offset
    /// defined by `window_src.start_col`.
    /// If the current row is the last row in the grid, the mesh is collapsed vertically,
    /// so that the bottom row nodes are set equal to the top row ones.
    ///
    /// # Arguments
    ///
    /// - `grid_row_idx`: The current row index in the iteration.
    #[inline]
    pub fn next_src_row(&mut self, grid_row_idx: usize) {
        let node1 = grid_row_idx * self.grid_ncol + self.window_src.start_col;
        self.node1 = node1;
        self.node2 = node1 + 1;
        self.node4 = node1 + self.grid_ncol;
        self.node3 = self.node4 + 1;
        
        // If on the last row, collapse the mesh to a horizontal line
        if grid_row_idx == self.grid_nrow - 1 {
            self.node4 = self.node1;
            self.node3 = self.node2;
        }
    }
    
    /// Advances the current position within the interpolation grid.
    ///
    /// This is the core method invoked by the main interpolation loop to traverse the grid's mesh.
    /// It manages both the global output position and the relative position within the current
    /// grid cell's oversampled mesh.
    ///
    /// # Arguments
    ///
    /// * `grid_row_idx`: The current row index in the input grid iteration.
    /// * `grid_col_idx`: The current column index in the input grid iteration.
    /// * `out_col_idx`: The current column index within the *output* row being generated.
    /// * `windowed_out_idx`: The absolute index for output buffer assignment within the current window.
    /// * `window_out_idx_jump`: The index increment required when moving to the next output row.
    /// * `ncol_out`: The total number of columns in an output row.
    ///
    /// # Returns
    ///
    /// A tuple containing the updated indices:
    /// `(grid_row_idx, grid_col_idx, out_col_idx, windowed_out_idx)`
    ///
    /// # Behavior
    ///
    /// This method progresses the interpolation by:
    /// 1. Incrementing the global output column index (`out_col_idx`) and the windowed output index (`windowed_out_idx`).
    /// 2. Advancing the relative column index (`gmi_col_idx`) within the current mesh cell.
    /// 3. If all columns within the current mesh cell's oversampling are processed:
    ///    - Resets `gmi_col_idx` to 0.
    ///    - Increments the input grid's column index (`grid_col_idx`).
    ///    - Calls `self.next_src_col()` to update the source nodes for the next input column.
    /// 4. If the end of the current output row (`ncol_out`) is reached:
    ///    - Resets `out_col_idx` to 0.
    ///    - Updates `windowed_out_idx` to point to the start of the next row's window.
    ///    - Resets `grid_col_idx` to the starting column of the current processing window (`self.window_src.start_col`).
    ///    - Resets `gmi_col_idx` to the starting relative column of the mesh window (`self.window_rel.start_col`).
    ///    - Increments the relative row index (`gmi_row_idx`) within the current mesh cell.
    ///    - If all rows within the current mesh cell's oversampling are processed:
    ///        - Resets `gmi_row_idx` to 0.
    ///        - Increments the input grid's row index (`grid_row_idx`).
    ///    - Calls `self.next_src_row()` to update the source nodes for the next input row.
    ///    - Updates the complement of the relative mesh row index (`self.gmi_row_idx_t`).
    /// 5. Updates the complement of the relative mesh column index (`self.gmi_col_idx_t`).
    #[inline]
    pub fn next(&mut self,
            grid_row_idx: usize,
            grid_col_idx: usize,
            out_col_idx: usize,
            windowed_out_idx: usize,
            window_out_idx_jump: usize,
            ncol_out: usize,
    ) -> (usize, usize, usize, usize)
    {
        let mut grid_row_idx = grid_row_idx;
        let mut grid_col_idx = grid_col_idx;
        // Global output : go next output column
        let mut out_col_idx = out_col_idx + 1;
        let mut windowed_out_idx = windowed_out_idx + 1;
        
        // Advance to the next column within the current mesh interpolation cell
        self.gmi_col_idx += 1;
        
        // Check if all columns in the current mesh oversampling have been processed
        if self.gmi_col_idx == self.grid_col_oversampling {
            // Current row within mesh oversampling is done.
            // Warning : we cant go to next mesh on the same row if the current mesh is the last one
            // Go to next mesh on the same line
            // - reset relative mesh column index
            self.gmi_col_idx = 0;
            // - increment column index relative to input grid
            grid_col_idx += 1;
            // - shift all nodes to right - the method takes care of the last mesh border
            self.next_src_col(grid_col_idx);
        }
        
        // Test if current output row is done
        if out_col_idx == ncol_out {
            
            // Reset output column index to 0
            out_col_idx = 0;
            windowed_out_idx += window_out_idx_jump;
            
            // Reset column index relative to input grid
            grid_col_idx = self.window_src.start_col;
            
            // Reset relative mesh column index
            self.gmi_col_idx = self.window_rel.start_col;
            
            // Increment relative mesh row index
            self.gmi_row_idx += 1;
            
            // Test if row oversampling in current mesh is done.
            if self.gmi_row_idx == self.grid_row_oversampling {
                // Reset mesh relative row index
                self.gmi_row_idx = 0;
                // Increment row index relative to input grid
                grid_row_idx += 1;
            }
            
            // Update interpolation nodes - going next src row
            self.next_src_row(grid_row_idx);
            
            // Update mesh relative row index complement
            self.gmi_row_idx_t = self.grid_row_oversampling - self.gmi_row_idx;
        }
        // Update mesh relative col index complement
        self.gmi_col_idx_t = self.grid_col_oversampling - self.gmi_col_idx;
        
        (grid_row_idx, grid_col_idx, out_col_idx, windowed_out_idx)
    }
} 

/// Perform a bilinear grid interpolation with oversampling, computing interpolated
/// values for both grid rows and columns. The grid is iterated over in a row-major
/// order with oversampling applied to both rows and columns.
///
/// # Parameters
/// - `grid_row_array`: A structure holding the grid of row values to be interpolated.
/// - `grid_col_array`: A structure holding the grid of column values to be interpolated.
/// - `grid_row_oversampling`: The oversampling factor for the row dimension.
/// - `grid_col_oversampling`: The oversampling factor for the column dimension.
/// - `out_win`: An optional window to define the region where to save the interpolated values in `ima_out`.
/// - `ima_in_origin_row`: An optional `f64` bias value that adjusts the row coordinate obtained from `grid_row_array`.
///    Its primary use cases include aligning with alternative grid origin conventions or handling situations where the 
///    provided `ima_in` array corresponds to a subregion of the complete source raster.
/// - `ima_out_origin_row`: An optional `f64` bias value that adjusts the row coordinate obtained from `grid_col_array`.
///    Its primary use cases include aligning with alternative grid origin conventions or handling situations where the 
///    provided `ima_in` array corresponds to a subregion of the complete source raster.
/// - `check_boundaries`: If True the BoundsCheck strategy is adopted otherwise it is the NoBoundsCheck strategy. Use it
///    with caution !
///
/// # Output
/// The function computes interpolated values for both grid rows and columns at each
/// output position. The grid is processed in a row-major order, with each output
/// position corresponding to an interpolated point on the grid.
///
/// # Process
/// - The output grid size is determined based on the oversampling factors and the
///   dimensions of the input grid.
/// - Bilinear interpolation is performed by determining the weights (`gmi_w1`, `gmi_w2`, `gmi_w3`, `gmi_w4`)
///   for each grid mesh and applying them to the neighboring nodes.
/// - The interpolation is done both on the grid columns and rows separately.
/// - The mesh index is updated iteratively, advancing over the grid and applying
///   the appropriate interpolation for each position.
///
/// # Flow
/// 1. The size of the output grid is computed based on oversampling factors.
/// 2. The algorithm loops over all output positions, applying bilinear interpolation.
/// 3. It tracks the current mesh position and advances through the grid as necessary.
/// 4. For each output position, interpolation weights are calculated, and the interpolated
///    values are stored.
/// 5. The loop advances to the next output column and row, updating mesh and output positions.
/// 6. When the row or column within the mesh is completed, the indices are reset, and the
///    algorithm moves to the next mesh position.
///
/// # Bilinear interpolation on grid mesh
/// Each interpolated point within a mesh uses the four neighbors in order to perform a bilinear 
/// interpolation.
///
///        (0,0) ----- (0,1)
///         |             |
///         |             |
///        (1,0) ----- (1,1)
///
/// Each grid mesh has 4 nodes :
/// - (0,0) ('gmi_node_1' in code)
/// - (0,1) ('gmi_node_2' in code)
/// - (1,1) ('gmi_node_3' in code)
/// - (1,0) ('gmi_node_4' in code)
///

pub fn array1_grid_resampling<'a, T, V, W, I, C>(
        interp: &I,
        grid_validity_checker: &C,
        ima_in: &GxArrayView<'_, T>,
        grid_row_array: &GxArrayView<'_, W>,
        grid_col_array: &GxArrayView<'_, W>,
        grid_row_oversampling: usize,
        grid_col_oversampling: usize,
        ima_out: &mut GxArrayViewMut<'_, V>,
        nodata_val_out: V,
        ima_mask_in: Option<&GxArrayView<'_,u8>>,
        ima_mask_out: Option<&'a mut GxArrayViewMut<'a, u8>>, 
        grid_win: Option<&GxArrayWindow>,
        out_win: Option<&GxArrayWindow>,
        ima_in_origin_row: Option<f64>,
        ima_in_origin_col: Option<f64>,
        check_boundaries: bool,
        ) -> Result<(), GxError>
where
    T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
    //U: Copy + PartialEq + Into<f64>,
    V: Copy + PartialEq + From<f64>,
    W: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
    I: GxArrayViewInterpolator,
    C: GridMeshValidator<W>,
    //<U as Mul<f64, Output=f64>>::Output: Add,
{
    // Check that both grid_row_array and grid_col_array have same size
    if (grid_row_array.nrow != grid_col_array.nrow) || (grid_row_array.ncol != grid_col_array.ncol) {
        return Err(GxError::ShapesMismatch { field1:"grid_row_array", field2:"grid_col_array" })
    }
    
    // Check that if grid_row_oversampling is not 1 we get enough data to 
    // interpolate (ie. at least 2)
    if grid_row_oversampling > 1 && grid_row_array.nrow < 2 {
        return Err(GxError::InsufficientGridCoverage { field1:"rows" })
    }
    // Check that if grid_col_oversampling is not 1 we get enough data to 
    // interpolate (ie. at least 2)
    if grid_col_oversampling > 1 && grid_row_array.ncol < 2 {
        return Err(GxError::InsufficientGridCoverage { field1:"columns" })
    }
    
    // Manage the optional ima_in_origin_* ; if not provided set it to 0.
    let ima_in_origin_row: f64 = ima_in_origin_row.unwrap_or(0.);
    let ima_in_origin_col: f64 = ima_in_origin_col.unwrap_or(0.);
    
    // Manage the optional grid_win
    // The grid_win contains production limit to apply on the full resolution grid.
    // If not given we init a window corresponding to the full grid
    let full_res_grid_window = match grid_win {
        Some(some_grid_win) => {
            some_grid_win
        },
        None =>  &GxArrayWindow {
            start_row: 0,
            end_row: (grid_row_array.nrow - 1) * grid_row_oversampling,
            start_col: 0,
            end_col: (grid_row_array.ncol - 1) * grid_col_oversampling,
        }, 
    };
    // Compute number of rows and columns for output
    let ncol_out = full_res_grid_window.width(); //full_res_grid_window.end_col - full_res_grid_window.start_col + 1;
    let size_out = full_res_grid_window.size(); //nrow_out * ncol_out;
    
    // Compute the grid interval containing the window
    // If no window was given it is directly the full grid but we still make the code generic
    // since it is performed only once before the loop
    // The interpolation nodes should be taken from grid_window_src, but the grid_window_rel is used
    // to set the start and stop indexes for columns and rows.
    let (grid_window_src, grid_window_rel) = GxArrayWindow::get_wrapping_window_for_resolution(
            (grid_row_oversampling, grid_col_oversampling), full_res_grid_window)?;

    // Manage the output window
    let out_window = match out_win {
        Some(some_out_win) => {
            some_out_win
        },
        None => &GxArrayWindow {
            start_row: 0,
            end_row: full_res_grid_window.height() - 1,
            start_col: 0,
            end_col: ncol_out - 1,
        },
    };
    // Check that the dimension of `out_window` is sufficient with `ima_out`
    out_window.validate_with_array(ima_out)?;

    match (ima_mask_in, ima_mask_out, check_boundaries) {
        (Some(in_mask), Some(out_mask), true) => {
            let mut context = GxArrayViewInterpolationContext::new(
                BinaryInputMask { mask: in_mask},
                BinaryOutputMask { mask: out_mask },
                BoundsCheck {},
            );
            
            perform_grid_resampling_loop( interp,
                grid_validity_checker,
                ima_in,
                grid_row_array,
                grid_col_array,
                grid_row_oversampling,
                grid_col_oversampling,
                ima_out,
                nodata_val_out,
                ima_in_origin_row,
                ima_in_origin_col,
                &grid_window_src,
                &grid_window_rel,
                out_window,
                ncol_out,
                size_out,
                &mut context)
        },
        
        (Some(in_mask), None, true) => {
            let mut context = GxArrayViewInterpolationContext::new(
                BinaryInputMask { mask: in_mask},
                NoOutputMask {},
                BoundsCheck {},
            );
            
            perform_grid_resampling_loop( interp,
                grid_validity_checker,
                ima_in,
                grid_row_array,
                grid_col_array,
                grid_row_oversampling,
                grid_col_oversampling,
                ima_out,
                nodata_val_out,
                ima_in_origin_row,
                ima_in_origin_col,
                &grid_window_src,
                &grid_window_rel,
                out_window,
                ncol_out,
                size_out,
                &mut context)
        },
        
        (None, Some(out_mask), true) => {
            let mut context = GxArrayViewInterpolationContext::new(
                NoInputMask {},
                BinaryOutputMask { mask: out_mask },
                BoundsCheck {},
            );
            
            perform_grid_resampling_loop( interp,
                grid_validity_checker,
                ima_in,
                grid_row_array,
                grid_col_array,
                grid_row_oversampling,
                grid_col_oversampling,
                ima_out,
                nodata_val_out,
                ima_in_origin_row,
                ima_in_origin_col,
                &grid_window_src,
                &grid_window_rel,
                out_window,
                ncol_out,
                size_out,
                &mut context)
        },
        
        (None, None, true) => {
            let mut context = GxArrayViewInterpolationContext::default();
            
            perform_grid_resampling_loop( interp,
                    grid_validity_checker,
                    ima_in,
                    grid_row_array,
                    grid_col_array,
                    grid_row_oversampling,
                    grid_col_oversampling,
                    ima_out,
                    nodata_val_out,
                    ima_in_origin_row,
                    ima_in_origin_col,
                    &grid_window_src,
                    &grid_window_rel,
                    out_window,
                    ncol_out,
                    size_out,
                    &mut context)
        },
        
        (Some(in_mask), Some(out_mask), false) => {
            let mut context = GxArrayViewInterpolationContext::new(
                BinaryInputMask { mask: in_mask},
                BinaryOutputMask { mask: out_mask },
                NoBoundsCheck {},
            );
            
            perform_grid_resampling_loop( interp,
                grid_validity_checker,
                ima_in,
                grid_row_array,
                grid_col_array,
                grid_row_oversampling,
                grid_col_oversampling,
                ima_out,
                nodata_val_out,
                ima_in_origin_row,
                ima_in_origin_col,
                &grid_window_src,
                &grid_window_rel,
                out_window,
                ncol_out,
                size_out,
                &mut context)
        },
        
        (Some(in_mask), None, false) => {
            let mut context = GxArrayViewInterpolationContext::new(
                BinaryInputMask { mask: in_mask},
                NoOutputMask {},
                NoBoundsCheck {},
            );
            
            perform_grid_resampling_loop( interp,
                grid_validity_checker,
                ima_in,
                grid_row_array,
                grid_col_array,
                grid_row_oversampling,
                grid_col_oversampling,
                ima_out,
                nodata_val_out,
                ima_in_origin_row,
                ima_in_origin_col,
                &grid_window_src,
                &grid_window_rel,
                out_window,
                ncol_out,
                size_out,
                &mut context)
        },
        
        (None, Some(out_mask), false) => {
            let mut context = GxArrayViewInterpolationContext::new(
                NoInputMask {},
                BinaryOutputMask { mask: out_mask },
                NoBoundsCheck {},
            );
            
            perform_grid_resampling_loop( interp,
                grid_validity_checker,
                ima_in,
                grid_row_array,
                grid_col_array,
                grid_row_oversampling,
                grid_col_oversampling,
                ima_out,
                nodata_val_out,
                ima_in_origin_row,
                ima_in_origin_col,
                &grid_window_src,
                &grid_window_rel,
                out_window,
                ncol_out,
                size_out,
                &mut context)
        },
        
        (None, None, false) => {
            let mut context = GxArrayViewInterpolationContext::new(
                NoInputMask {},
                NoOutputMask {},
                NoBoundsCheck {},
            );
            
            perform_grid_resampling_loop( interp,
                    grid_validity_checker,
                    ima_in,
                    grid_row_array,
                    grid_col_array,
                    grid_row_oversampling,
                    grid_col_oversampling,
                    ima_out,
                    nodata_val_out,
                    ima_in_origin_row,
                    ima_in_origin_col,
                    &grid_window_src,
                    &grid_window_rel,
                    out_window,
                    ncol_out,
                    size_out,
                    &mut context)
        },
    }
}
    

fn perform_grid_resampling_loop<T, V, W, I, C, IC>(   
        interp: &I,
        grid_validity_checker: &C,
        ima_in: &GxArrayView<'_, T>,
        grid_row_array: &GxArrayView<'_, W>,
        grid_col_array: &GxArrayView<'_, W>,
        grid_row_oversampling: usize,
        grid_col_oversampling: usize,
        ima_out: &mut GxArrayViewMut<'_, V>,
        nodata_val_out: V,
        ima_in_origin_row: f64,
        ima_in_origin_col: f64,
    grid_window_src: &GxArrayWindow,
    grid_window_rel: &GxArrayWindow,
    out_window: &GxArrayWindow,
    ncol_out: usize,
    size_out: usize,
    context: &mut IC,
        ) -> Result<(), GxError>
where
    T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
    V: Copy + PartialEq + From<f64>,
    W: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
    I: GxArrayViewInterpolator,
    C: GridMeshValidator<W>,
    IC: GxArrayViewInterpolationContextTrait,
{

    // Current position in grid
    let mut grid_row_idx: usize = grid_window_src.start_row;
    let mut grid_col_idx: usize = grid_window_src.start_col;
    
    // Current output position
    let mut out_col_idx: usize = 0;
    
    // The 'gmi' prefix stands for grid mesh interpolation
    // It is used for all variable relative to the grid mesh bilinear interpolation
    // It defines the relative position in current grid mesh :
    // - gmi_{row|col}_idx can be set in [0, grid_{row|col}_oversampling[
    // - the init values (first iteration) are taken from the grid_window_rel
    
    /*
    // The init idx is given by the relative position in the source window.
    let mut gmi_row_idx: usize = grid_window_rel.start_row;
    let mut gmi_col_idx: usize = grid_window_rel.start_col;
    
    // Complement of the relative position in current grid mesh 
    let mut gmi_col_idx_t: usize = grid_col_oversampling - gmi_col_idx;
    let mut gmi_row_idx_t: usize = grid_row_oversampling - gmi_row_idx;
    */
    
    // Init the mesh used for grid values bilinear interpolation.
    let mut gmi_mesh = GridMesh::new(grid_row_array.nrow, grid_row_array.ncol, grid_row_oversampling,
            grid_col_oversampling, &grid_window_src, &grid_window_rel);
    
    // Mesh interpolation norm factor
    let gmi_norm_factor: f64 = (grid_col_oversampling * grid_row_oversampling) as f64;
    
    // Call the GxArrayViewInterpolator allocate_kernel_buffer to allocate
    // the buffer to store the kernel weights.
    // That buffer will be passed to the array1_interp2 method.
    let mut weights_buffer = interp.allocate_kernel_buffer();
    
    // Init the target idx in `ima_out`
    let mut windowed_out_idx: usize = out_window.start_row * ima_out.ncol + out_window.start_col;
    // Compute the shift to apply to go to next row first col
    // Please note we substract 1 here just because the jump is applied in the loop after a +1 addition
    let window_out_idx_jump: usize = ima_out.ncol - out_window.end_col + out_window.start_col - 1;
    let ima_out_var_size: usize = ima_out.nrow * ima_out.ncol;
        
    for mut out_idx in 0..size_out {
        
        gmi_mesh.update_weights();
        
        // Here we call the validity checker for each oversampled index.
        // We may improve this loop by jumping to the next mesh directly
        if grid_validity_checker.validate(&mut gmi_mesh, &mut out_idx, &grid_row_array) {
        
            // Bilinear grid interpolation with oversampling
            let gmi_w1: f64 = gmi_mesh.gmi_w1 as f64;
            let gmi_w2: f64 = gmi_mesh.gmi_w2 as f64;
            let gmi_w3: f64 = gmi_mesh.gmi_w3 as f64;
            let gmi_w4: f64 = gmi_mesh.gmi_w4 as f64;
            
            // Perform the interpolation on column + apply origin bias
            let grid_col_val: f64 = (
                    grid_col_array.data[gmi_mesh.node1] * gmi_w1 +
                    grid_col_array.data[gmi_mesh.node2] * gmi_w2 +
                    grid_col_array.data[gmi_mesh.node3] * gmi_w3 +
                    grid_col_array.data[gmi_mesh.node4] * gmi_w4
                    ) / gmi_norm_factor + ima_in_origin_col;
            
            // Perform the interpolation on rows + apply origin bias
            let grid_row_val: f64 = (
                    grid_row_array.data[gmi_mesh.node1] * gmi_w1 +
                    grid_row_array.data[gmi_mesh.node2] * gmi_w2 +
                    grid_row_array.data[gmi_mesh.node3] * gmi_w3 +
                    grid_row_array.data[gmi_mesh.node4] * gmi_w4
                    ) / gmi_norm_factor + ima_in_origin_row;
                        
            // Do grid interpolation here
            let _ = interp.array1_interp2(
                    &mut weights_buffer,
                    grid_row_val,
                    grid_col_val,
                    windowed_out_idx,
                    ima_in,
                    ima_out,
                    nodata_val_out,
                    context,
                    );
        } else {
            // do something
            for ivar in 0..ima_in.nvar {
                // Write nodata value to output buffer
                //ima_out.data[out_idx + ivar * size_out] = nodata_val_out;
                ima_out.data[windowed_out_idx + ivar * ima_out_var_size] = nodata_val_out;
            }
            context.output_mask().set_value(windowed_out_idx, 0);
        }
        // Prepare next iteration
        (grid_row_idx, grid_col_idx, out_col_idx, windowed_out_idx) = gmi_mesh.next(grid_row_idx,
                grid_col_idx, out_col_idx, windowed_out_idx, window_out_idx_jump, ncol_out);

    }
    Ok(())
}


#[cfg(test)]
mod gx_grid_resampling_test {
    use super::*;
    use crate::core::interp::gx_optimized_bicubic_kernel::{GxOptimizedBicubicInterpolator};
    use crate::core::gx_array::{gx_array_data_approx_eq_window};
    
    /// Checks if two slices of f64 values are approximately equal within a given tolerance.
    ///
    /// # Arguments
    /// * `a` - First slice of f64 values.
    /// * `b` - Second slice of f64 values.
    /// * `tol` - The allowed tolerance for differences.
    ///
    /// # Returns
    /// * `true` if all corresponding elements of `a` and `b` differ by at most `tol`, otherwise `false`.
    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (*x - *y).abs() <= tol)
    }
    
    /// This test makes sure that an identity transformation is correct
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let nrow_in = 15;
        let ncol_in = 10;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let margin = 0;
        
        // we will not be able to interpolate at edge.
        let nrow_out = nrow_in-2*margin;
        let ncol_out = ncol_in-2*margin;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        let mut data_expected = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_out * ncol_out];
        let mut grid_col = vec![0.0; nrow_out * ncol_out];
        
        // Init data_in values
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                data_in[irow * ncol_in + icol] = 10.* (irow as f64) * (ncol_in as f64) + 2.5* (icol as f64);
            }
        }
        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                grid_row[irow * ncol_out + icol] = irow as f64 + margin as f64;
                grid_col[irow * ncol_out + icol] = icol as f64 + margin as f64;
                data_expected[irow * ncol_out + icol] = data_in[(irow + margin) * ncol_in + (icol + margin)];
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_out, ncol_out);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_out, ncol_out);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = NoCheckGridMeshValidator{};
        
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                1, //grid_row_oversampling: usize,
                1, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                None, //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
        
        assert!(approx_eq(&data_out, &data_expected, 1e-10));
    }
    
    
    /// This test makes sure that an identity transformation with a invalid value
    // is correct
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity_w_grid_nodata() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let nrow_in = 6;
        let ncol_in = 5;
        // Value to use as nodata
        let grid_nodata = -99999.0;
        // Row index used to fill the row grid with grid_nodata
        let grid_nodata_row = 2;
        // Col index used to fill the col grid with grid_nodata
        let grid_nodata_col = 1;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let margin = 0;
        
        // we will not be able to interpolate at edge.
        let nrow_out = nrow_in-2*margin;
        let ncol_out = ncol_in-2*margin;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let nodata_val_out = 0.;
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        let mut data_expected = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_out * ncol_out];
        let mut grid_col = vec![0.0; nrow_out * ncol_out];
        
        // Init data_in values
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                data_in[irow * ncol_in + icol] = 10.* (irow as f64) * (ncol_in as f64) + 2.5* (icol as f64);
            }
        }
        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                
                if irow == grid_nodata_row {
                    grid_row[irow * ncol_out + icol] = grid_nodata;
                }
                else {
                    grid_row[irow * ncol_out + icol] = irow as f64 + margin as f64;
                }
                if icol == grid_nodata_col {
                    grid_col[irow * ncol_out + icol] = grid_nodata;
                }
                else {
                    grid_col[irow * ncol_out + icol] = icol as f64 + margin as f64;
                }
                
                if irow == grid_nodata_row || icol == grid_nodata_col {
                    data_expected[irow * ncol_out + icol] = nodata_val_out;
                } else {
                    data_expected[irow * ncol_out + icol] = data_in[(irow + margin) * ncol_in + (icol + margin)];
                }
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_out, ncol_out);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_out, ncol_out);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = InvalidValueGridMeshValidator{invalid_value: grid_nodata, epsilon: 1e-10};
        
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, InvalidValueGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                1, //grid_row_oversampling: usize,
                1, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                nodata_val_out, //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                None, //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
                
        /*      UNCOMMENT FOR MANUAL DEBUGGING
        println!("\ngrid_row\n");
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                let c = irow * ncol_out + icol;
                print!("{} ", grid_row[c]);
                //println!( "(row {}, col {}) - mask = {} - in = {} - out = {}", irow, icol , grid_mask[c], data_in[c], data_out[c]);
            }
            println!("");
        }
        println!("\ngrid_col\n");
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                let c = irow * ncol_out + icol;
                print!("{} ", grid_col[c]);
                //println!( "(row {}, col {}) - mask = {} - in = {} - out = {}", irow, icol , grid_mask[c], data_in[c], data_out[c]);
            }
            println!("");
        }
        println!("\ndata_in\n");
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                let c = irow * ncol_out + icol;
                print!("{} ", data_in[c]);
                //println!( "(row {}, col {}) - mask = {} - in = {} - out = {}", irow, icol , grid_mask[c], data_in[c], data_out[c]);
            }
            println!("");
        }
        println!("\ndata_out\n");
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                let c = irow * ncol_out + icol;
                print!("{} ", data_out[c]);
                //println!( "(row {}, col {}) - mask = {} - in = {} - out = {}", irow, icol , grid_mask[c], data_in[c], data_out[c]);
            }
            println!("");
        } */
        assert!(approx_eq(&data_out, &data_expected, 1e-10));
    }
    
    /// This test makes sure that an identity transformation with a grid mask
    // is correct
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity_w_grid_mask() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let nrow_in = 6;
        let ncol_in = 5;
        // Row index used to fill the row grid with grid_nodata
        let grid_nodata_row = 2;
        // Col index used to fill the col grid with grid_nodata
        let grid_nodata_col = 1;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let margin = 0;
        
        // we will not be able to interpolate at edge.
        let nrow_out = nrow_in-2*margin;
        let ncol_out = ncol_in-2*margin;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let nodata_val_out = 0.;
        // data out expected to be same size (oversampling 1)
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        let mut data_expected = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_out * ncol_out];
        let mut grid_col = vec![0.0; nrow_out * ncol_out];
        // init mask
        let grid_mask_valid_value = 1;
        let mut grid_mask = vec![grid_mask_valid_value; nrow_out * ncol_out];
        
        // Init data_in values
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                data_in[irow * ncol_in + icol] = 10.* (irow as f64) * (ncol_in as f64) + 2.5* (icol as f64);
            }
        }
        // Init grid, mask and data_out
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                
                grid_row[irow * ncol_out + icol] = irow as f64 + margin as f64;
                grid_col[irow * ncol_out + icol] = icol as f64 + margin as f64;
                
                if irow == grid_nodata_row || icol == grid_nodata_col {
                    grid_mask[irow * ncol_out + icol] = 0;
                    data_expected[irow * ncol_out + icol] = nodata_val_out;
                } else {
                    data_expected[irow * ncol_out + icol] = data_in[(irow + margin) * ncol_in + (icol + margin)];
                }
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_out, ncol_out);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_out, ncol_out);
        let mask_view = GxArrayView::new(&grid_mask, 1, nrow_in, ncol_in);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = MaskGridMeshValidator{ mask_view: &mask_view, valid_value: grid_mask_valid_value };
        
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, MaskGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                1, //grid_row_oversampling: usize,
                1, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                nodata_val_out, //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                None, //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
                
        /*      UNCOMMENT FOR MANUAL DEBUGGING
        println!("mask\n");
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                let c = irow * ncol_out + icol;
                print!("{} ", grid_mask[c]);
                //println!( "(row {}, col {}) - mask = {} - in = {} - out = {}", irow, icol , grid_mask[c], data_in[c], data_out[c]);
            }
            println!("");
        }
        println!("\ndata_in\n");
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                let c = irow * ncol_out + icol;
                print!("{} ", data_in[c]);
                //println!( "(row {}, col {}) - mask = {} - in = {} - out = {}", irow, icol , grid_mask[c], data_in[c], data_out[c]);
            }
            println!("");
        }
        println!("\ndata_out\n");
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                let c = irow * ncol_out + icol;
                print!("{} ", data_out[c]);
                //println!( "(row {}, col {}) - mask = {} - in = {} - out = {}", irow, icol , grid_mask[c], data_in[c], data_out[c]);
            }
            println!("");
        } */
        
        assert!(approx_eq(&data_out, &data_expected, 1e-10));
    }
    
    /// TODO
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity_zoom_win() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let nrow_in = 15;
        let ncol_in = 10;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let resolution = (2, 5);
        
        // we will not be able to interpolate at edge.
        let nrow_grid = nrow_in;
        let ncol_grid = ncol_in;
        let nrow_out = (nrow_in-1)*resolution.0 + 1;
        let ncol_out = (ncol_in-1)*resolution.1 + 1;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        //let mut data_expected = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_grid * ncol_grid];
        let mut grid_col = vec![0.0; nrow_grid * ncol_grid];
        
        // Init data_in values
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                data_in[irow * ncol_in + icol] = 10.* (irow as f64) * (ncol_in as f64) + 2.5* (icol as f64);
            }
        }
        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_grid {
            for icol in 0..ncol_grid {
                grid_row[irow * ncol_grid + icol] = irow as f64;
                grid_col[irow * ncol_grid + icol] = icol as f64;
                //data_expected[irow * ncol_out + icol] = data_in[(irow + margin) * ncol_in + (icol + margin)];
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_grid, ncol_grid);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_grid, ncol_grid);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = NoCheckGridMeshValidator{};
        //let win = GxArrayWindow(
        
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                resolution.0, //grid_row_oversampling: usize,
                resolution.1, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                None, //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
        
        //assert!(approx_eq(&data_out, &data_expected, 1e-10));
    }
    
    /// TODO
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_translate() {
        let interp = GxOptimizedBicubicInterpolator::new();
        
        let dx = 10.5;
        let dy = -20.5;
        let nrow_in = 15;
        let ncol_in = 10;
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let margin = 0;
        
        // we will not be able to interpolate at edge.
        let nrow_out = nrow_in-2*margin;
        let ncol_out = ncol_in-2*margin;
        //let nrow_out = 2;
        //let ncol_out = 2;
        let mut data_out = vec![0.0; nrow_out * ncol_out];
        let mut data_expected = vec![0.0; nrow_out * ncol_out];
        let mut grid_row = vec![0.0; nrow_out * ncol_out];
        let mut grid_col = vec![0.0; nrow_out * ncol_out];
        
        // Init data_in values
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                data_in[irow * ncol_in + icol] = 10.* (irow as f64) * (ncol_in as f64) + 2.5* (icol as f64);
            }
        }
        // Init grid (identity from 2 - margin)
        for irow in 0..nrow_out {
            for icol in 0..ncol_out {
                grid_row[irow * ncol_out + icol] = irow as f64 + margin as f64 + dy;
                grid_col[irow * ncol_out + icol] = icol as f64 + margin as f64 + dx;
                data_expected[irow * ncol_out + icol] = data_in[(irow + margin) * ncol_in + (icol + margin)];
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_out, ncol_out);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_out, ncol_out);
        let mut array_out = GxArrayViewMut::new(&mut data_out, 1, nrow_out, ncol_out);
        let grid_checker = NoCheckGridMeshValidator{};
        
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                1, //grid_row_oversampling: usize,
                1, //grid_col_oversampling: usize,
                &mut array_out, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>,
                None, //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
    }
    
    /// This test aims to check the respect of the windowing when oversampling
    /// is involved.
    /// 
    /// Principles :
    /// - A first resampling on the full image domain is performed
    /// - Another resampling limited to the target window is performed
    /// - We check that the 2nd resampling matches with the window extracted
    ///   from the full image.
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity_oversampling_window() {
        let interp = GxOptimizedBicubicInterpolator::new();
        let tol = 1e-6;
        
        let oversampling_row = 6;
        let oversampling_col = 7;
        let nrow_in = 20;
        let ncol_in = 30;
        let nrow_grid = nrow_in;
        let ncol_grid = ncol_in;
        
        // Define full output
        let nrow_out_full = (nrow_in - 1) * oversampling_row + 1;
        let ncol_out_full = (ncol_in - 1) * oversampling_col + 1;
        
        // Define window related 
        let window = GxArrayWindow{start_row: 3, end_row: 87, start_col: 18, end_col: 183};
        
        // Create the data in and out buffers
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        let mut data_out_full_res = vec![0.0; nrow_out_full * ncol_out_full];
        let mut data_out_win = vec![0.0; window.size()];
        let mut grid_row = vec![0.0; nrow_grid * ncol_grid];
        let mut grid_col = vec![0.0; nrow_grid * ncol_grid];
        
        // Init data_in values with a bicubic function -> we should be able
        // to find similar values by interpolation of a decimated array
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                let xf = icol as f64;
                let yf = irow as f64;
                data_in[irow * ncol_in + icol] = 1.0 + 2.0 * xf + 3.0 * yf + 4.0 * xf * yf
                        + 5.0 * xf.powi(2) + 6.0 * yf.powi(2)
                        + 7.0 * xf.powi(3) + 8.0 * yf.powi(3);
            }
        }
                
        // Init grid to apply a simple transformation
        for irow in 0..nrow_grid {
            for icol in 0..ncol_grid {
                grid_row[irow * ncol_grid + icol] = irow as f64 + 3.5;
                grid_col[irow * ncol_grid + icol] = icol as f64 * 0.25;
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_grid, ncol_grid);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_grid, ncol_grid);
        let mut array_out_full = GxArrayViewMut::new(&mut data_out_full_res, 1, nrow_out_full, ncol_out_full);
        let mut array_out_win = GxArrayViewMut::new(&mut data_out_win, 1, window.height(), window.width());
        let grid_checker = NoCheckGridMeshValidator{};
        
        // Run resampling on full grid
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                oversampling_row, //grid_row_oversampling: usize,
                oversampling_col, //grid_col_oversampling: usize,
                &mut array_out_full, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                None, //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
        
        // Run resampling on window
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                oversampling_row, //grid_row_oversampling: usize,
                oversampling_col, //grid_col_oversampling: usize,
                &mut array_out_win, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                Some(&window), //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
        
        // Compare the results
        let win_array_out_win = GxArrayWindow { start_row: 0, end_row:window.height()-1,
                start_col: 0, end_col: window.width()-1,};
        assert!(gx_array_data_approx_eq_window( &array_out_win, &win_array_out_win, &array_out_full,
                &window, tol));
    }
    
    /// This test aims to check the respect of the output windowing 
    #[test]
    fn test_array1_grid_resampling_optimized_bicubic_identity_output_window() {
        let interp = GxOptimizedBicubicInterpolator::new();
        let tol = 1e-6;
        
        let oversampling_row = 6;
        let oversampling_col = 7;
        let nrow_in = 20;
        let ncol_in = 30;
        let nrow_grid = nrow_in;
        let ncol_grid = ncol_in;
        
        // Define full output
        let nrow_out_full = (nrow_in - 1) * oversampling_row + 1;
        let ncol_out_full = (ncol_in - 1) * oversampling_col + 1;
        
        // Define window related 
        let window = GxArrayWindow{start_row: 3, end_row: 87, start_col: 18, end_col: 183};
        
        // define a larger output
        let nrow_out_full_larger = nrow_out_full + 100;
        let ncol_out_full_larger = ncol_out_full + 20;
        let out_shift_row = 15 ;
        let out_shift_col = 10;
        let out_window = GxArrayWindow{start_row: out_shift_row, end_row: out_shift_row + window.height()-1, start_col: out_shift_col, end_col: out_shift_col + window.width()-1};
        
        
        // Create the data in and out buffers
        let mut data_in = vec![0.0; nrow_in * ncol_in ];
        //let mut data_out_full_res = vec![0.0; nrow_out_full * ncol_out_full];
        let mut data_out_full_res_larger = vec![0.0; nrow_out_full_larger * ncol_out_full_larger];
        let mut data_out_win = vec![0.0; window.size()];
        let mut grid_row = vec![0.0; nrow_grid * ncol_grid];
        let mut grid_col = vec![0.0; nrow_grid * ncol_grid];
        
        // Init data_in values with a bicubic function -> we should be able
        // to find similar values by interpolation of a decimated array
        for irow in 0..nrow_in {
            for icol in 0..ncol_in {
                let xf = icol as f64;
                let yf = irow as f64;
                data_in[irow * ncol_in + icol] = 1.0 + 2.0 * xf + 3.0 * yf + 4.0 * xf * yf
                        + 5.0 * xf.powi(2) + 6.0 * yf.powi(2)
                        + 7.0 * xf.powi(3) + 8.0 * yf.powi(3);
            }
        }
                
        // Init grid to apply a simple transformation
        for irow in 0..nrow_grid {
            for icol in 0..ncol_grid {
                grid_row[irow * ncol_grid + icol] = irow as f64 + 3.5;
                grid_col[irow * ncol_grid + icol] = icol as f64 * 0.25;
            }
        }
        
        // Init input structures
        let array_in = GxArrayView::new(&data_in, 1, nrow_in, ncol_in);
        let array_grid_row_in = GxArrayView::new(&grid_row, 1, nrow_grid, ncol_grid);
        let array_grid_col_in = GxArrayView::new(&grid_col, 1, nrow_grid, ncol_grid);
        let mut array_out_full_larger = GxArrayViewMut::new(&mut data_out_full_res_larger, 1, nrow_out_full_larger, ncol_out_full_larger);
        let mut array_out_win = GxArrayViewMut::new(&mut data_out_win, 1, window.height(), window.width());
        let grid_checker = NoCheckGridMeshValidator{};
        
        // Run resampling on full grid
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                oversampling_row, //grid_row_oversampling: usize,
                oversampling_col, //grid_col_oversampling: usize,
                &mut array_out_full_larger, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                Some(&window), //grid_win: Option<&GxArrayWindow>,
                Some(&out_window), //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
        
        // Run resampling on window 
        let _ = array1_grid_resampling::<f64, f64, f64, GxOptimizedBicubicInterpolator, NoCheckGridMeshValidator>(&interp,
                &grid_checker, //grid_validity_checker
                &array_in,
                &array_grid_row_in, //grid_row_array: &GxArrayView<'_, U>,
                &array_grid_col_in, //grid_col_array: &GxArrayView<'_, U>,
                oversampling_row, //grid_row_oversampling: usize,
                oversampling_col, //grid_col_oversampling: usize,
                &mut array_out_win, //ima_out: &mut GxArrayViewMut<'_, V>,
                0., //nodata_val_out: V,
                None, //ima_mask_in: Option<&GxArrayView<'_, U>>,
                None, //ima_mask_out: &mut Option<&mut GxArrayViewMut<'_, i8>>, 
                Some(&window), //grid_win: Option<&GxArrayWindow>,
                None, //out_win: Option<&GxArrayWindow>,
                None, //ima_in_origin_row: Option<f64>,
                None, //ima_in_origin_col: Option<f64>,
                true, // check_boundaries
                );
        
        // Compare the results
        let win_array_out_win = GxArrayWindow { start_row: 0, end_row:window.height()-1,
                start_col: 0, end_col: window.width()-1,};
        assert!(gx_array_data_approx_eq_window( &array_out_win, &win_array_out_win, &array_out_full_larger,
                &out_window, tol));
    }
}