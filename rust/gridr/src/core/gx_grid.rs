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
//! # Grid generals
//!
//! This crate provides core functionality for validating grid nodes with flexible strategies.
//!
//! ## Trait: GridNodeValidator<W>
//!
//! ```rust
//! pub trait GridNodeValidator<W> {
//!     fn validate(&self, node_idx: usize, grid_view: &GxArrayView<W>) -> bool;
//! }
//! ```
//!
//! Standardizes node validation with:
//! - Generic data type `W`
//! - Single-node validation method
//! - Boolean return indicating validity
//!
//! ## Implementations
//!
//! ### 1. NoCheckGridNodeValidator
//! - Always returns `true`
//! - No validation performed
//! - Most efficient option
//!
//! ### 2. InvalidValueGridNodeValidator
//! - Validates against sentinel values
//! - Uses epsilon comparison for floats
//! - Requires `W: Into<f64> + Copy`
//!
//! ### 3. MaskGridNodeValidator
//! - Validates using binary mask
//! - Requires `W: Copy`
//! - Checks mask value against `valid_value`
//!
//! ## Usage
//!
//! ```rust
//! // Basic usage
//! let validator = InvalidValueGridNodeValidator { invalid_value: -9999.0, epsilon: 1e-6 };
//!
//! // Validation check
//! if validator.validate(node_idx, grid_view) {
//!     // Process valid node
//! }
//! ```

use crate::core::gx_array::{GxArrayView};

/// A trait that standardizes grid node validation logic for individual positions in a grid.
///
/// Implementors of this trait define a validation method used to determine whether
/// a single grid node (identified by its index) is valid for further computation.
/// This enables composable and efficient validation strategies, such as value-based filtering
/// or mask exclusion, independent of any mesh or multi-node configuration.
///
/// # Type Parameters
///
/// * `W` - The type of data stored in the grid array (e.g., `f64`, `u8`, etc.).
pub trait GridNodeValidator<W> {
    /// Validates whether the specified node is suitable for computation.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the grid node to validate.
    /// * `grid_view` - A view into the grid data being validated.
    ///
    /// # Returns
    ///
    /// Returns `true` if the node is valid and can be processed, `false` otherwise.
    fn validate<'a>(&self, node_idx: usize, grid_view: &GxArrayView<'a, W>) -> bool;
}

/// A validator implementation that unconditionally accepts all nodes.
///
/// This is the simplest implementation of `GridNodeValidator`, which always returns `true`,
/// effectively disabling any validation logic.
///
/// Useful as a default or placeholder when no filtering is required.
#[derive(Debug)]
pub struct NoCheckGridNodeValidator;

impl<W> GridNodeValidator<W> for NoCheckGridNodeValidator {
    /// Validates whether the specified node is suitable for computation.
    /// This implementation for NoCheckGridNodeValidator always returns true.
    ///
    /// # Arguments
    ///
    /// * `_node_idx` - The index of the grid node to validate.
    /// * `_grid_view` - A view into the grid data being validated.
    ///
    /// # Returns
    ///
    /// Returns `true`.
    #[inline]
    fn validate<'a>(&self, _node_idx: usize, _grid_view: &GxArrayView<'a, W>) -> bool {
        true
    }
}

/// A validator that excludes nodes based on a specific invalid value.
///
/// This implementation considers a node invalid if its value is within a small threshold (`epsilon`)
/// of a predefined `invalid_value`. This is typically used to ignore missing or masked data
/// encoded as sentinel values (e.g., -9999.0).
///
/// # Fields
///
/// * `invalid_value` - The sentinel value that indicates invalid data.
/// * `epsilon` - The tolerance threshold for comparing floating-point values.
///
/// # Examples
///
/// ```
/// let validator = InvalidValueGridNodeValidator {
///     invalid_value: -9999.0,
///     epsilon: 1e-6,
/// };
/// `
#[derive(Debug)]
pub struct InvalidValueGridNodeValidator {
    /// The sentinel value that indicates invalid data.
    pub invalid_value: f64,
    /// The tolerance threshold for comparing floating-point values.
    pub epsilon: f64,
}

impl InvalidValueGridNodeValidator {
    /// Checks if a node value is considered invalid based on the epsilon threshold.
    ///
    /// # Arguments
    ///
    /// * `node` - The index of the node to check.
    /// * `grid_view` - A view into the grid data being validated.
    ///
    /// # Returns
    ///
    /// Returns `true` if the node value is within epsilon of the invalid value, `false` otherwise.
    #[inline]
    fn is_invalid<W>(&self, node: usize, grid_view: &GxArrayView<W>) -> bool
    where
        W: Into<f64> + Copy,
    {
        (grid_view.data[node].into() - self.invalid_value).abs() < self.epsilon
    }
}

impl<W> GridNodeValidator<W> for InvalidValueGridNodeValidator
where
    W: Into<f64> + Copy,
{
    /// Validates whether the specified node is suitable for computation.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the grid node to validate.
    /// * `grid_view` - A view into the grid data being validated.
    ///
    /// # Returns
    ///
    /// Returns `true` if the node is valid and can be processed, `false` otherwise.
    #[inline]
    fn validate<'a>(&self, node_idx: usize, grid_view: &GxArrayView<'a, W>) -> bool {
        !self.is_invalid(node_idx, grid_view)
    }
}

/// A validator that uses a binary mask array to determine node validity.
///
/// This implementation checks a mask array and considers a node invalid if its corresponding
/// mask value differs from a predefined `valid_value`. This is commonly used for excluding
/// regions using precomputed masks (e.g., land/sea masks).
#[derive(Debug)]
pub struct MaskGridNodeValidator<'a> {
    /// Immutable view to the associated mask
    pub mask_view: &'a GxArrayView<'a, u8>,
    /// Value to consider valid in the mask
    pub valid_value: u8,
}

impl<'a, W> GridNodeValidator<W> for MaskGridNodeValidator<'a>
where
    W: Copy,
{
    /// Validates whether the specified node is suitable for computation based on the mask.
    ///
    /// # Arguments
    ///
    /// * `node_idx` - The index of the grid node to validate.
    /// * `grid_view` - A view into the grid data being validated.
    ///
    /// # Returns
    ///
    /// Returns `true` if the mask contains `valid_value` at index `node_idx`, `false` otherwise.
    #[inline]
    fn validate(&self, node_idx: usize, _grid_view: &GxArrayView<W>) -> bool {
        self.mask_view.data[node_idx] == self.valid_value
    }
}
