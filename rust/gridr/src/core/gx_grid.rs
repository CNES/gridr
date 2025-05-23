#![warn(missing_docs)]
//! # Grid generals 


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
#[derive(Debug)]
pub struct InvalidValueGridNodeValidator {
    pub invalid_value: f64,
    pub epsilon: f64,
}

impl InvalidValueGridNodeValidator {
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
    pub mask_view: &'a GxArrayView<'a, u8>,
    pub valid_value: u8,
}

impl<'a, W> GridNodeValidator<W> for MaskGridNodeValidator<'a>
where
    W: Copy,
{
    #[inline]
    fn validate(&self, node_idx: usize, _grid_view: &GxArrayView<W>) -> bool {
        self.mask_view.data[node_idx] == self.valid_value
    }
}
