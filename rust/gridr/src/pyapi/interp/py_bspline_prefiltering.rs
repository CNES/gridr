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
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyArrayMethods, ToPyArray};

use crate::core::gx_array::{
    GxArrayView,
    GxArrayViewMut
};
use crate::core::interp::gx_bspline_prefiltering::{
    compute_2d_truncation_index,
    compute_2d_domain_extension,
    array1_bspline_prefiltering_ext_gene,
    array1_bspline_prefiltering_ext_gene_mask_safe_win,
};
use crate::pyapi::py_array::PyArrayWindow2;

/// Truncation index computation for max order n
/// The truncation index array is 1-based indexing.
///
/// # Arguments
///
/// * `n` - The spline order
/// * `epsilon` - The precision parameter for the truncation index calculation
///
/// # Returns
///
/// An array of $$N(i, \epsilon)$$ for 1 ≤ i ≤ \tilde{n}.
/// The first element of the returned array corresponds to $$N(1, \epsilon)$$.
/// The array is of fixed size `TRUNCATION_INDEX_BUFFER_MAX_SIZE` and contains zeros on non computed indexes.
#[pyfunction]
#[pyo3(signature = (n, epsilon))]
#[allow(clippy::too_many_arguments)]
pub fn py_compute_2d_truncation_index(
    py: Python<'_>,
    n: usize,
    epsilon: f64,
) -> Py<PyArray1<usize>>
{
    let trunc_idx = compute_2d_truncation_index(n, epsilon);
    trunc_idx.as_slice().to_pyarray(py).into()
}

/// Computes the extended domain lengths for B-spline pre-filtering using Approach 1 (Extended Domain) Eq. (54).
///
/// The function wraps the `core::interp::compute_2d_domain_extension` implementation
///
/// This function implements the first approach for computing pre-filtering coefficients with precision,
/// as described in the paper. It calculates the extended domain lengths $$L_j^{(n, \epsilon)}$$ for the B-spline, 
/// pre-filtering process.
///
/// The addition of the number of poles and the total sum of the domain lengths extension gives
/// the margin required for the full bspline interpolation process in order to achieve a precision
/// of `precision`.
///
/// The extended domain length is computed recursively using the formula:
///
// $$ \begin{cases}
// L_{\tilde{n}}^{(n, \epsilon)} = \tilde{n}\\
// L_{j}^{(n, \epsilon)} = L_{j+1}^{(n, \epsilon)} + N^{(j+1, \epsilon)}, j = \tilde{n}-1 to 0.
// \end{cases} $$
///
/// Where:
/// - $$n$$ is the spline order
/// - $$\epsilon$$ is the precision parameter
/// - $$\tilde{n} = \lfloor \frac{n}{2} \rfloor$$
/// - $$N^{(i, \epsilon)}$$ is the truncation index computed by `compute_2d_truncation_index`
///
/// # Arguments
///
/// * `n` - The spline order
/// * `epsilon` - The precision parameter for the truncation index calculation
///
/// # Returns
///
/// An array of extended domain lengths $$L_{j}^{(n, \epsilon)}$$ for $$j = 0 to \tilde{n}$$, where $$\tilde{n} = \lfloor \frac{n}{2} \rfloor$$.
/// The array is of fixed size `TRUNCATION_L_BUFFER_MAX_SIZE`.
///
/// # Panics
///
/// This function will panic if:
/// - `n` is 0 (division by zero)
/// - `n/2` exceeds `TRUNCATION_L_BUFFER_MAX_SIZE`
#[pyfunction]
#[pyo3(signature = (n, epsilon))]
#[allow(clippy::too_many_arguments)]
pub fn py_compute_2d_domain_extension(
    py: Python<'_>,
    n: usize,
    epsilon: f64,
) -> Py<PyArray1<usize>>
{
    let lext = compute_2d_domain_extension(n, epsilon);
    lext.as_slice().to_pyarray(py).into()
}

/// B-spline prefiltering on the extended domain with mask propagation
///
/// The function wraps the `core::interp::array1_bspline_prefiltering_ext_gene` implementation
///
/// This implementation follows **Algorithm 4** from Briand & Monasse (2018) {cite}`briand2018theory`.
///
/// The B-spline interpolation process requires both causal and anti-causal recursive 
/// prefiltering applied to the input image. While these filters theoretically require 
/// infinite sums, this implementation approximates them using finite sums as proposed 
/// in the reference article. The approximation relies on either the image's immediate 
/// neighborhood or extrapolated data through boundary conditions.
///
/// # Internal Data Transposition
///
/// The inner function `array1_bspline_prefiltering_ext_var_iter` performs an internal
/// transposition of the data to apply the exponential filter along columns in order
/// to align the column data in memory.
/// This approach is significantly more time-efficient than processing directly, though
/// it requires allocating a temporary buffer to perform the transposition operation
/// efficiently.
///
/// # Mask Influence Propagation (GridR-Specific Extension)
///
/// **Note:** The mask handling strategy described below is an **original contribution 
/// of GridR** and is not part of the reference article, which does not address masking.
///
/// The recursive filters used in B-spline interpolation exhibit an **exponential decay** 
/// property. When a pixel is invalid, its influence propagates to neighboring pixels 
/// through the recursive filtering process, but this influence decreases exponentially 
/// with distance.
///
/// For a pole $z_k$ (where $-1 < z_k < 0$), the exponential filter has the form:
///
/// $$h^{(z_k)}_j = \frac{z_k}{z_k^2 - 1} z_k^{|j|}$$
///
/// This shows that the 1D influence at distance $j$ pixels decays as $|z_k|^{|j|}$.
/// In practice, the dominant pole (largest absolute value, typically $z_1$) determines 
/// the decay rate.
///
/// To determine at what distance $d$ the 1D influence becomes negligible:
///
/// $$|z_k|^{|d|} \le s$$
///
/// where $d$ is the distance in pixels and $s$ is the acceptable residual influence 
/// threshold (`mask_influence_threshold` parameter). This gives:
///
/// $$d \ge \frac{\ln(s)}{\ln(|z_k|)}$$
///
/// Since prefiltering is performed separably on rows and columns, the 2D influence 
/// can be expressed as:
///
/// $$\text{Influence}(i,j) \propto |z_k|^{|i|} \times |z_k|^{|j|} = |z_k|^{|i| + |j|}$$
///
/// This uses Manhattan distance (not Euclidean), yielding a diamond-shaped (45° oriented 
/// square) influence zone centered around the invalid pixel's position.
///
/// Instead of filtering the validity mask (which can produce overshoot artifacts), 
/// this implementation **dilates the invalid area within the validity mask** by the 
/// computed influence radius using a Manhattan distance structuring element.
///
/// Additionally, mask elements corresponding to image pixels used to approximate the 
/// infinite sums (domain extension) are automatically invalidated after the propagation 
/// of invalid mask elements.
///
/// # Important Note on Threshold Selection
///
/// The residual influence threshold $s$ is **relative to the invalid pixel value itself**. 
/// For extreme outlier values (e.g., contaminated pixel with value 10000 in a 0-255 image), 
/// even a small relative threshold (e.g., $s = 10^{-3} = 0.1\%$) can correspond to 
/// significant absolute contamination ($10000 \times 10^{-3} = 10$), requiring a much 
/// larger dilation radius than for typical data values. When dealing with aberrant values, 
/// the threshold $s$ should be chosen to ensure the absolute contamination remains 
/// acceptable for your application.
///
/// # Prerequisites
///
/// This function assumes boundary extension has **already been applied** on the input 
/// image `ima_in`. The required extension size depends on the truncation index, which 
/// is determined by the B-spline order `n` and precision parameter `epsilon`.
///
/// # Parameters
///
/// - `n`: B-spline order (must be odd: 3, 5, 7, 9, or 11)
/// - `epsilon`: Precision parameter for the truncation index calculation. Defines the 
///   acceptable error when approximating the infinite sums. Smaller values require 
///   larger margins for prefiltering. The total required margin combines both the 
///   prefiltering margin (truncation index) and the interpolation kernel radius.
/// - `array_in`: Bound mutable reference to the input 1D array containing source data
/// - `array_in_shape`: Tuple `(depth, rows, cols)` defining the shape of the input array
/// - `array_in_mask`: Optional bound mutable reference to input validity mask
/// - `trunc_idx`: Optional bound immutable reference holding precomputed $N(i, \epsilon)$ 
///   truncation indices. 
/// - `mask_influence_threshold`: Optional residual influence threshold $s$ used to compute the 
///   radius of the propagation of masked data. Required when `ima_mask_in` is provided.
///   Determines the acceptable relative contamination from invalid pixels.
///
/// # Returns
/// - `Ok(())` if prefiltering completes successfully
/// - `Err(PyErr)` if prefiltering fails due to invalid parameters, computation errors, or internal issues
///
/// # References
///
/// Briand, T., & Monasse, P. (2018). Theory and Practice of Image B-Spline 
/// Interpolation. *Image Processing On Line*, 8, 99-141. 
/// https://doi.org/10.5201/ipol.2018.221
#[pyfunction]
#[pyo3(signature = (n, epsilon, array_in, array_in_shape, array_in_mask=None, trunc_idx=None, mask_influence_threshold=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_bspline_prefiltering_f64(
    n: usize,
    epsilon: f64,
    array_in: &Bound<'_, PyArray1<f64>>,
    array_in_shape: (usize, usize, usize),
    array_in_mask: Option<&Bound<'_, PyArray1<u8>>>,
    trunc_idx: Option<&Bound<'_, PyArray1<usize>>>,
    mask_influence_threshold: Option<f64>,
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
    
    // Prepare optional trunc_idx mask to the wrapped core function
    let trunc_idx_readonly = trunc_idx.map(|array| array.readonly());
    let trunc_idx_opt: Option<&[usize]> = trunc_idx_readonly.as_ref().map(|r| r.as_slice()).transpose()?;

    array1_bspline_prefiltering_ext_gene(
        n,
        epsilon,
        trunc_idx_opt,
        &mut array_in_arrayview,
        mask_in_array_view.as_mut(),
        mask_influence_threshold,
    ).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute the safe-valid window after B-spline prefiltering with mask propagation
///
/// The function wraps the `core::interp::array1_bspline_prefiltering_ext_gene_mask_safe_win`
/// implementation.
///
/// This is the **companion function** to [`array1_bspline_prefiltering_ext_gene`] when
/// the caller maintains a *safe-valid window* alongside the validity mask. Instead of
/// running the actual prefiltering, this function predicts how the safe-valid window
/// shrinks once the prefiltering has been applied, *without* materially modifying the
/// mask or the image.
///
/// A *safe-valid window* is a sub-region of the input mask that is guaranteed to
/// contain only valid pixels. Propagating it through a stage that invalidates pixels
/// near borders and around invalid neighbours requires eroding the window by the
/// worst-case influence radius of that stage.
///
/// # Window Shrinkage
///
/// Two effects contribute to the erosion of the safe window during prefiltering:
///
/// 1. **Influence radius propagation:** as documented in
///    [`array1_bspline_prefiltering_ext_gene`], an invalid pixel contaminates its
///    neighbourhood up to a Manhattan distance of $d = \lceil \ln(s) / \ln(|z_k|) \rceil$,
///    where $s$ is the residual influence threshold and $z_k$ is the dominant pole.
///    The safe window must therefore be eroded by `influence_radius` pixels on each
///    side to exclude any region that could be reached by invalid contamination.
///
/// 2. **Domain extension invalidation:** the recursive prefilter relies on extended
///    boundary samples (the first/last `l0 - n/2` rows and columns) that are only
///    valid for prefiltering, not for the subsequent interpolation. These elements
///    are unconditionally invalidated by [`array1_bspline_prefiltering_ext_gene`],
///    so the safe window must also exclude them.
///
/// The final erosion margin is the maximum of these two contributions:
///
/// $$\text{margin} = \max(\text{influence\_radius},\ L_0 + N/2)$$
///
/// where $L_0$ is the domain extension width and $N$ is the B-spline order.
///
/// # Return Value
///
/// Returns `Some(window)` with the shrunk safe-valid window when the erosion still
/// yields a non-empty region inside the input mask, or `None` when the requested
/// margin is too large for the input dimensions or for the original safe window.
/// A `None` result means the caller cannot rely on a safe-valid window after
/// prefiltering and must fall back to per-pixel validity checks.
///
/// # Prerequisites
///
/// This function assumes the same boundary extension contract as
/// [`array1_bspline_prefiltering_ext_gene`]: the input mask `ima_mask_in` is the
/// extended mask, and `ima_mask_in_safe_win` is expressed in the coordinate frame
/// of that extended mask.
///
/// # Parameters
///
/// - `n`: B-spline order (must be odd: 3, 5, 7, 9, or 11)
/// - `epsilon`: Precision parameter for the truncation index calculation. Defines the
///   acceptable error when approximating the infinite sums. Smaller values require
///   larger margins for prefiltering, and therefore shrink the safe window more.
/// - `trunc_idx`: Optional buffer holding the $N(i, \epsilon)$ truncation indices.
///   If not provided, computed internally via `compute_2d_truncation_index`.
/// - `ima_mask_in`: Input mask array (flattened 2D view). Not modified; only its
///   dimensions are used to bound-check the eroded window.
/// - `ima_mask_in_safe_win`: Safe-valid window inside `ima_mask_in` (in extended
///   coordinates).
/// - `mask_influence_threshold`: Residual influence threshold $s$ used to compute
///   the radius of the propagation of masked data. Determines the acceptable
///   relative contamination from invalid pixels.
///
/// # Returns
///
/// - `Some(GxArrayWindow)`: the eroded safe-valid window, when it remains non-empty
///   and fits inside the input mask.
/// - `None`: when the erosion margin is too large for either the input mask
///   dimensions or the original safe window.
///
/// # Panics
///
/// - If `n` is not odd (even orders are not supported)
/// - If `n / 2 == 0` (invalid B-spline order)
#[pyfunction]
#[pyo3(signature = (n, epsilon, array_in_mask, array_in_mask_shape, array_in_mask_safe_win, mask_influence_threshold, trunc_idx=None))]
#[allow(clippy::too_many_arguments)]
pub fn py_array1_bspline_prefiltering_mask_safe_win_f64(
    n: usize,
    epsilon: f64,
    array_in_mask: &Bound<'_, PyArray1<u8>>,
    array_in_mask_shape: (usize, usize),
    array_in_mask_safe_win: PyArrayWindow2,
    mask_influence_threshold: f64,
    trunc_idx: Option<&Bound<'_, PyArray1<usize>>>,
) -> Result<Option<PyArrayWindow2>, PyErr>
{
    // Create a safe read-only array view over the mask
    let array_in_mask_view = array_in_mask.readonly();
    let array_in_mask_slice = array_in_mask_view.as_slice().expect("Failed to get slice");
    let mask_array_view = GxArrayView::new(array_in_mask_slice, 1, array_in_mask_shape.0, array_in_mask_shape.1);

    // Prepare optional trunc_idx to pass to the wrapped core function
    let trunc_idx_readonly = trunc_idx.map(|array| array.readonly());
    let trunc_idx_opt: Option<&[usize]> = trunc_idx_readonly.as_ref().map(|r| r.as_slice()).transpose()?;

    Ok(
        array1_bspline_prefiltering_ext_gene_mask_safe_win(
            n,
            epsilon,
            trunc_idx_opt,
            &mask_array_view,
            &array_in_mask_safe_win.into(),
            mask_influence_threshold,
        )
        .map(|w| PyArrayWindow2::new(w.start_row, w.end_row, w.start_col, w.end_col))
    )
}
