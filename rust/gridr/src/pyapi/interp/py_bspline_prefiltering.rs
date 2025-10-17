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

use crate::core::gx_array::GxArrayViewMut;
use crate::core::interp::gx_bspline_prefiltering::{compute_2d_truncation_index, compute_2d_domain_extension, array1_bspline_prefiltering_ext_gene};

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

