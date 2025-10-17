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
//! Cardinal B-spline prefiltering for B-spline interpolation.
//!
//! This module implements the prefiltering operations required for cardinal B-spline 
//! interpolation, following the theoretical framework described in Briand & Monasse (2018).
//! 
//! # Core Functionality
//!
//! - **Exponential filtering**: Applies causal and anti-causal recursive filters to 1D signals
//! - **B-spline prefiltering**: Processes 2D images with separable row/column filtering
//! - **Mask propagation**: Dilates invalid regions based on filter influence radius (GridR extension)
//! - **Truncation index computation**: Calculates finite approximation parameters for recursive filters
//!
//! # Key Features
//!
//! ## Mask Handling (GridR-Specific)
//!
//! The mask influence propagation strategy is an original contribution of GridR, not part 
//! of the reference article. It accounts for the exponential decay of invalid pixel influence 
//! through recursive filtering, using Manhattan distance-based dilation.
//!
//! ## Supported B-spline Orders
//!
//! Odd orders only: 3, 5, 7, 9, and 11. Filter poles are precomputed as constants.
//!
//! # References
//!
//! Briand, T., & Monasse, P. (2018). Theory and Practice of Image B-Spline Interpolation.
//! *Image Processing On Line*, 8, 99-141. https://doi.org/10.5201/ipol.2018.221
use crate::core::gx_array::GxArrayViewMut;
use crate::core::gx_errors::GxError;
use transpose;


/// Defines poles as constant for differents bspline order
const BSPLINE3_FILTER_POLES: [f64; 1] = [-2.679491924311227e-1];
const BSPLINE5_FILTER_POLES: [f64; 2] = [-4.305753470999738e-1, -4.309628820326465e-2];
const BSPLINE7_FILTER_POLES: [f64; 3] = [-5.352804307964382e-1, -1.225546151923267e-1, -9.148694809608277e-3];
const BSPLINE9_FILTER_POLES: [f64; 4] = [-6.079973891686259e-1, -2.017505201931532e-1, -4.322260854048175e-2, -2.121306903180818e-3];
const BSPLINE11_FILTER_POLES: [f64; 5] = [-6.612660689007345e-1, -2.721803492947859e-1, -8.975959979371331e-2, -1.666962736623466e-2, -5.105575344465021e-4];

/// Maximum buffer size for mu coefficients.
/// 
/// This constant defines the maximum buffer size required for storing both the truncation index 
/// and mu coefficients in B-spline filtering operations. It is set to accommodate the largest 
/// supported spline order (11th order) number of poles (5).
pub const TRUNCATION_INDEX_BUFFER_MAX_SIZE: usize = 5;

/// Maximum buffer size for L coefficients.
/// 
/// This constant defines the maximum buffer size required for storing L coefficients
/// in B-spline filtering operations.
pub const TRUNCATION_L_BUFFER_MAX_SIZE: usize = TRUNCATION_INDEX_BUFFER_MAX_SIZE + 1;

/// Retrieves the filter poles for a specified B-spline order.
/// 
/// This function provides access to precomputed filter poles for different B-spline orders.
/// The poles are stored as constant arrays and retrieved based on the requested order.
/// 
/// # Parameters
/// - `n`: The order of the B-spline (must be one of: 3, 5, 7, 9, 11)
/// 
/// # Returns
/// - Reference to the constant array of f64 values representing the filter poles
/// 
/// # Panics
/// - If the requested spline order `n` is not supported (not in {3, 5, 7, 9, 11})
#[inline(always)]
pub fn get_poles(n: usize) -> &'static [f64]
{
    // Get the poles corresponding to the bspline order
    match n {
        3 => &BSPLINE3_FILTER_POLES,
        5 => &BSPLINE5_FILTER_POLES,
        7 => &BSPLINE7_FILTER_POLES,
        9 => &BSPLINE9_FILTER_POLES,
        11 => &BSPLINE11_FILTER_POLES,
        _ => panic!("Unsupported value for spline order n: {}", n),
    }
}

/// Compute influence radius of a value through the B-Spline preprocessing
///
/// The influence at distance d is proportional to Max(abs(Z_i))^data
/// Max(abs(Z_i))^data <= epsilon => d >= ln(epsilon) / ln(Max(abs(Z_i)))
///
/// # Parameters
/// 
/// - `n`: The bspline order
/// - `threshold`: The influence treshold between 0 and 1
#[inline]
fn compute_prefiltering_influence_radius(n: usize, threshold: f64) -> f64
{
    threshold.ln() / (-1. * get_poles(n)[0]).ln()
}


fn propagate_zeros_influence(mask: &mut [u8], width: usize, height: usize, radius: f64) {
    if radius > 0. {
        let radius_ceil = radius.ceil() as isize;
        let temp_result = mask.to_vec();
        
        // Create the diamond-shaped structuring element using Manhattan L1 distance
        let mut diamond_offsets = Vec::with_capacity((radius * radius * 4.) as usize);
        
        for dy in -radius_ceil as isize..=radius_ceil as isize {
            for dx in -radius_ceil as isize..=radius_ceil as isize {
                let distance_l1 = (dx.abs() + dy.abs()) as f64;
                if distance_l1 <= radius {
                    diamond_offsets.push((dx, dy));
                }
            }
        }
        
        // Dilatate inplace using the structuring element
        for y in 0..height {
            for x in 0..width {
                if temp_result[y * width + x] == 0 {
                    for &(dx, dy) in &diamond_offsets {
                        let ny = (y as isize + dy).max(0) as usize;
                        let nx = (x as isize + dx).max(0) as usize;
                        if ny < height && nx < width {
                            mask[ny * width + nx] = 0;
                        }
                    }
                }
            }
        }
    }
}

/// Truncation indices computation
/// Ref. Thibaud Briand, Pascal Monasse, "Theory and Practice of Image B-Spline Interpolation", https://doi.org/10.5201/ipol.2018.221
/// Eq. (50).
///
/// The computation of the truncation indice is detailed in the above mentioned article as :
///
/// Define (µj) for 1 ≤ j ≤ $$\tilde{n}$$ with $$\tilde{n} = \lfloor \frac{n}{2} \rfloor$$
///
///
/// $$ \begin{cases}
/// \mu_1 = 0 \\
/// \mu_k = \left( \frac{1 + \frac{1}{\log |z_k|} \prod_{i=1}^{k-1} \frac{1}{\log |z_i|}} \right)^{-1}, \quad 2 \leq k \leq \tilde{n}.
/// \end{cases} $$
///
/// Let ε > 0. Define for 1 ≤ i ≤ \tilde{n},
///
/// $$ N(i, \epsilon) = \left\lfloor \frac{\log \left( \epsilon \rho^{(n)} (1 - z_i)(1 - \mu_i) \prod_{j=i+1}^{\tilde{n}} \mu_j \right)}{\log |z_i|} \right\rfloor + 1 $$
///
/// where
///
/// $$ \rho^{(n)} = \left( \prod_{j=1}^{\tilde{n}}  \frac{1 + z_j}{1 - z_j} \right)^2 $$

/// Fallback function to compute mu coefficients given poles
/// Compute mu coefficients based on previous formula using bspline's poles and order
/// The buffer that holds the computed mu coefficients has to be preallocated by the caller.
/// Its size is given by $$\tilde{n} = \lfloor \frac{n}{2} \rfloor$$
/// 
/// # Parameters
/// 
/// - `n`: The bspline order
/// - `poles`: The poles of the corresponding bspline
/// - `mu`: A mutable buffer that will hold the mu coefficients
#[inline]
pub fn compute_truncation_mu(n: usize, poles: &[f64], mu: &mut[f64])
{
    // Compute $$\tilde{n} = \lfloor \frac{n}{2} \rfloor$$
    let nt: usize = n / 2;
    // Init sum_log_pole to hold the cumulative sum of 1/log(abs(zi))
    let mut sum_log_pole: f64 = 0.0;
    // Also note that as poles are defined to be negative, the absolute value
    // value is directly given by their opposite value
    let mut log_pole = (-poles[0]).ln();
    mu[0] = 0.0;
    for k in 1..nt {
        sum_log_pole += 1./log_pole;
        log_pole = (-poles[k]).ln();
        mu[k] = log_pole * sum_log_pole / (1. + log_pole * sum_log_pole);
    }
}

/// Fallback function to compute the truncation index $$N(i, \epsilon)$$ from a precomputed $$\rho$$.
/// 
/// # Parameters
/// 
/// - `n`: The bspline order
/// - `epsilon` : The precision parameter for the truncation index calculation
/// - `rho`: The precomputed $$\rho$$ of the corresponding bspline
/// - `poles`: The poles of the corresponding bspline
/// - `mu`: The precomputed $$\mu$$ of the corresponding bspline
/// - `trunc_idx`: A mutable buffer that will hold the $$N(i, \epsilon)$$ truncation index
#[inline]
pub fn compute_2d_truncation_index_from_rho(n: usize, epsilon: f64, rho: f64, poles: &[f64], mu: &[f64], trunc_idx: &mut[usize])
{
    // As mentioned in Theorem 2 from the reference article, epsilon is adapted for the 2d algorithm
    // in order to achieve the desired precision epsilon
    let ln_epsilon_rho = (0.5 * rho * rho * epsilon).ln();
    let mut mu_prod: f64 = 1.0;
    let nt: usize = n / 2;
    
    // The loop is performed reversed in order to optimize the computation of the mu product by 
    // avoiding an additional inner loop
    // Please notice the index start here at 0 in the trunc_idx buffer, the 0-index corresponds to
    // N(i,epsilon) for i = 1 as it is not defined for i = 0.
    for i in (0..nt).rev() {
        // All poles are negative by definition, we take their opposite instead of computing their
        // absolute value.
        let frac_num = ln_epsilon_rho + ((1. - poles[i]) * (1. - mu[i]) * mu_prod).ln();
        let frac_den = (- poles[i]).ln();
        trunc_idx[i] = 1 + (frac_num / frac_den).floor() as usize;
        mu_prod *= mu[i];
    }
}

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
#[inline]
pub fn compute_2d_truncation_index(n: usize, epsilon: f64
) -> [usize; TRUNCATION_INDEX_BUFFER_MAX_SIZE]
{
    let mut truncation_index_buffer: [usize; TRUNCATION_INDEX_BUFFER_MAX_SIZE] = [0; TRUNCATION_INDEX_BUFFER_MAX_SIZE];
    //let truncation_index: usize = 1;
    // Init the buffer that will hold the mu coefficients
    // We use TRUNCATION_INDEX_BUFFER_MAX_SIZE in order to make this allocation static.
    let mut mu_buffer: [f64; TRUNCATION_INDEX_BUFFER_MAX_SIZE] = [0.0; TRUNCATION_INDEX_BUFFER_MAX_SIZE];
    // Init $$ \rho^{(n)}$$
    let mut rho: f64 = 1.0;
    // Get the poles for order n
    let poles = get_poles(n);
    
    // Compute the truncation $$\mu$$
    compute_truncation_mu(n, &poles, &mut mu_buffer);
    
    // Compute $$\rho$$
    for k in 0..n/2 {
        rho = rho * ((1.0 + poles[k]) / (1.0 - poles[k]));
    }
    rho *= rho;
    
    // Compute truncation index $$N^{(i, \epsilon)}$$ from $$\rho$$
    compute_2d_truncation_index_from_rho(n, epsilon, rho, &poles, &mu_buffer, &mut truncation_index_buffer);
    
    truncation_index_buffer
}

/// Computes the extended domain lengths for B-spline pre-filtering using Approach 1 (Extended Domain) Eq. (54).
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
pub fn compute_2d_domain_extension(n: usize, epsilon: f64
) -> [usize; TRUNCATION_L_BUFFER_MAX_SIZE]
{
    // Compute the truncation index buffer.
    // It holds N(i,epsilon) for i <= 1 <= nt
    // The first element corresponds to N(1, epsilon)
    let truncation_index = compute_2d_truncation_index(n, epsilon);
    
    compute_2d_domain_extension_from_truncation_idx(n, &truncation_index)
}

/// Computes the extended domain lengths for B-spline pre-filtering using Approach 1 (Extended Domain) Eq. (54) with
/// a precomputed truncation index tabulation.
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
pub fn compute_2d_domain_extension_from_truncation_idx(n: usize, trunc_idx: &[usize]) -> [usize; TRUNCATION_L_BUFFER_MAX_SIZE]
{
    let nt = n / 2;
    let mut ext_lj: [usize; TRUNCATION_L_BUFFER_MAX_SIZE] = [0; TRUNCATION_L_BUFFER_MAX_SIZE];
    
    ext_lj[nt] = nt;
    let mut tmp: usize = nt;
    for i in (0..nt).rev() {
        // Get N(i+1, epsilon)
        tmp += trunc_idx[i];
        ext_lj[i] = tmp;
    }
    ext_lj
}


/// Applies an exponential filter $$h(\alpha)$$ in-place to a 1D signal represented as a flat mutable slice.
/// The operation can be applied either along rows or columns of a 2D array depending on the `step` parameter.
/// 
/// When working with 2D arrays:
/// - Use `step = 1` for row-wise filtering.
/// - Use `step = num_columns` for column-wise filtering.
///
/// This implementation follows **Algorithm 3** from the referenced paper.
///
/// # Parameters
/// - `array`: Mutable slice of f64 values representing the signal data. The function modifies the input array in place.
/// - `step`: Step size used to traverse elements. Typically 1 for row-wise operations or the number of columns for column-wise operations.
/// - `alpha`: The BSpline pole value $$\alpha$$, controlling the filter's behavior.
/// - `n`: Total number of elements involved in filtering, including those needed for the infinite sum approximation. Must be greater than $$2n_0$$.
/// - `n0`: Truncation index $$n_0$$, defining where the finite portion of the sum begins and ends within the range `[n0, n - 1 - n0]`.
///
/// # Safety
/// This function does not perform bounds checking or validate consistency between `n` and `n0`. 
/// It is assumed that these parameters are valid and that sufficient memory exists at the indices accessed.
///
/// # Notes
/// - Assumes that the starting point of the signal corresponds to the beginning of the slice (`n0` is relative to this).
/// - The algorithm computes both causal and anti-causal components using approximations derived from equations (38), (44), (48), and (49) in the reference.
pub fn array1_exponential_filter_ext<'a>(
    array: &mut[f64],
    step: usize,
    alpha: f64,
    n: usize,
    n0: usize,
)
{
    // Flat 1D index where to start the finite part of the sum 
    let iflat_start_idx = n0 * step;
    // Flat 1D index where to stop the finite part of the sum
    let iflat_end_idx = (n - 1 - n0) * step;
    // Number of iterations for the finite part of the sum
    let niter = n - 1 - 2*n0;
    
    // Compute infinite sums approximation :
    //   - left : the approximation Eq. (48) of the approximation sum Eq. (38) for the initialization
    //     of the causal filter
    //   - right: the approximation Eq. (49) of the approximation sum Eq. (44) for the initialization
    //     of the anti-causal filter without the first element of the sum.
    //
    // Note. We use the same loop for the performance gain.
    let mut left = array[iflat_start_idx].into();
    let mut right: f64 = 0.0;
    
    // Init the alpha zero power.
    let mut pow_alpha: f64 = 1.;
    
    // Setting starting index for the left and the right infinite sum approximation.
    let mut iflat = iflat_start_idx;
    let mut iflat_r = iflat_end_idx;
    
    if step == 1 {
        for _ in 0..n0 {
            pow_alpha = pow_alpha * alpha;
            
            // Sum on left using Eq. (48)
            iflat -= 1;
            left += array[iflat] * pow_alpha;
            
            // Sum on right using Eq. (49)
            iflat_r += 1;
            right += array[iflat_r] * pow_alpha;
        }
        array[iflat_start_idx] = left;

        // Compute the causal filtering using Eq. (39)
        iflat = iflat_start_idx;
        for _ in 0..niter {
            iflat += 1;
            array[iflat] += left * alpha;
            left = array[iflat];
        }
        
        // Init the anti-causal filtering using Eq. (43)
        right = alpha / (alpha * alpha - 1.0) * (left + right);
        iflat = iflat_end_idx;
        array[iflat] = right;
        
        // Compute the anti-causal filtering using Eq. (47)
        for _ in 0..niter {
            iflat -= 1;
            right = alpha * (right - array[iflat]);
            array[iflat] = right;
        }
    }
    else {
        for _ in 0..n0 {
            pow_alpha = pow_alpha * alpha;
            
            // Sum on left using Eq. (48)
            iflat -= step;
            left += array[iflat] * pow_alpha;
            
            // Sum on right using Eq. (49)
            iflat_r += step;
            right += array[iflat_r] * pow_alpha;
        }
        array[iflat_start_idx] = left;

        // Compute the causal filtering using Eq. (39)
        iflat = iflat_start_idx;
        for _ in 0..niter {
            iflat += step;
            array[iflat] += left * alpha;
            left = array[iflat];
        }
        
        // Init the anti-causal filtering using Eq. (43)
        right = alpha / (alpha * alpha - 1.0) * (left + right);
        iflat = iflat_end_idx;
        array[iflat] = right;
        
        // Compute the anti-causal filtering using Eq. (47)
        for _ in 0..niter {
            iflat -= step;
            right = alpha * (right - array[iflat]);
            array[iflat] = right;
        }
    }
}

/// Prefiltering function on the Extended Domain - internal var iterations
fn array1_bspline_prefiltering_ext_var_iter(
    nt: usize,
    ivar: usize, 
    nrow: usize,
    ncol: usize,
    trunc_idx: &[usize],
    lext: &[usize],
    l0: usize,
    ljump_row: &[usize],
    ljump_col: &[usize],
    lcount_row: &[usize],
    lcount_col: &[usize],
    data: &mut [f64],
    poles: &[f64],
    scratch: &mut [f64],
) -> Result<(), GxError>
{
    let ivar_idx = ivar * nrow * ncol;
    // First index for column.
    // Assigned to 0 because of transposition in temporary buffer
    // With no transposition this would be : let mut icol_idx = ivar_idx;
    let mut icol_idx = 0; // Due to transpose
    
    // Transpose current variable data into temporary buffer.
    // This is way more efficient due to optimized path in exponential filter
    // computation with step = 1
    transpose::transpose(&data[ivar_idx..ivar_idx+nrow*ncol], scratch, ncol, nrow);

    for _icol in 0..ncol {
        // Cascading application of the kth exponential filter
        for k in 0..nt {
            // Here the starting index of the slice passed to the array1_exponential_filter_ext function
            // should be equal to :
            // ivar_idx -> the start of the current variable memory section
            // + icol -> the current column index in the input data 
            // + (L0 - Lext[k]) * ncol -> the full rows to jump

            // If no transposition this would be :
            //    let k_idx = icol_idx + ljump_row[k];
            //    let data_k = &mut data[k_idx..];
            //    array1_exponential_filter_ext(data_k, ncol, poles[k], lcount_row[k], trunc_idx[k]);
            let k_idx = icol_idx + ljump_row[k];
            let data_k = &mut scratch[k_idx..];
            array1_exponential_filter_ext(data_k, 1, poles[k], lcount_row[k], trunc_idx[k]);
        }
        // Due to transposition the column index jumps of nrow.
        // Otherwise this would be : icol_idx += 1;
        icol_idx += nrow;
    }
    // Transpose back into the data buffer
    transpose::transpose(&scratch, &mut data[ivar_idx..ivar_idx+nrow*ncol], nrow, ncol);
    
    // Prefiltering on the rows
    let irow_pad = l0 - lext[nt];
    let mut irow_idx = ivar_idx + irow_pad * ncol;
    
    for _irow in irow_pad..nrow-irow_pad {
        // Cascading application of the kth exponential filter
        for k in 0..nt {
            // Here the starting index of the slice passed to the array1_exponential_filter_ext function
            // should be equal to :
            // ivar_idx -> the start of the current variable memory section
            // + irow * ncol -> the number of rows to jump 
            // + (L0 - Lext[k]) -> the number of columns to jump
            
            let k_idx = irow_idx + ljump_col[k];
            let data_k = &mut data[k_idx..];
            array1_exponential_filter_ext(data_k, 1, poles[k], lcount_col[k], trunc_idx[k]);
            
        }
        irow_idx += ncol;
    }
    Ok(())
}


/// B-spline prefiltering on the extended domain with mask propagation
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
/// - `trunc_idx`: Optional buffer holding the $N(i, \epsilon)$ truncation indices. 
///   If not provided, computed internally via `compute_2d_truncation_index`.
/// - `ima_in`: Mutable input data array (flattened 3D view). Modified in-place during 
///   prefiltering.
/// - `ima_mask_in`: Optional mutable input mask array (flattened 2D view). If provided, 
///   invalid areas are dilated based on the influence radius, and border regions 
///   corresponding to the domain extension are invalidated. Modified in-place.
/// - `mask_influence_threshold`: Residual influence threshold $s$ used to compute the 
///   radius of the propagation of masked data. Required when `ima_mask_in` is provided.
///   Determines the acceptable relative contamination from invalid pixels.
///
/// # Returns
///
/// Returns `Ok(())` on success, or `Err(GxError)` if an error occurs during processing.
///
/// # Panics
///
/// - If `n` is not odd (even orders are not supported)
/// - If `n / 2 == 0` (invalid B-spline order)
/// - If input dimensions are too small relative to domain extension requirements
///
/// # References
///
/// Briand, T., & Monasse, P. (2018). Theory and Practice of Image B-Spline 
/// Interpolation. *Image Processing On Line*, 8, 99-141. 
/// https://doi.org/10.5201/ipol.2018.221
pub fn array1_bspline_prefiltering_ext_gene<'a>(
        n: usize,
        epsilon: f64,
        trunc_idx: Option<&'a [usize]>,
        ima_in: &mut GxArrayViewMut<'_, f64>,
        ima_mask_in: Option<&'a mut GxArrayViewMut<'a, u8>>,
        mask_influence_treshold: Option<f64>,
) -> Result<(), GxError>
{
    // Retrieve the number of poles
    let nt = n / 2;
    assert!(nt > 0);
    // Ensure bspline is odd (otherwise we need to implement normalization)
    assert!(n % 2 == 1, "Only odd order are supported");
    
    // Get the poles corresponding to the bspline order n
    let poles = get_poles(n);
    
    // The parameter trunc_idx is optional, if not given it must be computed.
    let trunc_idx = match trunc_idx {
        Some(buffer) => buffer,
        None => {
            // trunc_idx was not given we have to compute it.
            &compute_2d_truncation_index(n, epsilon)
        }
    };
    
    // Compute the required domain extension based on the truncation index for order n
    let lext = compute_2d_domain_extension_from_truncation_idx(n, &trunc_idx);
    let l0 = lext[0];
    
    // We assume boundaries extension have already been applied on the input arrayview `ima_in`.
    assert!(ima_in.ncol > 2*l0, "Test ncol={0} > 2*l0 with l0={1}", ima_in.ncol, l0);
    assert!(ima_in.nrow > 2*l0, "Test nrow={0} > 2*l0 with l0={1}", ima_in.nrow, l0);

    // Precompute (L0 - Lext[k]) * NROW for column loop
    // and (L0 - Lext[k]) * NROW for row loop
    let mut lk_diff: usize;    
    let mut ljump_row: [usize; TRUNCATION_L_BUFFER_MAX_SIZE] = [0; TRUNCATION_L_BUFFER_MAX_SIZE];
    let mut ljump_col: [usize; TRUNCATION_L_BUFFER_MAX_SIZE] = [0; TRUNCATION_L_BUFFER_MAX_SIZE];
    // Also precompute the number of element to compute for each pole along each dimension
    let mut lcount_row: [usize; TRUNCATION_L_BUFFER_MAX_SIZE] = [0; TRUNCATION_L_BUFFER_MAX_SIZE];
    let mut lcount_col: [usize; TRUNCATION_L_BUFFER_MAX_SIZE] = [0; TRUNCATION_L_BUFFER_MAX_SIZE];
    for k in 0..nt {
        lk_diff = l0 - lext[k];
        // Because of transposition used in array1_bspline_prefiltering_ext_var_iter
        // otherwise this would be ljump_row[k] = lk_diff * ima_in.ncol;
        ljump_row[k] = lk_diff; 
        ljump_col[k] = lk_diff;
        lcount_row[k] = ima_in.nrow - 2 * lk_diff;
        lcount_col[k] = ima_in.ncol - 2 * lk_diff;
    }
 
    //--------------------------------------------------------------------------
    // Prefiltering on image
    //--------------------------------------------------------------------------
    // Allocate a temporary buffer passed to array1_bspline_prefiltering_ext_var_iter
    // in order to perform data transposition.
    let mut scratch = vec![0.; ima_in.nrow * ima_in.ncol];
    
    for ivar in 0..ima_in.nvar {
        array1_bspline_prefiltering_ext_var_iter(nt, ivar, ima_in.nrow, ima_in.ncol,
            &trunc_idx, &lext, l0, &ljump_row, &ljump_col, &lcount_row, &lcount_col,
            &mut ima_in.data, &poles, &mut scratch)?;
    }
 
    //--------------------------------------------------------------------------
    // Prefiltering on mask if given
    //--------------------------------------------------------------------------
    match ima_mask_in {
        Some(mask_view) => {
                // Compute the influence radius of an element
                let influence_treshold = mask_influence_treshold.expect("Error : missing optional parameter `influence_threshold`");
                let influence_radius = compute_prefiltering_influence_radius(n, influence_treshold);
                
                // Propagate the zeros
                propagate_zeros_influence(&mut mask_view.data, mask_view.ncol, mask_view.nrow, influence_radius);
                
                // By definition the prefiltering process is only valid on elements that will be
                // used for the interpolation part, ie. the first and last (l0 - n/2) elements
                // Should be marked as invalid. Anyhow they do not serve the interpolation.
                // We also have to normalize the values lying in the complementary region in order
                // to be interpreted as mask (1: valid, 0: invalid).
                // For that we have here to apply the normalization factor that is not applied on 
                // image as the normalization as been moved into the B-Spline functions expressions.
                // For an odd order N B-Spline it is equal to squared factorial N (N! * N!)
                
                // First loop on the invalid
                //let mut norm_factor: f64 = (1..=n).map(|x| x as f64).product();
                //norm_factor *= norm_factor;
                let margin = l0 - nt;
                // First index to be normalized, ie. not invalid by construction
                let norm_flat_row_start = mask_view.ncol * margin;
                // Last index to be normalized, ie. not invalid by construction
                let norm_flat_row_stop = (mask_view.nrow - margin)*mask_view.ncol - 1;
                // Number of rows to normalized
                let norm_nrow = mask_view.nrow - 2 * margin;
                let mask_size = mask_view.ncol * mask_view.nrow;
                
                // top
                for idx in 0..norm_flat_row_start {
                    mask_view.data[idx] = 0;
                }
                // bottom
                for idx in norm_flat_row_stop+1..mask_size {
                    mask_view.data[idx] = 0;
                }
                
                let mut row_flat_idx = norm_flat_row_start;
                
                for _ in 0..norm_nrow {
                    
                    let mut row_flat_idx_rev = row_flat_idx + mask_view.ncol - 1;
                    
                    // Invalidate data on left and right borders
                    for _ in 0..margin {
                        // Invalidate left
                        mask_view.data[row_flat_idx] = 0;
                        // Invalidate right
                        mask_view.data[row_flat_idx_rev] = 0;
                        row_flat_idx = row_flat_idx + 1;
                        row_flat_idx_rev = row_flat_idx_rev - 1;
                    }
                    // Loop on inside to normalize them
                    //for idx in row_flat_idx..=row_flat_idx_rev {
                    //    mask_f64[idx] = mask_f64[idx] * norm_factor;
                    //    mask_view.data[idx] = (mask_f64[idx] >= 0.99) as u8;
                    //}
                    
                    // Prepare for next iter
                    row_flat_idx = row_flat_idx_rev + margin + 1;
                }
        }
        None => {}
    };
    
    Ok(())
}


// Tests unitaires
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gx_array::{GxArrayViewMut};
    use rstest::rstest;
    
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
    
    #[rstest]
    #[case(3, 0.01, 3.5)]
    #[case(3, 0.001, 5.25)]
    #[case(5, 0.001, 8.20)]
    #[case(7, 0.001, 11.05)]
    #[case(9, 0.001, 13.88)]
    #[case(11, 0.001, 16.70)]
    fn test_compute_prefiltering_influence_radius(
        #[case] n: usize,
        #[case] threshold : f64,
        #[case] expected: f64,
    )
    {
        let d: f64 = compute_prefiltering_influence_radius(n, threshold);
        assert!((d - expected).abs() <= 1e-2); 
    }
    
    #[rstest]
    // full valid => full valid
    #[case(&mut [1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1],
            6, 5, 3.,
            &[1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1])]
    // radius 0 => no change
    #[case(&mut [1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 0, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1],
            6, 5, 0.,
            &[1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1,
             1, 1, 1, 0, 1, 1,
             1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1])]
    // check with radius < 2 (manhattan distance)
    #[case(&mut [1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 0, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1],
            6, 5, 1.99,
            &[1, 1, 1, 1, 1, 1,
             1, 1, 0, 1, 1, 1,
             1, 0, 0, 0, 1, 1,
             1, 1, 0, 1, 1, 1,
             1, 1, 1, 1, 1, 1])]
    // check with radius >= 2 (manhattan distance)
    #[case(&mut [1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 0, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1],
            6, 5, 2.,
            &[1, 1, 0, 1, 1, 1,
             1, 0, 0, 0, 1, 1,
             0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 1, 1,
             1, 1, 0, 1, 1, 1])]
    #[case(&mut [1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 0, 0, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1],
            6, 5, 2.,
            &[1, 1, 0, 0, 1, 1,
              1, 0, 0, 0, 0, 1,
              0, 0, 0, 0, 0, 0,
              1, 0, 0, 0, 0, 1,
              1, 1, 0, 0, 1, 1])]
    #[case(&mut [0, 1, 1, 1, 1, 0,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1,
                 0, 1, 1, 1, 1, 0],
            6, 5, 1.,
            &[0, 0, 1, 1, 0, 0,
             0, 1, 1, 1, 1, 0,
             1, 1, 1, 1, 1, 1,
             0, 1, 1, 1, 1, 0,
             0, 0, 1, 1, 0, 0])]
    fn test_propagate_zeros_influence(
        #[case] mut mask: &mut [u8],
        #[case] width: usize,
        #[case] height: usize,
        #[case] radius: f64,
        #[case] expected: &[u8],
    )
    {
        propagate_zeros_influence(&mut mask, width, height, radius);
        assert_eq!(&mask, &expected);
    }

    #[rstest]
    #[case(3, &mut [0.,], &BSPLINE3_FILTER_POLES, &[0.0,])]
    #[case(5, &mut [0.,0., ], &BSPLINE5_FILTER_POLES, &[0.0, 7.886523126940e-01])]
    #[case(7, &mut [0.,0.,0., ], &BSPLINE7_FILTER_POLES, &[0.0, 7.705847640303e-01, 9.069526580526e-01])]
    #[case(9, &mut [0.,0.,0.,0., ], &BSPLINE9_FILTER_POLES, &[0.0, 7.628638545451e-01, 8.921921530330e-01, 9.478524258427e-01])]
    #[case(11, &mut [0.,0.,0.,0.,0., ], &BSPLINE11_FILTER_POLES, &[0.0, 7.588188483679e-01, 8.848043627422e-01, 9.364817379863e-01, 9.668300605685e-01])]
    fn test_compute_truncation_mu(
        #[case] n: usize,
        #[case] mut mu_buffer: &mut [f64],
        #[case] poles: &[f64],
        #[case] expected: &[f64],
    
    ) {
        compute_truncation_mu(n, &poles, &mut mu_buffer);
        assert!(approx_eq(&mu_buffer, &expected, 1e-10));

    }
    
    /// Checks the computation of the 1-side domain extension $$L_0^(n, \epsilon)$$ for the 2D algorithm.
    /// The expected values are not taken from the Table 4. in the reference paper, as it shows incoherency with
    /// in text previous mentionned values and self computation from the formulae.
    #[rstest]
    #[case(3, 1e-2, 7)]
    #[case(3, 1e-3, 9)]
    #[case(3, 1e-4, 11)]
    #[case(3, 1e-5, 12)]
    #[case(3, 1e-6, 14)]
    #[case(3, 1e-7, 16)]
    #[case(3, 1e-8, 18)]
    #[case(3, 1e-9, 19)]
    #[case(3, 1e-10, 21)]
    #[case(3, 1e-11, 23)]
    #[case(3, 1e-12, 24)]
    #[case(5, 1e-2, 17)]
    #[case(5, 1e-3, 21)]
    #[case(5, 1e-4, 24)]
    #[case(5, 1e-5, 28)]
    #[case(5, 1e-6, 31)]
    #[case(5, 1e-7, 35)]
    #[case(5, 1e-8, 38)]
    #[case(5, 1e-9, 42)]
    #[case(5, 1e-10, 45)]
    #[case(5, 1e-11, 49)]
    #[case(5, 1e-12, 52)]
    #[case(7, 1e-2, 30)]
    #[case(7, 1e-3, 37)]
    #[case(7, 1e-4, 42)]
    #[case(7, 1e-5, 47)]
    #[case(7, 1e-6, 52)]
    #[case(7, 1e-7, 58)]
    #[case(7, 1e-8, 62)]
    #[case(7, 1e-9, 68)]
    #[case(7, 1e-10, 73)]
    #[case(7, 1e-11, 78)]
    #[case(7, 1e-12, 83)]
    #[case(9, 1e-2, 47)]
    #[case(9, 1e-3, 55)]
    #[case(9, 1e-4, 63)]
    #[case(9, 1e-5, 70)]
    #[case(9, 1e-6, 77)]
    #[case(9, 1e-7, 85)]
    #[case(9, 1e-8, 91)]
    #[case(9, 1e-9, 99)]
    #[case(9, 1e-10, 105)]
    #[case(9, 1e-11, 112)]
    #[case(9, 1e-12, 121)]
    #[case(11, 1e-2, 70)]
    #[case(11, 1e-3, 78)]
    #[case(11, 1e-4, 87)]
    #[case(11, 1e-5, 96)]
    #[case(11, 1e-6, 105)]
    #[case(11, 1e-7, 115)]
    #[case(11, 1e-8, 125)]
    #[case(11, 1e-9, 133)]
    #[case(11, 1e-10, 142)]
    #[case(11, 1e-11, 152)]
    #[case(11, 1e-12, 161)]
    fn test_compute_2d_domain_extension(
        #[case] n: usize,
        #[case] epsilon: f64,
        #[case] expected: usize,
    )
    {
        let result = compute_2d_domain_extension(n, epsilon);
        assert_eq!(result[0], expected, "Index 0 for n={} and epsilon={} does not match with expected", n, epsilon);
    }
    

    /// Test for array1_exponential_filter_ext function
    #[rstest]
    #[case(&mut [39.0,  47.0,  7.0,  60.0,  90.0,  83.0 , 124.0,  58.0, 61.0,  42.0,  92.0,  57.0 , 113.0,  30.0,  24.0 ],
           1, -0.5, 15, 3,
           &[39.0, 47.0, 7.0, 16.8671875, 29.640625, -0.96875, 55.78125, -14.484375, 38.4296875, -20.58984375, 55.044921875, -25.0224609375, 113.0, 30.0, 24.0])]
    #[case(&mut [39.0,  47.0,  7.0,  60.0,  90.0,  83.0 , 124.0,  58.0, 61.0,  42.0,  92.0,  57.0 , 113.0,  30.0,  24.0, 130.0, 33.0, ],
           2, -0.5, 9, 2,
           &[39.0, 47.0, 7.0, 60.0, 29.885416666666668, 83.0, 36.479166666666664, 58.0, 2.916666666666666, 42.0, 17.229166666666668, 57.0, 46.010416666666664, 30.0, 24.0, 130.0, 33.0])]
    
    #[case(&mut [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19. ],
           1, 0.5, 20, 7,
           &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -14.006999, -15.998372, -17.988932, -19.973958, -21.945964, -23.890951, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])]
    fn test_array1_exponential_filter_ext(
        #[case] mut array: &mut [f64],
        #[case] step: usize,
        #[case] alpha: f64,
        #[case] n: usize,
        #[case] n0: usize,
        #[case] expected: &[f64],
    )
    {
        array1_exponential_filter_ext(&mut array, step, alpha, n, n0);
        assert!(approx_eq(&array, &expected, 1e-6));
    }
    
    
    /// Test for array1_bspline_prefiltering_ext_gene
    #[rstest]
    // First test with B-Spline order 3 and precision at 2 decimals - input mask is full valid
    #[case(3, 1e-2, None, 29, 24,
           &mut [77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.,
                 67., 66., 65., 64., 63., 62., 61., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 68., 67., 66., 65., 64., 63., 62.,
                 57., 56., 55., 54., 53., 52., 51., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 58., 57., 56., 55., 54., 53., 52.,
                 47., 46., 45., 44., 43., 42., 41., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 48., 47., 46., 45., 44., 43., 42.,
                 37., 36., 35., 34., 33., 32., 31., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 38., 37., 36., 35., 34., 33., 32.,
                 27., 26., 25., 24., 23., 22., 21., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 28., 27., 26., 25., 24., 23., 22.,
                 17., 16., 15., 14., 13., 12., 11., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 18., 17., 16., 15., 14., 13., 12.,
                 7., 6., 5., 4., 3., 2., 1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 8., 7., 6., 5., 4., 3., 2.,
                 17., 16., 15., 14., 13., 12., 11., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 18., 17., 16., 15., 14., 13., 12.,
                 27., 26., 25., 24., 23., 22., 21., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 28., 27., 26., 25., 24., 23., 22.,
                 37., 36., 35., 34., 33., 32., 31., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 38., 37., 36., 35., 34., 33., 32.,
                 47., 46., 45., 44., 43., 42., 41., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 48., 47., 46., 45., 44., 43., 42.,
                 57., 56., 55., 54., 53., 52., 51., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 58., 57., 56., 55., 54., 53., 52.,
                 67., 66., 65., 64., 63., 62., 61., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 68., 67., 66., 65., 64., 63., 62.,
                 77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.,
                 87., 86., 85., 84., 83., 82., 81., 80., 81., 82., 83., 84., 85., 86., 87., 88., 89., 88., 87., 86., 85., 84., 83., 82.,
                 97., 96., 95., 94., 93., 92., 91., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 98., 97., 96., 95., 94., 93., 92.,
                 107., 106., 105., 104., 103., 102., 101., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 108., 107., 106., 105., 104., 103., 102.,
                 117., 116., 115., 114., 113., 112., 111., 110., 111., 112., 113., 114., 115., 116., 117., 118., 119., 118., 117., 116., 115., 114., 113., 112.,
                 127., 126., 125., 124., 123., 122., 121., 120., 121., 122., 123., 124., 125., 126., 127., 128., 129., 128., 127., 126., 125., 124., 123., 122.,
                 137., 136., 135., 134., 133., 132., 131., 130., 131., 132., 133., 134., 135., 136., 137., 138., 139., 138., 137., 136., 135., 134., 133., 132.,
                 147., 146., 145., 144., 143., 142., 141., 140., 141., 142., 143., 144., 145., 146., 147., 148., 149., 148., 147., 146., 145., 144., 143., 142.,
                 137., 136., 135., 134., 133., 132., 131., 130., 131., 132., 133., 134., 135., 136., 137., 138., 139., 138., 137., 136., 135., 134., 133., 132.,
                 127., 126., 125., 124., 123., 122., 121., 120., 121., 122., 123., 124., 125., 126., 127., 128., 129., 128., 127., 126., 125., 124., 123., 122.,
                 117., 116., 115., 114., 113., 112., 111., 110., 111., 112., 113., 114., 115., 116., 117., 118., 119., 118., 117., 116., 115., 114., 113., 112.,
                 107., 106., 105., 104., 103., 102., 101., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 108., 107., 106., 105., 104., 103., 102.,
                 97., 96., 95., 94., 93., 92., 91., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 98., 97., 96., 95., 94., 93., 92.,
                 87., 86., 85., 84., 83., 82., 81., 80., 81., 82., 83., 84., 85., 86., 87., 88., 89., 88., 87., 86., 85., 84., 83., 82.,
                 77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.],
           &mut [1u8; 29*24],
           0.001, // accepted influence of 0.1%
           &[ 77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000, 71.000000, 70.000000, 71.000000, 72.000000, 73.000000, 74.000000, 75.000000, 76.000000, 77.000000, 78.000000, 79.000000, 78.000000, 77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000,
                67.000000, 66.000000, 65.000000, 64.000000, 63.000000, 62.000000, 61.000000, 60.000000, 61.000000, 62.000000, 63.000000, 64.000000, 65.000000, 66.000000, 67.000000, 68.000000, 69.000000, 68.000000, 67.000000, 66.000000, 65.000000, 64.000000, 63.000000, 62.000000,
                57.000000, 56.000000, 55.000000, 54.000000, 53.000000, 52.000000, 51.000000, 50.000000, 51.000000, 52.000000, 53.000000, 54.000000, 55.000000, 56.000000, 57.000000, 58.000000, 59.000000, 58.000000, 57.000000, 56.000000, 55.000000, 54.000000, 53.000000, 52.000000,
                47.000000, 46.000000, 45.000000, 44.000000, 43.000000, 42.000000, 41.000000, 40.000000, 41.000000, 42.000000, 43.000000, 44.000000, 45.000000, 46.000000, 47.000000, 48.000000, 49.000000, 48.000000, 47.000000, 46.000000, 45.000000, 44.000000, 43.000000, 42.000000,
                37.000000, 36.000000, 35.000000, 34.000000, 33.000000, 32.000000, 31.000000, 30.000000, 31.000000, 32.000000, 33.000000, 34.000000, 35.000000, 36.000000, 37.000000, 38.000000, 39.000000, 38.000000, 37.000000, 36.000000, 35.000000, 34.000000, 33.000000, 32.000000,
                27.000000, 26.000000, 25.000000, 24.000000, 23.000000, 22.000000, 21.000000, 20.000000, 21.000000, 22.000000, 23.000000, 24.000000, 25.000000, 26.000000, 27.000000, 28.000000, 29.000000, 28.000000, 27.000000, 26.000000, 25.000000, 24.000000, 23.000000, 22.000000,
                3.093084, 2.926395, 2.759706, 2.593016, 2.426327, 2.259638, 0.353195, 0.304984, 0.353128, 0.375452, 0.404703, 0.432064, 0.460056, 0.487418, 0.516666, 0.539001, 0.587104, 0.539046, 3.093084, 2.926395, 2.759706, 2.593016, 2.426327, 2.259638,
                0.203903, 0.037242, -0.129419, -0.296079, -0.462740, -0.629400, -0.128372, -0.176493, -0.128379, -0.106053, -0.076809, -0.049451, -0.021465, 0.005892, 0.035137, 0.057462, 0.105579, 0.057446, 0.203903, 0.037242, -0.129419, -0.296079, -0.462740, -0.629400,
                3.091305, 2.924637, 2.757969, 2.591300, 2.424632, 2.257964, 0.352919, 0.304714, 0.352852, 0.375173, 0.404420, 0.431778, 0.459766, 0.487125, 0.516370, 0.538701, 0.586799, 0.538746, 3.091305, 2.924637, 2.757969, 2.591300, 2.424632, 2.257964,
                4.430877, 4.264211, 4.097544, 3.930878, 3.764212, 3.597546, 0.576213, 0.527970, 0.576118, 0.598436, 0.627684, 0.655041, 0.683029, 0.710388, 0.739631, 0.761965, 0.810051, 0.762038, 4.430877, 4.264211, 4.097544, 3.930878, 3.764212, 3.597546,
                6.185188, 6.018521, 5.851854, 5.685187, 5.518521, 5.351854, 0.868638, 0.820344, 0.868506, 0.890820, 0.920069, 0.947426, 0.975414, 1.002773, 1.032015, 1.054353, 1.102426, 1.054463, 6.185188, 6.018521, 5.851854, 5.685187, 5.518521, 5.351854,
                7.828372, 7.661706, 7.495039, 7.328372, 7.161706, 6.995039, 1.142539, 1.094199, 1.142373, 1.164683, 1.193933, 1.221290, 1.249278, 1.276637, 1.305879, 1.328220, 1.376280, 1.328364, 7.828372, 7.661706, 7.495039, 7.328372, 7.161706, 6.995039,
                9.501323, 9.334656, 9.167990, 9.001323, 8.834656, 8.667990, 1.421402, 1.373013, 1.421200, 1.443508, 1.472758, 1.500115, 1.528103, 1.555463, 1.584703, 1.607048, 1.655095, 1.607227, 9.501323, 9.334656, 9.167990, 9.001323, 8.834656, 8.667990,
                11.166336, 10.999669, 10.833003, 10.666336, 10.499669, 10.333003, 1.698942, 1.650506, 1.698705, 1.721009, 1.750261, 1.777618, 1.805605, 1.832965, 1.862205, 1.884553, 1.932587, 1.884767, 11.166336, 10.999669, 10.833003, 10.666336, 10.499669, 10.333003,
                12.833333, 12.666667, 12.500000, 12.333333, 12.166667, 12.000000, 1.976812, 1.928328, 1.976541, 1.998841, 2.028094, 2.055450, 2.083438, 2.110798, 2.140037, 2.162388, 2.210410, 2.162637, 12.833333, 12.666667, 12.500000, 12.333333, 12.166667, 12.000000,
                14.500330, 14.333664, 14.166997, 14.000330, 13.833664, 13.666997, 2.254683, 2.206151, 2.254376, 2.276673, 2.305927, 2.333283, 2.361271, 2.388631, 2.417869, 2.440224, 2.488233, 2.440508, 14.500330, 14.333664, 14.166997, 14.000330, 13.833664, 13.666997,
                16.165345, 15.998678, 15.832012, 15.665345, 15.498678, 15.332012, 2.532223, 2.483644, 2.531882, 2.554175, 2.583429, 2.610786, 2.638773, 2.666134, 2.695371, 2.717729, 2.765725, 2.718048, 16.165345, 15.998678, 15.832012, 15.665345, 15.498678, 15.332012,
                17.838290, 17.671623, 17.504956, 17.338290, 17.171623, 17.004956, 2.811085, 2.762458, 2.810708, 2.832999, 2.862254, 2.889610, 2.917597, 2.944958, 2.974194, 2.996556, 3.044539, 2.996910, 17.838290, 17.671623, 17.504956, 17.338290, 17.171623, 17.004956,
                19.481497, 19.314830, 19.148163, 18.981497, 18.814830, 18.648163, 3.084989, 3.036315, 3.084579, 3.106866, 3.136122, 3.163477, 3.191465, 3.218826, 3.248061, 3.270426, 3.318397, 3.270815, 19.481497, 19.314830, 19.148163, 18.981497, 18.814830, 18.648163,
                21.235723, 21.069057, 20.902391, 20.735724, 20.569058, 20.402392, 3.377401, 3.328676, 3.376953, 3.399236, 3.428493, 3.455849, 3.483836, 3.511197, 3.540431, 3.562800, 3.610758, 3.563225, 21.235723, 21.069057, 20.902391, 20.735724, 20.569058, 20.402392,
                22.575611, 22.408943, 22.242275, 22.075606, 21.908938, 21.742270, 3.600744, 3.551980, 3.600268, 3.622549, 3.651807, 3.679162, 3.707150, 3.734512, 3.763746, 3.786117, 3.834065, 3.786571, 22.575611, 22.408943, 22.242275, 22.075606, 21.908938, 21.742270,
                25.461832, 25.295172, 25.128511, 24.961850, 24.795190, 24.628529, 4.081853, 4.033009, 4.081317, 4.103591, 4.132849, 4.160203, 4.188189, 4.215550, 4.244781, 4.267158, 4.315081, 4.267671, 25.461832, 25.295172, 25.128511, 24.961850, 24.795190, 24.628529,
                22.577060, 22.410370, 22.243681, 22.076992, 21.910303, 21.743613, 3.600965, 3.552195, 3.600489, 3.622773, 3.652034, 3.679393, 3.707385, 3.734750, 3.763987, 3.786362, 3.834315, 3.786815, 22.577060, 22.410370, 22.243681, 22.076992, 21.910303, 21.743613,
                127.000000, 126.000000, 125.000000, 124.000000, 123.000000, 122.000000, 121.000000, 120.000000, 121.000000, 122.000000, 123.000000, 124.000000, 125.000000, 126.000000, 127.000000, 128.000000, 129.000000, 128.000000, 127.000000, 126.000000, 125.000000, 124.000000, 123.000000, 122.000000,
                117.000000, 116.000000, 115.000000, 114.000000, 113.000000, 112.000000, 111.000000, 110.000000, 111.000000, 112.000000, 113.000000, 114.000000, 115.000000, 116.000000, 117.000000, 118.000000, 119.000000, 118.000000, 117.000000, 116.000000, 115.000000, 114.000000, 113.000000, 112.000000,
                107.000000, 106.000000, 105.000000, 104.000000, 103.000000, 102.000000, 101.000000, 100.000000, 101.000000, 102.000000, 103.000000, 104.000000, 105.000000, 106.000000, 107.000000, 108.000000, 109.000000, 108.000000, 107.000000, 106.000000, 105.000000, 104.000000, 103.000000, 102.000000,
                97.000000, 96.000000, 95.000000, 94.000000, 93.000000, 92.000000, 91.000000, 90.000000, 91.000000, 92.000000, 93.000000, 94.000000, 95.000000, 96.000000, 97.000000, 98.000000, 99.000000, 98.000000, 97.000000, 96.000000, 95.000000, 94.000000, 93.000000, 92.000000,
                87.000000, 86.000000, 85.000000, 84.000000, 83.000000, 82.000000, 81.000000, 80.000000, 81.000000, 82.000000, 83.000000, 84.000000, 85.000000, 86.000000, 87.000000, 88.000000, 89.000000, 88.000000, 87.000000, 86.000000, 85.000000, 84.000000, 83.000000, 82.000000,
                77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000, 71.000000, 70.000000, 71.000000, 72.000000, 73.000000, 74.000000, 75.000000, 76.000000, 77.000000, 78.000000, 79.000000, 78.000000, 77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000],
           & [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           true)]
    // Another test with B-Spline order 3 and precision at 2 decimals - input mask has one nonvalid point
    #[case(3, 1e-2, None, 29, 24,
           &mut [77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.,
                 67., 66., 65., 64., 63., 62., 61., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 68., 67., 66., 65., 64., 63., 62.,
                 57., 56., 55., 54., 53., 52., 51., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 58., 57., 56., 55., 54., 53., 52.,
                 47., 46., 45., 44., 43., 42., 41., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 48., 47., 46., 45., 44., 43., 42.,
                 37., 36., 35., 34., 33., 32., 31., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 38., 37., 36., 35., 34., 33., 32.,
                 27., 26., 25., 24., 23., 22., 21., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 28., 27., 26., 25., 24., 23., 22.,
                 17., 16., 15., 14., 13., 12., 11., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 18., 17., 16., 15., 14., 13., 12.,
                 7., 6., 5., 4., 3., 2., 1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 8., 7., 6., 5., 4., 3., 2.,
                 17., 16., 15., 14., 13., 12., 11., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 18., 17., 16., 15., 14., 13., 12.,
                 27., 26., 25., 24., 23., 22., 21., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 28., 27., 26., 25., 24., 23., 22.,
                 37., 36., 35., 34., 33., 32., 31., 30., 31., 32., 33., 34., 35., 36., 37., 38., 39., 38., 37., 36., 35., 34., 33., 32.,
                 47., 46., 45., 44., 43., 42., 41., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 48., 47., 46., 45., 44., 43., 42.,
                 57., 56., 55., 54., 53., 52., 51., 50., 51., 52., 53., 54., 55., 56., 57., 58., 59., 58., 57., 56., 55., 54., 53., 52.,
                 67., 66., 65., 64., 63., 62., 61., 60., 61., 62., 63., 64., 65., 66., 67., 68., 69., 68., 67., 66., 65., 64., 63., 62.,
                 77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.,
                 87., 86., 85., 84., 83., 82., 81., 80., 81., 82., 83., 84., 85., 86., 87., 88., 89., 88., 87., 86., 85., 84., 83., 82.,
                 97., 96., 95., 94., 93., 92., 91., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 98., 97., 96., 95., 94., 93., 92.,
                 107., 106., 105., 104., 103., 102., 101., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 108., 107., 106., 105., 104., 103., 102.,
                 117., 116., 115., 114., 113., 112., 111., 110., 111., 112., 113., 114., 115., 116., 117., 118., 119., 118., 117., 116., 115., 114., 113., 112.,
                 127., 126., 125., 124., 123., 122., 121., 120., 121., 122., 123., 124., 125., 126., 127., 128., 129., 128., 127., 126., 125., 124., 123., 122.,
                 137., 136., 135., 134., 133., 132., 131., 130., 131., 132., 133., 134., 135., 136., 137., 138., 139., 138., 137., 136., 135., 134., 133., 132.,
                 147., 146., 145., 144., 143., 142., 141., 140., 141., 142., 143., 144., 145., 146., 147., 148., 149., 148., 147., 146., 145., 144., 143., 142.,
                 137., 136., 135., 134., 133., 132., 131., 130., 131., 132., 133., 134., 135., 136., 137., 138., 139., 138., 137., 136., 135., 134., 133., 132.,
                 127., 126., 125., 124., 123., 122., 121., 120., 121., 122., 123., 124., 125., 126., 127., 128., 129., 128., 127., 126., 125., 124., 123., 122.,
                 117., 116., 115., 114., 113., 112., 111., 110., 111., 112., 113., 114., 115., 116., 117., 118., 119., 118., 117., 116., 115., 114., 113., 112.,
                 107., 106., 105., 104., 103., 102., 101., 100., 101., 102., 103., 104., 105., 106., 107., 108., 109., 108., 107., 106., 105., 104., 103., 102.,
                 97., 96., 95., 94., 93., 92., 91., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99., 98., 97., 96., 95., 94., 93., 92.,
                 87., 86., 85., 84., 83., 82., 81., 80., 81., 82., 83., 84., 85., 86., 87., 88., 89., 88., 87., 86., 85., 84., 83., 82.,
                 77., 76., 75., 74., 73., 72., 71., 70., 71., 72., 73., 74., 75., 76., 77., 78., 79., 78., 77., 76., 75., 74., 73., 72.],
           &mut [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           0.001, // 0.001 for order 3 => radius 5.25
           &[ 77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000, 71.000000, 70.000000, 71.000000, 72.000000, 73.000000, 74.000000, 75.000000, 76.000000, 77.000000, 78.000000, 79.000000, 78.000000, 77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000,
                67.000000, 66.000000, 65.000000, 64.000000, 63.000000, 62.000000, 61.000000, 60.000000, 61.000000, 62.000000, 63.000000, 64.000000, 65.000000, 66.000000, 67.000000, 68.000000, 69.000000, 68.000000, 67.000000, 66.000000, 65.000000, 64.000000, 63.000000, 62.000000,
                57.000000, 56.000000, 55.000000, 54.000000, 53.000000, 52.000000, 51.000000, 50.000000, 51.000000, 52.000000, 53.000000, 54.000000, 55.000000, 56.000000, 57.000000, 58.000000, 59.000000, 58.000000, 57.000000, 56.000000, 55.000000, 54.000000, 53.000000, 52.000000,
                47.000000, 46.000000, 45.000000, 44.000000, 43.000000, 42.000000, 41.000000, 40.000000, 41.000000, 42.000000, 43.000000, 44.000000, 45.000000, 46.000000, 47.000000, 48.000000, 49.000000, 48.000000, 47.000000, 46.000000, 45.000000, 44.000000, 43.000000, 42.000000,
                37.000000, 36.000000, 35.000000, 34.000000, 33.000000, 32.000000, 31.000000, 30.000000, 31.000000, 32.000000, 33.000000, 34.000000, 35.000000, 36.000000, 37.000000, 38.000000, 39.000000, 38.000000, 37.000000, 36.000000, 35.000000, 34.000000, 33.000000, 32.000000,
                27.000000, 26.000000, 25.000000, 24.000000, 23.000000, 22.000000, 21.000000, 20.000000, 21.000000, 22.000000, 23.000000, 24.000000, 25.000000, 26.000000, 27.000000, 28.000000, 29.000000, 28.000000, 27.000000, 26.000000, 25.000000, 24.000000, 23.000000, 22.000000,
                3.093084, 2.926395, 2.759706, 2.593016, 2.426327, 2.259638, 0.353195, 0.304984, 0.353128, 0.375452, 0.404703, 0.432064, 0.460056, 0.487418, 0.516666, 0.539001, 0.587104, 0.539046, 3.093084, 2.926395, 2.759706, 2.593016, 2.426327, 2.259638,
                0.203903, 0.037242, -0.129419, -0.296079, -0.462740, -0.629400, -0.128372, -0.176493, -0.128379, -0.106053, -0.076809, -0.049451, -0.021465, 0.005892, 0.035137, 0.057462, 0.105579, 0.057446, 0.203903, 0.037242, -0.129419, -0.296079, -0.462740, -0.629400,
                3.091305, 2.924637, 2.757969, 2.591300, 2.424632, 2.257964, 0.352919, 0.304714, 0.352852, 0.375173, 0.404420, 0.431778, 0.459766, 0.487125, 0.516370, 0.538701, 0.586799, 0.538746, 3.091305, 2.924637, 2.757969, 2.591300, 2.424632, 2.257964,
                4.430877, 4.264211, 4.097544, 3.930878, 3.764212, 3.597546, 0.576213, 0.527970, 0.576118, 0.598436, 0.627684, 0.655041, 0.683029, 0.710388, 0.739631, 0.761965, 0.810051, 0.762038, 4.430877, 4.264211, 4.097544, 3.930878, 3.764212, 3.597546,
                6.185188, 6.018521, 5.851854, 5.685187, 5.518521, 5.351854, 0.868638, 0.820344, 0.868506, 0.890820, 0.920069, 0.947426, 0.975414, 1.002773, 1.032015, 1.054353, 1.102426, 1.054463, 6.185188, 6.018521, 5.851854, 5.685187, 5.518521, 5.351854,
                7.828372, 7.661706, 7.495039, 7.328372, 7.161706, 6.995039, 1.142539, 1.094199, 1.142373, 1.164683, 1.193933, 1.221290, 1.249278, 1.276637, 1.305879, 1.328220, 1.376280, 1.328364, 7.828372, 7.661706, 7.495039, 7.328372, 7.161706, 6.995039,
                9.501323, 9.334656, 9.167990, 9.001323, 8.834656, 8.667990, 1.421402, 1.373013, 1.421200, 1.443508, 1.472758, 1.500115, 1.528103, 1.555463, 1.584703, 1.607048, 1.655095, 1.607227, 9.501323, 9.334656, 9.167990, 9.001323, 8.834656, 8.667990,
                11.166336, 10.999669, 10.833003, 10.666336, 10.499669, 10.333003, 1.698942, 1.650506, 1.698705, 1.721009, 1.750261, 1.777618, 1.805605, 1.832965, 1.862205, 1.884553, 1.932587, 1.884767, 11.166336, 10.999669, 10.833003, 10.666336, 10.499669, 10.333003,
                12.833333, 12.666667, 12.500000, 12.333333, 12.166667, 12.000000, 1.976812, 1.928328, 1.976541, 1.998841, 2.028094, 2.055450, 2.083438, 2.110798, 2.140037, 2.162388, 2.210410, 2.162637, 12.833333, 12.666667, 12.500000, 12.333333, 12.166667, 12.000000,
                14.500330, 14.333664, 14.166997, 14.000330, 13.833664, 13.666997, 2.254683, 2.206151, 2.254376, 2.276673, 2.305927, 2.333283, 2.361271, 2.388631, 2.417869, 2.440224, 2.488233, 2.440508, 14.500330, 14.333664, 14.166997, 14.000330, 13.833664, 13.666997,
                16.165345, 15.998678, 15.832012, 15.665345, 15.498678, 15.332012, 2.532223, 2.483644, 2.531882, 2.554175, 2.583429, 2.610786, 2.638773, 2.666134, 2.695371, 2.717729, 2.765725, 2.718048, 16.165345, 15.998678, 15.832012, 15.665345, 15.498678, 15.332012,
                17.838290, 17.671623, 17.504956, 17.338290, 17.171623, 17.004956, 2.811085, 2.762458, 2.810708, 2.832999, 2.862254, 2.889610, 2.917597, 2.944958, 2.974194, 2.996556, 3.044539, 2.996910, 17.838290, 17.671623, 17.504956, 17.338290, 17.171623, 17.004956,
                19.481497, 19.314830, 19.148163, 18.981497, 18.814830, 18.648163, 3.084989, 3.036315, 3.084579, 3.106866, 3.136122, 3.163477, 3.191465, 3.218826, 3.248061, 3.270426, 3.318397, 3.270815, 19.481497, 19.314830, 19.148163, 18.981497, 18.814830, 18.648163,
                21.235723, 21.069057, 20.902391, 20.735724, 20.569058, 20.402392, 3.377401, 3.328676, 3.376953, 3.399236, 3.428493, 3.455849, 3.483836, 3.511197, 3.540431, 3.562800, 3.610758, 3.563225, 21.235723, 21.069057, 20.902391, 20.735724, 20.569058, 20.402392,
                22.575611, 22.408943, 22.242275, 22.075606, 21.908938, 21.742270, 3.600744, 3.551980, 3.600268, 3.622549, 3.651807, 3.679162, 3.707150, 3.734512, 3.763746, 3.786117, 3.834065, 3.786571, 22.575611, 22.408943, 22.242275, 22.075606, 21.908938, 21.742270,
                25.461832, 25.295172, 25.128511, 24.961850, 24.795190, 24.628529, 4.081853, 4.033009, 4.081317, 4.103591, 4.132849, 4.160203, 4.188189, 4.215550, 4.244781, 4.267158, 4.315081, 4.267671, 25.461832, 25.295172, 25.128511, 24.961850, 24.795190, 24.628529,
                22.577060, 22.410370, 22.243681, 22.076992, 21.910303, 21.743613, 3.600965, 3.552195, 3.600489, 3.622773, 3.652034, 3.679393, 3.707385, 3.734750, 3.763987, 3.786362, 3.834315, 3.786815, 22.577060, 22.410370, 22.243681, 22.076992, 21.910303, 21.743613,
                127.000000, 126.000000, 125.000000, 124.000000, 123.000000, 122.000000, 121.000000, 120.000000, 121.000000, 122.000000, 123.000000, 124.000000, 125.000000, 126.000000, 127.000000, 128.000000, 129.000000, 128.000000, 127.000000, 126.000000, 125.000000, 124.000000, 123.000000, 122.000000,
                117.000000, 116.000000, 115.000000, 114.000000, 113.000000, 112.000000, 111.000000, 110.000000, 111.000000, 112.000000, 113.000000, 114.000000, 115.000000, 116.000000, 117.000000, 118.000000, 119.000000, 118.000000, 117.000000, 116.000000, 115.000000, 114.000000, 113.000000, 112.000000,
                107.000000, 106.000000, 105.000000, 104.000000, 103.000000, 102.000000, 101.000000, 100.000000, 101.000000, 102.000000, 103.000000, 104.000000, 105.000000, 106.000000, 107.000000, 108.000000, 109.000000, 108.000000, 107.000000, 106.000000, 105.000000, 104.000000, 103.000000, 102.000000,
                97.000000, 96.000000, 95.000000, 94.000000, 93.000000, 92.000000, 91.000000, 90.000000, 91.000000, 92.000000, 93.000000, 94.000000, 95.000000, 96.000000, 97.000000, 98.000000, 99.000000, 98.000000, 97.000000, 96.000000, 95.000000, 94.000000, 93.000000, 92.000000,
                87.000000, 86.000000, 85.000000, 84.000000, 83.000000, 82.000000, 81.000000, 80.000000, 81.000000, 82.000000, 83.000000, 84.000000, 85.000000, 86.000000, 87.000000, 88.000000, 89.000000, 88.000000, 87.000000, 86.000000, 85.000000, 84.000000, 83.000000, 82.000000,
                77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000, 71.000000, 70.000000, 71.000000, 72.000000, 73.000000, 74.000000, 75.000000, 76.000000, 77.000000, 78.000000, 79.000000, 78.000000, 77.000000, 76.000000, 75.000000, 74.000000, 73.000000, 72.000000],
           & [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           true)]
    fn test_array1_bspline_prefiltering_ext_gene(
        #[case] n: usize,
        #[case] epsilon: f64,
        #[case] trunc_idx: Option<& [usize]>,
        #[case] height: usize,
        #[case] width: usize,
        #[case] mut ima_in: &mut [f64],       
        #[case] mut ima_mask_in: &mut [u8],
        #[case] mask_influence_threshold: f64,
        #[case] expected: &[f64],
        #[case] expected_mask: &[u8],
        #[case] test_mask: bool,
    )
    {
        let mut ima_in_view = GxArrayViewMut::new(&mut ima_in, 1, height, width);
        let mut ima_mask_in_view = Some(GxArrayViewMut::new(&mut ima_mask_in, 1, height, width));

        if test_mask {            
            let _ = array1_bspline_prefiltering_ext_gene(n, epsilon, trunc_idx, &mut ima_in_view, ima_mask_in_view.as_mut(), Some(mask_influence_threshold));
            assert_eq!(&ima_mask_in, &expected_mask);
        } else {
            let _ = array1_bspline_prefiltering_ext_gene(n, epsilon, trunc_idx, &mut ima_in_view, None, None);
        }
        assert!(approx_eq(&ima_in, &expected, 1e-6));
    }
}
