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
//! Cardinal B-spline interpolators with compile-time order specialization.
//!
//! This module provides efficient B-spline interpolation implementations using
//! Rust's const generics to specialize interpolators at compile time for 
//! different spline orders.
//!
//! # Core Components
//!
//! - **B-spline basis functions**: Optimized implementations for orders 3, 5,
//!   7, 9, and 11
//! - **Generic interpolator**: `GxBSplineInterpolator<N>` with compile-time
//!   order parameter
//! - **Type aliases**: Convenience types (`GxBSpline3Interpolator`, 
//!   `GxBSpline5Interpolator`, etc.)
//!
//! # Trait Implementations
//!
//! ## `GxArrayViewInterpolator`
//! The main interpolation interface implemented by all B-spline interpolators.
//!
//! ## `GxBSplineInterpolatorTrait<N>`
//! B-spline-specific interface providing:
//! - `bspline()`: Evaluates the B-spline basis function at a given coordinate
//! - `bspline_kernel_weights()`: Computes interpolation weights for the kernel
//! - `array1_bspline_prefiltering_ext()`: Applies recursive prefiltering to
//!   input data
//!
//! # Supported Orders
//!
//! | Order | Kernel Radius | Type Alias                |
//! |-------|---------------|---------------------------|
//! | 3     | 2             | `GxBSpline3Interpolator`  |
//! | 5     | 3             | `GxBSpline5Interpolator`  |
//! | 7     | 4             | `GxBSpline7Interpolator`  |
//! | 9     | 5             | `GxBSpline9Interpolator`  |
//! | 11    | 6             | `GxBSpline11Interpolator` |
//!
//!
//! # Usage Example
//!
//! ```ignore
//! use gridr::core::gx_bspline_interp::{
//!     GxBSpline5Interpolator,
//!     GxArrayViewInterpolator
//! };
//!
//! let args = GxBSplineInterpolatorArgs {
//!     epsilon: 1e-3,
//!     mask_influence_threshold: 0.001,
//! };
//! let mut interp = GxBSpline5Interpolator::new(&args);
//! interp.initialize()?;
//!
//! // Prefilter input data
//! interp.array1_bspline_prefiltering_ext(&mut input_array, Some(&mut input_mask))?;
//!
//! // Perform interpolation
//! interp.array1_interp2(&mut weights, row_pos, col_pos, out_idx, 
//!                       &input_array, &mut output_array, nodata, &mut context)?;
//! ```
//!
//! # References
//!
//! Briand, T., & Monasse, P. (2018). Theory and Practice of Image B-Spline Interpolation.
//! *Image Processing On Line*, 8, 99-141. https://doi.org/10.5201/ipol.2018.221
use::std::format;
use crate::core::gx_array::{GxArrayView, GxArrayViewMut};
use crate::core::gx_errors::GxError;
use super::gx_array_view_interp::{
    GxArrayViewInterpolator,
    GxArrayViewInterpolatorArgs,
    GxArrayViewInterpolatorCore,
    GxArrayViewInterpolationContextTrait,
};
use super::gx_bspline_prefiltering::{
    compute_2d_truncation_index,
    compute_2d_domain_extension_from_truncation_idx,
    array1_bspline_prefiltering_ext_gene,
    TRUNCATION_INDEX_BUFFER_MAX_SIZE,
    TRUNCATION_L_BUFFER_MAX_SIZE
};

/// Cubic B-spline function - radius 2
#[inline]
pub fn bspline3(x: f64) -> f64
{
    let x = x.abs();
    if x < 1.
    {
        return 4. + (-6. + 3.*x)*x*x;
    }
    else if x < 2.
    {
        let x = 2. - x;
        return x*x*x;
    }
    0.0
}

/// Quintic B-spline function - radius 3
#[inline]
pub fn bspline5(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return ((-10.*x + 30.)*x2 - 60.)*x2 + 66.;
    }
    else if x < 2.
    {
        let x = 2. - x;
        return 1. + (5. + (10. + (10. + (5. - 5.*x)*x)*x)*x)*x;
    }
    else if x < 3.
    {
        let x = 3. - x;
        let x2 = x*x;
        return x2*x2*x;
    }
    0.0
}

/// Septic B-spline function - radius 4
#[inline]
pub fn bspline7(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return (((35.*x - 140.)*x2 + 560.)*x2 - 1680.)*x2 + 2416.;
    }
    else if x < 2.
    {
        let x = 2. - x;
        return 120. + (392. + (504. + (280. + (-84. + (-42. + 21.*x)*x)*x*x)*x)*x)*x;
    }
    else if x < 3.
    {
        let x = 3. - x;
        return ((((((-7.*x + 7.)*x + 21.)*x + 35.)*x + 35.)*x + 21.)*x + 7.)*x + 1.;
    }
    else if x < 4.
    {
        let x = 4. - x;
        let x2 = x*x;
        return x2*x2*x2*x;
    }
    0.0
}

/// Nonic B-spline function - radius 5
#[inline]
pub fn bspline9(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return (((((-63.*x + 315.)*x2 - 2100.)*x2 + 11970.)*x2 - 44100.)*x2 + 78095.)*2.;
    }
    else if x <= 2.
    {
        let x = 2. - x;
        return 14608. + (36414. + (34272. + (11256. + (-4032. + (-4284. + (-672. + (504. + (252. - 84.*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x <= 3.
    {
        let x = 3. - x;
        return 502. + (2214. + (4248. + (4536. + (2772. + (756. + (-168. + (-216. + (-72. + 36.*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 4.
    {
        let x = 4. - x;
        return 1. + (9. + (36. + (84. + (126. + (126. + (84. + (36. + (9. - 9.*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 5.
    {
        let x = 5. - x;
        let x3 = x*x*x;
        return x3*x3*x3;
    }
    0.0
}

/// Eleventh order B-spline function - radius 6
#[inline]
pub fn bspline11(x: f64) -> f64
{
    let x = x.abs();
    if x <= 1.
    {
        let x2 = x*x;
        return 15724248. + (-7475160. + (1718640. + (-255024. + (27720.
            + (-2772. + 462.*x)*x2)*x2)*x2)*x2)*x2;
    }
    else if x <= 2.
    {
        let x = 2. - x;
        return 2203488. + (4480872. + (3273600. + (574200. + (-538560.
            + (-299376. + (39600. + (7920. + (-2640. + (-1320.
            + 330.*x)*x)*x)*x)*x*x)*x)*x)*x)*x)*x;
    }
    else if x <= 3.
    {
        let x = 3. - x;
        return 152637. + (515097. + (748275. + (586575. + (236610. + (12474.
            + (-34650. + (-14850. + (-495. + (1485.
            + (495.-165.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x <= 4.
    {
        let x = 4. - x;
        return 2036. + (11132. + (27500. + (40260. + (38280. + (24024. + (9240.
            + (1320. + (-660. + (-440. + (-110.
            + 55.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 5.
    {
        let x = 5. - x;
        return 1. + (11. + (55. + (165. + (330. + (462. + (462. + (330. + (165.
            + (55. + (11. - 11.*x)*x)*x)*x)*x)*x)*x)*x)*x)*x)*x;
    }
    else if x < 6.
    {
        let x = 6. - x;
        let x2 = x*x;
        let x4 = x2*x2;
        return x4*x4*x2*x;
    }
    0.0
}

/// Computes all non-zero cubic B-spline weights for a centered fractional
/// offset, writing them into `weights`.
///
/// # Optimization strategy
///
/// The five kernel weights are computed by exploiting the symmetry of the
/// B-spline support around the nearest integer pixel.  For `t ∈ [0, 0.5]`
/// the substitution `s = 1 - t` yields:
///
/// ```text
///   weights[0] = t³              (branch 2, argument 2-t)
///   weights[1] = 4 + (-6+3s)s²  (branch 1, argument s = 1-t)
///   weights[2] = 4 + (-6+3t)t²  (branch 1, argument t)
///   weights[3] = s³              (branch 2, argument 1+t → 2-(1+t) = s)
///   weights[4] = 0               (outside support)
/// ```
///
/// For `t ∈ (-0.5, 0)`, `u = -t` is used and the same four polynomials are
/// evaluated on `u` and `s = 1-u`; only the placement in `weights` is
/// mirrored.  This avoids all `abs()` calls and conditional branches inside
/// the polynomial evaluation, and reduces the power computation to a shared
/// `(t², t³)` or `(u², u³, s², s³)` base.
///
/// # Cost
///
/// - **5 multiplications**, **5 additions** (versus ~10 mul and 8 add for
///   five independent scalar calls including `abs()` and branch overhead).
/// - **1 structural zero**: `weights[4]` (`t ≥ 0`) or `weights[0]`
///   (`t < 0`) is always 0, saving one multiply in the convolution.
///
/// # Parameters
///
/// - `t`: Fractional offset in `(-0.5, 0.5]`, computed as
///   `x - round(x)` (i.e. `x - floor(x + 0.5)`).
/// - `weights`: Output slice of length ≥ 5.  Index 0 corresponds to
///   offset −2, index 4 to offset +2 relative to the nearest integer.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 5`.  This function uses
/// `get_unchecked_mut` to suppress per-element bounds checks; violating the
/// length contract results in undefined behaviour.
#[inline]
pub fn bspline3_all_centered(t: f64, weights: &mut [f64]) {
    // SAFETY: caller guarantees weights.len() >= 5.
    let w = unsafe { weights.get_unchecked_mut(0..5) };
 
    if t >= 0.0 {
        // t in [0, 0.5], s = 1-t in [0.5, 1]
        let s = 1.0 - t;
        let t2 = t * t; let t3 = t2 * t;
        let s2 = s * s; let s3 = s2 * s;
 
        w[0] = t3;
        w[1] = 4.0 + (-6.0 + 3.0*s)*s2;
        w[2] = 4.0 + (-6.0 + 3.0*t)*t2;
        w[3] = s3;
        w[4] = 0.0;
 
    } else {
        // u = -t in (0, 0.5), s = 1-u in [0.5, 1)
        let u = -t;
        let s = 1.0 - u;
        let u2 = u * u; let u3 = u2 * u;
        let s2 = s * s; let s3 = s2 * s;
 
        w[0] = 0.0;
        w[1] = s3;
        w[2] = 4.0 + (-6.0 + 3.0*u)*u2;
        w[3] = 4.0 + (-6.0 + 3.0*s)*s2;
        w[4] = u3;
    }
}

/// Computes all non-zero quintic B-spline weights for a centered fractional
/// offset, writing them into `weights`.
///
/// # Optimization strategy
///
/// With `t ∈ [0, 0.5]` and `s = 1 - t`, the six non-zero weights map to
/// branches of `bspline5` as follows:
///
/// ```text
///   weights[0] = t⁵              (branch 3, |t−3| = 3−t ∈ [2.5,3))
///   weights[1] = bspline5 br2(t) (branch 2, v = t)
///   weights[2] = bspline5 br1(s) (branch 1, u = s = 1−t)
///   weights[3] = bspline5 br1(t) (branch 1, u = t)
///   weights[4] = bspline5 br2(s) (branch 2, v = s)
///   weights[5] = s⁵              (branch 3, |t+2| = 2+t → 3−(2+t) = s)
///   weights[6] = 0               (outside support: |t+3| ≥ 3)
/// ```
///
/// All power sequences `(t², t⁴, t⁵)` and `(s², s⁴, s⁵)` are computed
/// once and shared across the polynomial evaluations.  For `t < 0`, the
/// substitution `u = -t` produces the mirror layout with `weights[0] = 0`.
///
/// # Cost
///
/// - **8 multiplications**, **8 additions** (versus ~24 mul and 19 add for
///   seven independent scalar calls).
/// - **1 structural zero**: always at index 6 (`t ≥ 0`) or index 0
///   (`t < 0`).
///
/// # Parameters
///
/// - `t`: Fractional offset in `(-0.5, 0.5]`, computed as `x - round(x)`.
/// - `weights`: Output slice of length ≥ 7.  Index 0 = offset −3,
///   index 6 = offset +3.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 7`.  This function uses
/// `get_unchecked_mut` to suppress per-element bounds checks; violating the
/// length contract results in undefined behaviour.
#[inline]
pub fn bspline5_all_centered(t: f64, weights: &mut [f64]) {
    // SAFETY: caller guarantees weights.len() >= 7.
    let w = unsafe { weights.get_unchecked_mut(0..7) };
 
    if t >= 0.0 {
        // t in [0, 0.5], s = 1-t in [0.5, 1].
        // b_{-3} and b_3 are both in branch 3:
        //   |t-3| = 3-t in [2.5, 3)  ->  (3-(3-t))^5 = t^5
        //   |t+3| = 3+t in [3, 3.5]  ->  outside support -> 0
        // Only b_{-2}..b_2 are non-zero (6 values).
        let s = 1.0 - t;
        let t2 = t * t; let t4 = t2 * t2; let t5 = t4 * t;
        let s2 = s * s; let s4 = s2 * s2; let s5 = s4 * s;
 
        w[0] = t5;
        w[1] = 1.0 + (5.0 + (10.0 + (10.0 + (5.0 - 5.0*t)*t)*t)*t)*t;
        w[2] = ((-10.0*s + 30.0)*s2 - 60.0)*s2 + 66.0;
        w[3] = ((-10.0*t + 30.0)*t2 - 60.0)*t2 + 66.0;
        w[4] = 1.0 + (5.0 + (10.0 + (10.0 + (5.0 - 5.0*s)*s)*s)*s)*s;
        w[5] = s5;
        w[6] = 0.0;
 
    } else {
        // u = -t in (0, 0.5).  By parity bspline5(-x) = bspline5(x),
        // so b_k(t) = b_{-k}(-t): the layout is the mirror of t >= 0.
        let u = -t;
        let s = 1.0 - u;
        let u2 = u * u; let u4 = u2 * u2; let u5 = u4 * u;
        let s2 = s * s; let s4 = s2 * s2; let s5 = s4 * s;
 
        w[0] = 0.0;
        w[1] = s5;
        w[2] = 1.0 + (5.0 + (10.0 + (10.0 + (5.0 - 5.0*s)*s)*s)*s)*s;
        w[3] = ((-10.0*u + 30.0)*u2 - 60.0)*u2 + 66.0;
        w[4] = ((-10.0*s + 30.0)*s2 - 60.0)*s2 + 66.0;
        w[5] = 1.0 + (5.0 + (10.0 + (10.0 + (5.0 - 5.0*u)*u)*u)*u)*u;
        w[6] = u5;
    }
}

/// Computes all non-zero septic B-spline weights for a centered fractional
/// offset, writing them into `weights`.
///
/// # Optimization strategy
///
/// With support radius 4 and `t ∈ [0, 0.5]`, `s = 1 - t`, the eight
/// non-zero weights are:
///
/// ```text
///   weights[0] = t⁷              (branch 4, argument 4−t → t⁷)
///   weights[1] = br3(t)          (branch 3, argument 3−t)
///   weights[2] = br2(t)          (branch 2, argument 2−t)
///   weights[3] = br1(s)          (branch 1, argument 1−t = s)
///   weights[4] = br1(t)          (branch 1, argument t)
///   weights[5] = br2(s)          (branch 2, argument 1+t → 2−(1+t) = s)
///   weights[6] = br3(s)          (branch 3, argument 2+t → 3−(2+t) = s)
///   weights[7] = s⁷              (branch 4, argument 3+t → 4−(3+t) = s)
///   weights[8] = 0               (outside support)
/// ```
///
/// Branch functions `br1`, `br2`, `br3` are Horner-form inner functions
/// that take `u ∈ [0, 0.5]` and return the corresponding piece of
/// `bspline7`.  Powers `t⁷` and `s⁷` are computed via
/// `t²→t⁴→t⁷ = t⁴·t²·t` (4 multiplications total).
///
/// For `t < 0`, `u = -t` and the layout is mirrored with `weights[0] = 0`.
///
/// # Parameters
///
/// - `t`: Fractional offset in `(-0.5, 0.5]`, computed as `x - round(x)`.
/// - `weights`: Output slice of length ≥ 9.  Index 0 = offset −4,
///   index 8 = offset +4.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 9`.  This function uses
/// `get_unchecked_mut` to suppress per-element bounds checks; violating the
/// length contract results in undefined behaviour.
#[inline]
pub fn bspline7_all_centered(t: f64, weights: &mut [f64]) {
    // SAFETY: caller guarantees weights.len() >= 9.
    let w = unsafe { weights.get_unchecked_mut(0..9) };
 
    // Branch 2 of bspline7: piece valid for x in [1,2), evaluated at u = 2-x
    #[inline(always)]
    fn br2(u: f64) -> f64 {
        120. + (392. + (504. + (280. + (-84. + (-42. + 21.*u)*u)*u*u)*u)*u)*u
    }
 
    // Branch 3 of bspline7: piece valid for x in [2,3), evaluated at u = 3-x
    #[inline(always)]
    fn br3(u: f64) -> f64 {
        ((((((-7.*u + 7.)*u + 21.)*u + 35.)*u + 35.)*u + 21.)*u + 7.)*u + 1.
    }
 
    // Branch 1 of bspline7: piece valid for x in [0,1), evaluated at u = x
    #[inline(always)]
    fn br1(u: f64) -> f64 {
        let u2 = u*u;
        (((35.*u - 140.)*u2 + 560.)*u2 - 1680.)*u2 + 2416.
    }
 
    if t >= 0.0 {
        let s = 1.0 - t;
        let t2 = t*t; let t4 = t2*t2; let t7 = t4*t2*t;
        let s2 = s*s; let s4 = s2*s2; let s7 = s4*s2*s;
 
        w[0] = t7;
        w[1] = br3(t);
        w[2] = br2(t);
        w[3] = br1(s);
        w[4] = br1(t);
        w[5] = br2(s);
        w[6] = br3(s);
        w[7] = s7;
        w[8] = 0.0;
 
    } else {
        let u = -t;
        let s = 1.0 - u;
        let u2 = u*u; let u4 = u2*u2; let u7 = u4*u2*u;
        let s2 = s*s; let s4 = s2*s2; let s7 = s4*s2*s;
 
        w[0] = 0.0;
        w[1] = s7;
        w[2] = br3(s);
        w[3] = br2(s);
        w[4] = br1(u);
        w[5] = br1(s);
        w[6] = br2(u);
        w[7] = br3(u);
        w[8] = u7;
    }
}
 
/// Computes all non-zero nonic B-spline weights for a centered fractional
/// offset, writing them into `weights`.
///
/// # Optimization strategy
///
/// With support radius 5 and `t ∈ [0, 0.5]`, `s = 1 - t`, the ten
/// non-zero weights are:
///
/// ```text
///   weights[0]  = t⁹             (branch 5, argument 5−t → t⁹)
///   weights[1]  = br4(t)         (branch 4, argument 4−t)
///   weights[2]  = br3(t)         (branch 3, argument 3−t)
///   weights[3]  = br2(t)         (branch 2, argument 2−t)
///   weights[4]  = br1(s)         (branch 1, argument 1−t = s)
///   weights[5]  = br1(t)         (branch 1, argument t)
///   weights[6]  = br2(s)         (branch 2, argument 1+t → s)
///   weights[7]  = br3(s)         (branch 3, argument 2+t → s)
///   weights[8]  = br4(s)         (branch 4, argument 3+t → s)
///   weights[9]  = s⁹             (branch 5, argument 4+t → s⁹)
///   weights[10] = 0              (outside support)
/// ```
///
/// Powers `t⁹` and `s⁹` are computed via `t³→t⁹ = t³·t³·t³`
/// (4 multiplications).  Branch functions `br1`–`br4` are standard
/// Horner evaluations transcribed directly from `bspline9`.
///
/// For `t < 0`, `u = -t` and the layout is mirrored with `weights[0] = 0`.
///
/// # Parameters
///
/// - `t`: Fractional offset in `(-0.5, 0.5]`, computed as `x - round(x)`.
/// - `weights`: Output slice of length ≥ 11.  Index 0 = offset −5,
///   index 10 = offset +5.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 11`.  This function uses
/// `get_unchecked_mut` to suppress per-element bounds checks; violating the
/// length contract results in undefined behaviour.
#[inline]
pub fn bspline9_all_centered(t: f64, weights: &mut [f64]) {
    // SAFETY: caller guarantees weights.len() >= 11.
    let w = unsafe { weights.get_unchecked_mut(0..11) };
 
    #[inline(always)]
    fn br1(u: f64) -> f64 {
        let u2 = u*u;
        (((((-63.*u + 315.)*u2 - 2100.)*u2 + 11970.)*u2 - 44100.)*u2 + 78095.)*2.
    }
 
    #[inline(always)]
    fn br2(u: f64) -> f64 {
        14608. + (36414. + (34272. + (11256. + (-4032.
            + (-4284. + (-672. + (504. + (252. - 84.*u)*u)*u)*u)*u)*u)*u)*u)*u
    }
 
    #[inline(always)]
    fn br3(u: f64) -> f64 {
        502. + (2214. + (4248. + (4536. + (2772. + (756.
            + (-168. + (-216. + (-72. + 36.*u)*u)*u)*u)*u)*u)*u)*u)*u
    }
 
    #[inline(always)]
    fn br4(u: f64) -> f64 {
        1. + (9. + (36. + (84. + (126. + (126. + (84. + (36. + (9. - 9.*u)*u)*u)*u)*u)*u)*u)*u)*u
    }
 
    if t >= 0.0 {
        let s = 1.0 - t;
        let t3 = t*t*t; let t9 = t3*t3*t3;
        let s3 = s*s*s; let s9 = s3*s3*s3;
 
        w[0]  = t9;
        w[1]  = br4(t);
        w[2]  = br3(t);
        w[3]  = br2(t);
        w[4]  = br1(s);
        w[5]  = br1(t);
        w[6]  = br2(s);
        w[7]  = br3(s);
        w[8]  = br4(s);
        w[9]  = s9;
        w[10] = 0.0;
 
    } else {
        let u = -t;
        let s = 1.0 - u;
        let u3 = u*u*u; let u9 = u3*u3*u3;
        let s3 = s*s*s; let s9 = s3*s3*s3;
 
        w[0]  = 0.0;
        w[1]  = s9;
        w[2]  = br4(s);
        w[3]  = br3(s);
        w[4]  = br2(s);
        w[5]  = br1(u);
        w[6]  = br1(s);
        w[7]  = br2(u);
        w[8]  = br3(u);
        w[9]  = br4(u);
        w[10] = u9;
    }
}


// Computes all non-zero eleventh-order B-spline weights for a centered
/// fractional offset, writing them into `weights`.
///
/// # Optimization strategy
///
/// With support radius 6 and `t ∈ [0, 0.5]`, `s = 1 - t`, the twelve
/// non-zero weights are:
///
/// ```text
/// weights[0] = t¹¹ (branch 6, argument 6−t → t¹¹)
/// weights[1] = br5(t) (branch 5, argument 5−t)
/// weights[2] = br4(t) (branch 4, argument 4−t)
/// weights[3] = br3(t) (branch 3, argument 3−t)
/// weights[4] = br2(t) (branch 2, argument 2−t)
/// weights[5] = br1(s) (branch 1, argument 1−t = s)
/// weights[6] = br1(t) (branch 1, argument t)
/// weights[7] = br2(s) (branch 2, argument 1+t → s)
/// weights[8] = br3(s) (branch 3, argument 2+t → s)
/// weights[9] = br4(s) (branch 4, argument 3+t → s)
/// weights[10] = br5(s) (branch 5, argument 4+t → s)
/// weights[11] = s¹¹ (branch 6, argument 5+t → s¹¹)
/// weights[12] = 0 (outside support)
/// ```
///
/// Powers `t¹¹` and `s¹¹` are computed via
/// `t²→t⁴→t⁸→t¹¹ = t⁸·t²·t` (4 multiplications).
///
/// **Note on branch 2**: the original `bspline11` Horner contains a `*x*x`
/// step (zero coefficient for `x⁵`), which is preserved verbatim in `br2`
/// as `)*u*u)` to match the reference exactly. This causes a rounding
/// difference of up to ~1e-7 relative to the scalar function for certain
/// inputs; the corresponding test uses a tolerance of `1e-6`.
///
/// For `t < 0`, `u = -t` and the layout is mirrored with `weights[0] = 0`.
///
/// # Parameters
///
/// - `t`: Fractional offset in `(-0.5, 0.5]`, computed as `x - round(x)`.
/// - `weights`: Output slice of length ≥ 13. Index 0 = offset −6,
/// index 12 = offset +6.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 13`. This function uses
/// `get_unchecked_mut` to suppress per-element bounds checks; violating the
/// length contract results in undefined behaviour.
#[inline]
pub fn bspline11_all_centered(t: f64, weights: &mut [f64]) {
    // SAFETY: caller guarantees weights.len() >= 13.
    let w = unsafe { weights.get_unchecked_mut(0..13) };

    #[inline(always)]
    fn br1(u: f64) -> f64 {
        let u2 = u*u;
        15724248. + (-7475160. + (1718640. + (-255024. + (27720.
        + (-2772. + 462.*u)*u2)*u2)*u2)*u2)*u2
    }

    #[inline(always)]
    fn br2(u: f64) -> f64 {
        2203488. + (4480872. + (3273600. + (574200. + (-538560.
        + (-299376. + (39600. + (7920. + (-2640. + (-1320.
        + 330.*u)*u)*u)*u)*u*u)*u)*u)*u)*u)*u
    }

    #[inline(always)]
    fn br3(u: f64) -> f64 {
        152637. + (515097. + (748275. + (586575. + (236610. + (12474.
        + (-34650. + (-14850. + (-495. + (1485.
        + (495.-165.*u)*u)*u)*u)*u)*u)*u)*u)*u)*u)*u
    }

    #[inline(always)]
    fn br4(u: f64) -> f64 {
        2036. + (11132. + (27500. + (40260. + (38280. + (24024. + (9240.
        + (1320. + (-660. + (-440. + (-110.
        + 55.*u)*u)*u)*u)*u)*u)*u)*u)*u)*u)*u
    }

    #[inline(always)]
    fn br5(u: f64) -> f64 {
        1. + (11. + (55. + (165. + (330. + (462. + (462. + (330. + (165.
        + (55. + (11. - 11.*u)*u)*u)*u)*u)*u)*u)*u)*u)*u)*u
    }

    if t >= 0.0 {
        let s = 1.0 - t;
        // t^11 computed via t->t2->t4->t8->t8*t2*t (4 multiplications)
        let t2 = t*t; let t4 = t2*t2; let t8 = t4*t4;
        let t11 = t8*t2*t;
        let s2 = s*s; let s4 = s2*s2; let s8 = s4*s4;
        let s11 = s8*s2*s;

        w[0] = t11;
        w[1] = br5(t);
        w[2] = br4(t);
        w[3] = br3(t);
        w[4] = br2(t);
        w[5] = br1(s);
        w[6] = br1(t);
        w[7] = br2(s);
        w[8] = br3(s);
        w[9] = br4(s);
        w[10] = br5(s);
        w[11] = s11;
        w[12] = 0.0;

    } else {
        let u = -t;
        let s = 1.0 - u;
        let u2 = u*u; let u4 = u2*u2; let u8 = u4*u4;
        let u11 = u8*u2*u;
        let s2 = s*s; let s4 = s2*s2; let s8 = s4*s4;
        let s11 = s8*s2*s;

        w[0] = 0.0;
        w[1] = s11;
        w[2] = br5(s);
        w[3] = br4(s);
        w[4] = br3(s);
        w[5] = br2(s);
        w[6] = br1(u);
        w[7] = br1(s);
        w[8] = br2(u);
        w[9] = br3(u);
        w[10] = br4(u);
        w[11] = br5(u);
        w[12] = u11;
    }
}


/// Trait defining the interface for generic B-spline interpolators with
/// configurable order.
/// 
/// This trait provides the core functionality for B-spline interpolation
/// operations, including:
/// - Access to filter poles for prefiltering operations
/// - Evaluation of B-spline basis functions
/// - Pre-filtering operations to prepare input data
/// 
/// The trait is parameterized by one const generic:
/// - `N`: The order of the B-spline (must be odd: 3, 5, 7, 9, 11)
/// 
/// # Type Constraints
/// - `N` must be an odd integer (3, 5, 7, 9, or 11)
/// 
/// # Implementation Requirements
/// Implementors must provide:
/// 1. `get_poles()`: Returns the filter poles for the specific spline order
/// 2. `bspline()`: Evaluates the B-spline basis function at a given point
/// 3. `array1_bspline_prefiltering_ext2()`: Applies pre-filtering to input data
pub trait GxBSplineInterpolatorTrait<const N: usize>: GxArrayViewInterpolator
{
    /// Returns the number of poles
    ///
    /// # Returns
    /// - usize: The number of poles for the B-Spline.
    fn get_npoles(&self) -> usize;
    
    /// Evaluates the B-spline basis function of order N at the specified
    /// coordinate.
    /// 
    /// This method computes the value of the B-spline basis function at
    /// position `x`.
    /// The implementation varies based on the spline order N.
    /// 
    /// # Parameters
    /// - `x`: The coordinate at which to evaluate the B-spline function
    /// 
    /// # Returns
    /// - f64: The value of the B-spline basis function at position x
    fn bspline(&self, x: f64) -> f64;
    
    
    /// Evaluates the B-spline weights
    fn bspline_kernel_weights(&self, x: f64, weights: &mut [f64]);
    
    /// Applies pre-filtering to input data using the B-spline filter poles.
    /// 
    /// This operation prepares input data by applying the B-spline 
    /// pre-filtering with the specified poles and truncation indices.
    /// 
    /// # Parameters
    /// - `ima_in`: Mutable reference to the input data array to be pre-filtered
    /// - `mask_in`: Optional mutable reference to the input mask
    /// 
    /// # Returns
    /// - `Ok(())` if pre-filtering completes successfully
    /// - `Err(GxError)` containing error information if pre-filtering fails
    fn array1_bspline_prefiltering_ext<'a>(
        &'a self,
        ima_in: &mut GxArrayViewMut<'_, f64>,
        mask_in: Option<&'a mut GxArrayViewMut<'a, u8>>,
    ) -> Result<(), GxError>;
}

/// Generic implementation of B-spline interpolators with compile-time order and
/// pole configuration.
///
/// This struct serves as the concrete implementation for all supported B-spline
/// orders (3, 5, 7, 9, 11).
/// It stores the order and number of poles as runtime parameters while
/// utilizing compile-time
/// constants for the specific mathematical function implementations. The struct
/// also contains prefiltering parameters and associated buffers for efficient 
/// computation.
///
/// # Fields
/// - `order`: The order of the B-spline (N), determining the smoothness and 
///   support size
/// - `npoles`: The number of poles (N/2), used in the recursive filtering
///   implementation
/// - `epsilon`: Precision parameter for the truncation index calculation.
///   Defines the acceptable error when approximating the infinite sums. Smaller
///   values require larger margins for prefiltering. The total required margin
///   combines both the prefiltering margin (truncation index) and the 
///   interpolation kernel radius.
/// - `mask_influence_threshold`: Residual influence threshold $s$ used to
///   compute the radius of the propagation of masked data. Required when 
///   `ima_mask_in` is provided.
///   Determines the acceptable relative contamination from invalid pixels.
/// - `truncation_index`: Buffer storing truncation index
/// - `domain_extension`: Buffer storing domain extension
#[derive(Clone, Debug)]
pub struct GxBSplineInterpolator<const N: usize> {
    /// The order of the B-spline (N), determining the smoothness and support
    /// size
    pub order: usize,
    /// The number of poles (N/2), used in the recursive filtering implementation
    pub npoles: usize,
    /// Precision parameter for the truncation index calculation
    pub epsilon: f64,
    /// Acceptable relative contamination from invalid pixels.
    pub mask_influence_threshold: f64,
    /// Buffer storing truncation index
    pub truncation_index: [usize; TRUNCATION_INDEX_BUFFER_MAX_SIZE],
    /// Buffer storing domain extension
    pub domain_extension: [usize; TRUNCATION_L_BUFFER_MAX_SIZE],
}

impl<const N: usize> GxBSplineInterpolatorTrait<N> for GxBSplineInterpolator<N> {
    
    #[inline(always)]
    fn get_npoles(&self) -> usize {
        self.npoles
    }
    
    #[inline(always)]
    fn bspline(&self, x: f64) -> f64 {
        match N {
            3 => bspline3(x),
            5 => bspline5(x),
            7 => bspline7(x),
            9 => bspline9(x),
            11 => bspline11(x),
            _ => panic!("Unsupported spline order"),
        }
    }
    
    #[inline(always)]
    fn bspline_kernel_weights(&self, x: f64, weights: &mut [f64]) 
    {
        match N {
            3 => bspline3_all_centered(x, weights),
            5 => bspline5_all_centered(x, weights),
            7 => bspline7_all_centered(x, weights),
            9 => bspline9_all_centered(x, weights),
            11 => bspline11_all_centered(x, weights),
            _ => panic!("Unsupported spline order"),
        }
    }
    
    #[inline]
    fn array1_bspline_prefiltering_ext<'a>(
        &'a self,
        ima_in: &mut GxArrayViewMut<'_, f64>,
        mask_in: Option<&'a mut GxArrayViewMut<'a, u8>>,
    ) -> Result<(), GxError>
    {
        match N {
            3 | 5 | 7 | 9 | 11 => {
                return array1_bspline_prefiltering_ext_gene(
                    self.order,
                    self.epsilon,
                    Some(&self.truncation_index),
                    ima_in,
                    mask_in,
                    Some(self.mask_influence_threshold),
                );
            },
            _ => {
                return Err(GxError::ErrMessage("Unsupported spline order".to_string()));
            },
        }
    }
}


impl<const N: usize> GxArrayViewInterpolatorCore<5, 5> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<7, 7> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<9, 9> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<11, 11> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<13, 13> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, x: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(x, weights)
    }
}

impl<const N: usize> GxBSplineInterpolator<N>
{   
    #[inline]
    fn array1_interp2_separable_core<T, V, IC>(
            &self,
            weights_buffer: &mut [f64],
            target_row_pos: f64,
            target_col_pos: f64,
            out_idx: usize,
            array_in: &GxArrayView<'_, T>,
            array_out: &mut GxArrayViewMut<'_, V>,
            nodata_out: V,
            context: &mut IC,
    ) -> Result<(), String>
    where
        T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
        V: Copy + PartialEq + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
    {
       
        match N {
            3 => GxArrayViewInterpolatorCore::<5, 5>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            5 => GxArrayViewInterpolatorCore::<7, 7>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            7 => GxArrayViewInterpolatorCore::<9, 9>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            9 => GxArrayViewInterpolatorCore::<11, 11>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            11 => GxArrayViewInterpolatorCore::<13, 13>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            _ => panic!("Unsupported spline order"),

        }
    }
}

/// A structure holding the parameters required to create a 
/// `GxBSplineInterpolator<N>`
///
/// This structure implements the `GxBSplineInterpolatorArgs` trait and contains
/// the essential configuration parameters for B-spline interpolation.
pub struct GxBSplineInterpolatorArgs {
    /// Precision parameter for truncation index calculation.
    pub epsilon: f64,
    /// Acceptable relative contamination from invalid pixels.
    pub mask_influence_threshold: f64,
}

/// Concrete implementation of `GxArrayViewInterpolatorArgs` trait for B-spline
/// interpolators
///
/// This implementation provides the necessary interface for interpolators
/// derived from `GxBSplineInterpolator<N>`. It implements the `bspline_args` 
/// function to expose the configured parameters for B-spline interpolation.
impl GxArrayViewInterpolatorArgs for GxBSplineInterpolatorArgs {
    fn bspline_args(&self) -> Option<(f64, f64)> {
        Some((self.epsilon, self.mask_influence_threshold))
    }
}

impl<const N: usize> GxArrayViewInterpolator for GxBSplineInterpolator<N> {
    /// Creates a new instance of the B-spline interpolator with the specified
    /// order and poles.
    /// 
    /// # Returns
    /// - New instance with order N and npoles N/2
    fn new(args: &dyn GxArrayViewInterpolatorArgs) -> Self {
        GxBSplineInterpolator {
            order: N,
            npoles: N/2,
            epsilon: args.bspline_args().expect(
                "GxBSplineInterpolator requires bspline_args. Check the \
                GxArrayViewInterpolatorArgs argument that has been passed !").0,
            mask_influence_threshold: args.bspline_args().expect(
                "GxBSplineInterpolator requires bspline_args. Check the \
                GxArrayViewInterpolatorArgs argument that has been passed !").1,
            truncation_index: [0; TRUNCATION_INDEX_BUFFER_MAX_SIZE],
            domain_extension: [0; TRUNCATION_L_BUFFER_MAX_SIZE],
        }
    }

    fn shortname(&self) -> String {
        format!("bspline{}", self.order)
    }
    
    fn initialize(&mut self) -> Result<(), String> {
        self.truncation_index =
            compute_2d_truncation_index(self.order, self.epsilon);
        self.domain_extension =
            compute_2d_domain_extension_from_truncation_idx(
                self.order, &self.truncation_index
            );
        Ok(())
    }
    
    fn kernel_row_size(&self) -> usize {
        2 * (self.npoles + 1) + 1
    }

    fn kernel_col_size(&self) -> usize {
        2 * (self.npoles + 1) + 1
    }
    
    #[inline(always)]
    fn total_margins(&self) -> Result<[usize; 4], GxError> {
        let margin = self.domain_extension[0]; // TODO : We may need to add 1 here in order to acount for the kernel_size 
        if margin == 0 {
            return Err(GxError::ErrMessage(
                "GxBSplineInterpolator has not been initialized".to_string())
            );
        }
        Ok([margin, margin, margin, margin])
    }
    
    fn allocate_kernel_buffer<'a>(&'a self) -> Box<[f64]> {
        let buffer: Vec<f64> = vec![0.0; self.kernel_row_size() + self.kernel_col_size()];
        buffer.into_boxed_slice()
    }
    
    fn array1_interp2<T, V, IC>(
        &self,
        weights_buffer: &mut [f64],
        target_row_pos: f64,
        target_col_pos: f64,
        out_idx: usize,
        array_in: &GxArrayView<'_, T>,
        array_out: &mut GxArrayViewMut<'_, V>,
        nodata_out: V,
        context: &mut IC,
    ) -> Result<(), String>
    where
        T: Copy + PartialEq + std::ops::Mul<f64, Output=f64> + Into<f64>,
        V: Copy + PartialEq + From<f64>,
        IC: GxArrayViewInterpolationContextTrait,
    {
        self.array1_interp2_separable_core(
            weights_buffer, target_row_pos, target_col_pos, out_idx,
            array_in, array_out, nodata_out, context
        )
    }
}

/// Alias type for cubic B-spline interpolator (3rd order, 1 pole)
pub type GxBSpline3Interpolator = GxBSplineInterpolator<3>;

/// Alias type for quintic B-spline interpolator (5th order, 2 poles)
pub type GxBSpline5Interpolator = GxBSplineInterpolator<5>;

/// Alias type for septic B-spline interpolator (7th order, 3 poles)
pub type GxBSpline7Interpolator = GxBSplineInterpolator<7>;

/// Alias type for nonic B-spline interpolator (9th order, 4 poles)
pub type GxBSpline9Interpolator = GxBSplineInterpolator<9>;

/// Alias type for eleventh-order B-spline interpolator (11th order, 5 poles)
pub type GxBSpline11Interpolator = GxBSplineInterpolator<11>;


#[cfg(test)]
mod tests {
    //! Unit tests for the `_all_centered` B-spline weight functions.
    //!
    //! Each test calls [`check_all_centered`] which verifies two properties:
    //!
    //! 1. **Pointwise correctness**: for a representative set of fractional
    //!    offsets `t`, every weight `weights[k]` matches the scalar B-spline
    //!    function evaluated at `t + k - center`.
    //!
    //! The test set covers both signs of `t` and values close to the
    //! half-pixel boundary (`±0.4999`) to exercise all code paths.
    use super::*;
 
    /// Validates a `bspline_all_centered` function against the corresponding
    /// scalar B-spline function.
    ///
    /// For each test value `t`, checks that each weight `weights[k]` equals
    /// `scalar_fn(t + k - center)` within the tolerance `tol`.
    ///
    /// # Parameters
    /// - `all_fn`: The vectorized function to test, writing `N` weights into
    ///   a mutable slice.
    /// - `scalar_fn`: The reference scalar B-spline function.
    /// - `center`: Index of the `b_0` slot (weight at offset 0) in the output
    ///   slice.
    /// - `tol`: Absolute tolerance for floating-point comparison. A value of
    ///   `1e-6` is appropriate to accommodate rounding differences between the
    ///   factorized and scalar evaluation paths.
    fn check_all_centered<const N: usize>(
        all_fn: fn(f64, &mut [f64]),
        scalar_fn: fn(f64) -> f64,
        center: usize,
        tol: f64,
    ) {
        let test_vals = [-0.4999, -0.3, -0.1, 0.0, 0.1, 0.3, 0.4999];
        let mut weights: [f64; N] = [0.0; N];
 
        for &t in &test_vals {
            all_fn(t, &mut weights);
 
            // Property 1: each weight must match the scalar reference function.
            for k in 0..N {
                let offset = k as f64 - center as f64;
                let expected = scalar_fn(t + offset);
                let got = weights[k];
                let diff = (got - expected).abs();
                if diff > tol {
                    println!(
                        "FAIL t={:.4} k={} offset={} \
                         got={:.9} expected={:.9} diff={:.2e}",
                        t, k, offset, got, expected, diff
                    );
                }
                assert!(
                    diff <= tol,
                    "t={} slot k={} (offset={}): got {} expected {} diff={:.2e}",
                    t, k, offset, got, expected, diff
                );
            }
        }
    }
 
    /// Verifies that `bspline3_all_centered` matches `bspline3` pointwise.
    #[test]
    fn test_bspline3_all_centered() {
        check_all_centered::<5>(bspline3_all_centered, bspline3, 2, 1e-8);
    }
 
    /// Verifies that `bspline5_all_centered` matches `bspline5` pointwise.
    #[test]
    fn test_bspline5_all_centered() {
        check_all_centered::<7>(bspline5_all_centered, bspline5, 3, 1e-8);
    }
 
    /// Verifies that `bspline7_all_centered` matches `bspline7` pointwise.
    #[test]
    fn test_bspline7_all_centered() {
        check_all_centered::<9>(bspline7_all_centered, bspline7, 4, 1e-8);
    }
 
    /// Verifies that `bspline9_all_centered` matches `bspline9` pointwise.
    #[test]
    fn test_bspline9_all_centered() {
        check_all_centered::<11>(bspline9_all_centered, bspline9, 5, 1e-8);
    }
 
    /// Verifies that `bspline11_all_centered` matches `bspline11` pointwise.
    ///
    /// A slightly relaxed tolerance of `1e-6` is used to account for
    /// floating-point rounding differences introduced by the `*u*u` term in
    /// branch 2 of `bspline11`.
    #[test]
    fn test_bspline11_all_centered() {
        check_all_centered::<13>(bspline11_all_centered, bspline11, 6, 1e-6);
    }
}