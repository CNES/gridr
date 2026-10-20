// Copyright (c) 2024-2026 Centre National d'Etudes Spatiales (CNES).
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
//! | Order | Kernel Radius | Stencil (taps) | Type Alias                |
//! |-------|---------------|----------------|---------------------------|
//! | 3     | 2             | 4              | `GxBSpline3Interpolator`  |
//! | 5     | 3             | 6              | `GxBSpline5Interpolator`  |
//! | 7     | 4             | 8              | `GxBSpline7Interpolator`  |
//! | 9     | 5             | 10             | `GxBSpline9Interpolator`  |
//! | 11    | 6             | 12             | `GxBSpline11Interpolator` |
//!
//! The stencil is the native support of the kernel (`order + 1` taps),
//! anchored on `floor(x)`.
//!
//! # Weights on the native support
//!
//! The B-spline functions of this module are scaled by `n!`:
//! `bsplineN(x) = n! β^n(x)`, where `β^n` is the centered B-spline of odd
//! degree `n`, supported on `|x| < R` with `R = (n + 1) / 2`:
//!
//! ```latex
//! n!\,\beta^n(x) = \sum_{i=0}^{n+1} (-1)^i \binom{n+1}{i} \left(x + R - i\right)_+^{\,n}
//! ```
//!
//! For an odd degree the knots are the integers, so on each interval
//! `[k - 1, k)` of `|x|`, `k = 1..=R`, the function is a polynomial `Pk` of
//! degree `n`. Each piece is written in a variable that lies in `[0, 1]`:
//!
//! - `P1` in `u = |x|`;
//! - `Pk`, `k >= 2`, in `u = k - |x|`, the distance to the outer end of the
//!   piece. The outer piece is `PR(u) = u^n`.
//!
//! The stencil has `n + 1` taps, from `anchor - (R - 1)` to `anchor + R`,
//! anchored on `floor(x)`. With the phase `t = x - floor(x)` in `[0, 1)` and
//! `s = 1 - t`, the tap `anchor + d` lies at distance `|t - d|` from the
//! target, which always falls in a single piece:
//!
//! ```text
//! tap                  distance      weight
//! anchor - j, j >= 1   j + t         P(j+1)(s)
//! anchor               t             P1(t)
//! anchor + 1           s             P1(s)
//! anchor + j, j >= 2   (j - 1) + s   Pj(t)
//! ```
//!
//! Each weight is one polynomial evaluation in `t` or `s`: no piece is
//! selected at run time and there is no branch. Weights are stored in
//! inverted order, `weights[k]` being the weight of tap `anchor + R - k`.
//!
//! Properties, for every phase:
//!
//! - the weights sum to `n!` ;
//! - the weights at `1 - t` are the weights at `t` in reverse order;
//! - at `t = 0` the tap `anchor + R` lies at distance `R`, where `β^n`
//!   vanishes, so `weights[0] = t^n` is an exact zero and the other `n`
//!   weights are positive; for `t` in `(0, 1)` all `n + 1` weights are
//!   positive;
//! - the outer weights are the powers `t^n` and `s^n`, and the inner pieces
//!   do not vanish on `[0, 1]`, so rounding never turns a weight inside the
//!   support into an exact zero.
//!
//! # Usage Example
//!
//! ```ignore
//! use gridr::interp::gx_bspline_kernel::{
//!     GxBSpline5Interpolator,
//!     GxBSplineInterpolatorArgs,
//! };
//! use gridr::interp::gx_array_view_interp::GxArrayViewInterpolator;
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
//! *Image Processing On Line*, 8, 99-141. <https://doi.org/10.5201/ipol.2018.221>
use::std::format;
use crate::gx_array::{
    GxArrayView,
    GxArrayViewMut,
    GxArrayWindow,
};
use crate::gx_errors::GxError;
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
    array1_bspline_prefiltering_ext_gene_mask_safe_win,
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

/// Cubic B-spline (degree 3) weights on its native 4-tap support.
///
/// Radius 2, taps from `anchor - 1` to `anchor + 2`. The phase is
/// `t = x - floor(x)` in `[0, 1)` and `s = 1 - t`. With the pieces of
/// `bspline3` written as in the module documentation, the weights are:
///
/// ```text
/// slot        tap         distance  weight
/// weights[0]  anchor + 2  1 + s     t^3
/// weights[1]  anchor + 1  s         P1(s)
/// weights[2]  anchor      t         P1(t)
/// weights[3]  anchor - 1  1 + t     s^3
/// ```
///
/// The weights sum to `3! = 6`. At `t = 0` only `weights[0]` is zero.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 4`.
#[inline]
pub fn bspline3_weights_native(t: f64, weights: &mut [f64]) {
    debug_assert!((0.0..1.0).contains(&t), "phase out of [0, 1): {t}");
    // SAFETY: caller guarantees weights.len() >= 4.
    let w = unsafe { weights.get_unchecked_mut(0..4) };

    // t in [0, 1), s = 1 - t in (0, 1].
    let s = 1.0 - t;
    let t2 = t * t; let t3 = t2 * t;
    let s2 = s * s; let s3 = s2 * s;

    w[0] = t3;
    w[1] = 4.0 + (-6.0 + 3.0*s)*s2;
    w[2] = 4.0 + (-6.0 + 3.0*t)*t2;
    w[3] = s3;
}

/// Quintic B-spline (degree 5) weights on its native 6-tap support.
///
/// Radius 3, taps from `anchor - 2` to `anchor + 3`. The phase is
/// `t = x - floor(x)` in `[0, 1)` and `s = 1 - t`. With the pieces of
/// `bspline5` written as in the module documentation, the weights are:
///
/// ```text
/// slot        tap         distance  weight
/// weights[0]  anchor + 3  2 + s     t^5
/// weights[1]  anchor + 2  1 + s     P2(t)
/// weights[2]  anchor + 1  s         P1(s)
/// weights[3]  anchor      t         P1(t)
/// weights[4]  anchor - 1  1 + t     P2(s)
/// weights[5]  anchor - 2  2 + t     s^5
/// ```
///
/// The weights sum to `5! = 120`. At `t = 0` only `weights[0]` is zero.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 6`.
#[inline]
pub fn bspline5_weights_native(t: f64, weights: &mut [f64]) {
    debug_assert!((0.0..1.0).contains(&t), "phase out of [0, 1): {t}");
    // SAFETY: caller guarantees weights.len() >= 6.
    let w = unsafe { weights.get_unchecked_mut(0..6) };

    // t in [0, 1), s = 1 - t in (0, 1].
    let s = 1.0 - t;
    let t2 = t * t; let t4 = t2 * t2; let t5 = t4 * t;
    let s2 = s * s; let s4 = s2 * s2; let s5 = s4 * s;

    w[0] = t5;
    w[1] = 1.0 + (5.0 + (10.0 + (10.0 + (5.0 - 5.0*t)*t)*t)*t)*t;
    w[2] = ((-10.0*s + 30.0)*s2 - 60.0)*s2 + 66.0;
    w[3] = ((-10.0*t + 30.0)*t2 - 60.0)*t2 + 66.0;
    w[4] = 1.0 + (5.0 + (10.0 + (10.0 + (5.0 - 5.0*s)*s)*s)*s)*s;
    w[5] = s5;
}

/// Septic B-spline (degree 7) weights on its native 8-tap support.
///
/// Radius 4, taps from `anchor - 3` to `anchor + 4`. The phase is
/// `t = x - floor(x)` in `[0, 1)` and `s = 1 - t`. With the pieces of
/// `bspline7` written as in the module documentation (`Pk` is `brK` in the
/// body), the weights are:
///
/// ```text
/// slot        tap         distance  weight
/// weights[0]  anchor + 4  3 + s     t^7
/// weights[1]  anchor + 3  2 + s     P3(t)
/// weights[2]  anchor + 2  1 + s     P2(t)
/// weights[3]  anchor + 1  s         P1(s)
/// weights[4]  anchor      t         P1(t)
/// weights[5]  anchor - 1  1 + t     P2(s)
/// weights[6]  anchor - 2  2 + t     P3(s)
/// weights[7]  anchor - 3  3 + t     s^7
/// ```
///
/// The weights sum to `7! = 5040`. At `t = 0` only `weights[0]` is zero.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 8`.
#[inline]
pub fn bspline7_weights_native(t: f64, weights: &mut [f64]) {
    debug_assert!((0.0..1.0).contains(&t), "phase out of [0, 1): {t}");
    // SAFETY: caller guarantees weights.len() >= 8.
    let w = unsafe { weights.get_unchecked_mut(0..8) };

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
}

/// Nonic B-spline (degree 9) weights on its native 10-tap support.
///
/// Radius 5, taps from `anchor - 4` to `anchor + 5`. The phase is
/// `t = x - floor(x)` in `[0, 1)` and `s = 1 - t`. With the pieces of
/// `bspline9` written as in the module documentation (`Pk` is `brK` in the
/// body), the weights are:
///
/// ```text
/// slot        tap         distance  weight
/// weights[0]  anchor + 5  4 + s     t^9
/// weights[1]  anchor + 4  3 + s     P4(t)
/// weights[2]  anchor + 3  2 + s     P3(t)
/// weights[3]  anchor + 2  1 + s     P2(t)
/// weights[4]  anchor + 1  s         P1(s)
/// weights[5]  anchor      t         P1(t)
/// weights[6]  anchor - 1  1 + t     P2(s)
/// weights[7]  anchor - 2  2 + t     P3(s)
/// weights[8]  anchor - 3  3 + t     P4(s)
/// weights[9]  anchor - 4  4 + t     s^9
/// ```
///
/// The weights sum to `9! = 362880`. At `t = 0` only `weights[0]` is zero.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 10`.
#[inline]
pub fn bspline9_weights_native(t: f64, weights: &mut [f64]) {
    debug_assert!((0.0..1.0).contains(&t), "phase out of [0, 1): {t}");
    // SAFETY: caller guarantees weights.len() >= 10.
    let w = unsafe { weights.get_unchecked_mut(0..10) };

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
}

/// B-spline of degree 11 weights on its native 12-tap support.
///
/// Radius 6, taps from `anchor - 5` to `anchor + 6`. The phase is
/// `t = x - floor(x)` in `[0, 1)` and `s = 1 - t`. With the pieces of
/// `bspline11` written as in the module documentation (`Pk` is `brK` in the
/// body), the weights are:
///
/// ```text
/// slot         tap         distance  weight
/// weights[0]   anchor + 6  5 + s     t^11
/// weights[1]   anchor + 5  4 + s     P5(t)
/// weights[2]   anchor + 4  3 + s     P4(t)
/// weights[3]   anchor + 3  2 + s     P3(t)
/// weights[4]   anchor + 2  1 + s     P2(t)
/// weights[5]   anchor + 1  s         P1(s)
/// weights[6]   anchor      t         P1(t)
/// weights[7]   anchor - 1  1 + t     P2(s)
/// weights[8]   anchor - 2  2 + t     P3(s)
/// weights[9]   anchor - 3  3 + t     P4(s)
/// weights[10]  anchor - 4  4 + t     P5(s)
/// weights[11]  anchor - 5  5 + t     s^11
/// ```
///
/// The weights sum to `11! = 39916800`. At `t = 0` only `weights[0]` is zero.
///
/// # Safety
///
/// The caller must guarantee that `weights.len() >= 12`.
#[inline]
pub fn bspline11_weights_native(t: f64, weights: &mut [f64]) {
    debug_assert!((0.0..1.0).contains(&t), "phase out of [0, 1): {t}");
    // SAFETY: caller guarantees weights.len() >= 12.
    let w = unsafe { weights.get_unchecked_mut(0..12) };

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
    
    
    /// Computes the weights of the B-spline on its native support.
    ///
    /// # Parameters
    /// - `t`: phase of the target position relative to the anchor
    ///   `floor(x)`, in `[0, 1)`.
    /// - `weights`: output slice of length at least `N + 1`, in inverted
    ///   order (see the module documentation).
    fn bspline_kernel_weights(&self, t: f64, weights: &mut [f64]);
    
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

    /// Update Safe-Region definition considering pre-filtering
    ///
    /// # Parameters
    /// - `mask_in`: Reference to the input mask. Primarily used to validate the updated window.
    /// - `safe_win`: Reference to the safe window
    ///
    /// # Returns
    /// - Optional safe_window if not empty
    fn array1_bspline_prefiltering_ext_mask_safe_win(
        &self,
        mask_in: &GxArrayView<u8>,
        safe_win: &GxArrayWindow
    ) -> Result<Option<GxArrayWindow>, GxError>;
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
    fn bspline_kernel_weights(&self, t: f64, weights: &mut [f64]) 
    {
        match N {
            3 => bspline3_weights_native(t, weights),
            5 => bspline5_weights_native(t, weights),
            7 => bspline7_weights_native(t, weights),
            9 => bspline9_weights_native(t, weights),
            11 => bspline11_weights_native(t, weights),
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

    #[inline]
    fn array1_bspline_prefiltering_ext_mask_safe_win(
        &self,
        mask_in: &GxArrayView<u8>,
        safe_win: &GxArrayWindow,
    ) -> Result<Option<GxArrayWindow>, GxError>
    {
        match N {
            3 | 5 | 7 | 9 | 11 => {
                return Ok(
                    array1_bspline_prefiltering_ext_gene_mask_safe_win(
                        self.order,
                        self.epsilon,
                        Some(&self.truncation_index),
                        mask_in,
                        safe_win,
                        self.mask_influence_threshold,
                    )
                );
            },
            _ => {
                return Err(GxError::ErrMessage("Unsupported spline order".to_string()));
            },
        }
    }


}


impl<const N: usize> GxArrayViewInterpolatorCore<4, 4> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, t: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(t, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<6, 6> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, t: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(t, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<8, 8> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, t: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(t, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<10, 10> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, t: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(t, weights)
    }
}
impl<const N: usize> GxArrayViewInterpolatorCore<12, 12> for GxBSplineInterpolator<N> {
    #[inline(always)]
    fn compute_weights(&self, t: f64, weights: &mut [f64])
    {
        self.bspline_kernel_weights(t, weights)
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
            3 => GxArrayViewInterpolatorCore::<4, 4>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            5 => GxArrayViewInterpolatorCore::<6, 6>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            7 => GxArrayViewInterpolatorCore::<8, 8>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            9 => GxArrayViewInterpolatorCore::<10, 10>::array1_interp2_separable_core(
                    self, weights_buffer, target_row_pos, target_col_pos,
                    out_idx, array_in, array_out, nodata_out, context
                 ),
            11 => GxArrayViewInterpolatorCore::<12, 12>::array1_interp2_separable_core(
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
        2 * (self.npoles + 1)
    }

    fn kernel_col_size(&self) -> usize {
        2 * (self.npoles + 1)
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
    //! Unit tests for the native B-spline weight functions.
    use super::*;
 
    /// Check native weights against the scalar B-spline functions.
    /// GridR B-splines are scaled by `n!`, so the weights sum to `n!`.
    #[test]
    fn test_bspline_weights_native() {
        use crate::interp::gx_kernel_test_utils::check_native_weights;
        check_native_weights(bspline3_weights_native, 4, bspline3, 6.0, 1e-14);
        check_native_weights(bspline5_weights_native, 6, bspline5, 120.0, 1e-14);
        check_native_weights(bspline7_weights_native, 8, bspline7, 5040.0, 1e-14);
        check_native_weights(bspline9_weights_native, 10, bspline9, 362880.0, 1e-14);
        check_native_weights(bspline11_weights_native, 12, bspline11, 39916800.0, 1e-14);
    }
}
