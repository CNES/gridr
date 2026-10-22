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

//! Integration tests of `GxArrayViewInterpolator::array1_interp2`.
//!
//! Each interpolator is called through its public API (`new`, `initialize`,
//! `allocate_kernel_buffer`, `array1_interp2`), so that the kernel weights, the
//! convolution loop and the mask and bounds strategies are tested together.
//! The output is compared with a direct evaluation of the continuous kernel
//! `phi`:
//!
//! ```text
//! value(y, x) = sum_i sum_j phi(y - i) * phi(x - j) * a[i, j]
//! valid(y, x) = every (i, j) with phi(y - i) != 0 and phi(x - j) != 0
//!               lies inside the array and is valid in the input mask
//! ```
//!
//! The validity rule above is the effective support policy: samples with
//! a zero weight are ignored, whether they lie outside the array or are
//! masked.
//!
//! Every case is also run under the full support policy, where the rule is
//!
//! ```text
//! valid(y, x) = every tap of the native stencil lies inside the array
//!               and is valid in the input mask, whatever its weight
//! ```
//!
//! the stencil being `floor(y) - (K / 2 - 1) ..= floor(y) + K / 2` on each
//! axis for a support of `K = 2 * radius` taps.
//!
//! The tests do not depend on how the stencil is anchored nor on the order in
//! which the weights are stored. Output masks are compared exactly; values
//! within a tolerance relative to `sum |phi(y - i) * phi(x - j) * a[i, j]|`.

use gridr::gx_array::{GxArrayView, GxArrayViewMut};
use gridr::interp::gx_array_view_interp::{
    BinaryInputMask, BinaryInputMaskWithSafeWindow, BinaryOutputMask, BoundsCheck,
    Effective, GxArrayViewInterpolationContext, GxArrayViewInterpolator,
    GxArrayViewInterpolatorBoundsCheckStrategy, GxArrayViewInterpolatorInputMaskStrategy,
    GxArrayViewInterpolatorNoArgs, GxArrayViewInterpolatorSupportPolicy, NoBoundsCheck,
    NoInputMask, Full,
};
use gridr::interp::gx_bspline_kernel::{bspline11, bspline3, bspline5, bspline7, bspline9};

use crate::common::bspline;
use gridr::interp::gx_linear_kernel::GxLinearInterpolator;
use gridr::interp::gx_nearest_kernel::GxNearestInterpolator;
use gridr::interp::gx_optimized_bicubic_kernel::GxOptimizedBicubicInterpolator;

const NVAR: usize = 2;
const NROW: usize = 26;
const NCOL: usize = 29;
const NODATA: f64 = -99999.0;

// =============================================================================
// Deterministic pseudo-random generator (no external dependency)
// =============================================================================

/// xorshift64* generator, only used to build reproducible fixtures.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform sample in `[lo, hi)`.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }

    /// Uniform integer in `[lo, hi]`.
    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % ((hi - lo + 1) as u64)) as i64
    }
}

// =============================================================================
// Continuous kernels (reference definitions)
// =============================================================================
//
// The linear and cubic reference kernels are written here, independently of
// the crate code, on purpose: the reference must not share its definition of
// a kernel with the implementation it checks. The B-spline references are the
// scalar functions `bsplineN` of the crate, which are distinct from the weight
// functions under test and checked by their own unit tests.

/// Continuous kernel used as reference: `phi(x)` and its support radius.
#[derive(Clone, Copy)]
struct Kernel {
    phi: fn(f64) -> f64,
    radius: f64,
}

fn phi_linear(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// Keys cubic convolution kernel with `a = -0.5`.
///
/// Both pieces are written in factored form so that the value stays
/// accurate (and non-zero) next to the support boundaries `|x| = 1` and
/// `|x| = 2`, where the expanded polynomials suffer from cancellation.
fn phi_keys(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        // 1.5 x^3 - 2.5 x^2 + 1
        (x - 1.0) * (1.5 * x * x - x - 1.0)
    } else if x < 2.0 {
        // -0.5 x^3 + 2.5 x^2 - 4 x + 2
        let d = x - 2.0;
        -0.5 * (x - 1.0) * d * d
    } else {
        0.0
    }
}

const LINEAR: Kernel = Kernel { phi: phi_linear, radius: 1.0 };
const KEYS: Kernel = Kernel { phi: phi_keys, radius: 2.0 };
// GridR B-spline functions are scaled by n! and are the reference
// definitions already used by the B-spline unit tests.
const BSPLINE3: Kernel = Kernel { phi: bspline3, radius: 2.0 };
const BSPLINE5: Kernel = Kernel { phi: bspline5, radius: 3.0 };
const BSPLINE7: Kernel = Kernel { phi: bspline7, radius: 4.0 };
const BSPLINE9: Kernel = Kernel { phi: bspline9, radius: 5.0 };
const BSPLINE11: Kernel = Kernel { phi: bspline11, radius: 6.0 };

// =============================================================================
// Fixtures
// =============================================================================

struct Fixture {
    data: Vec<f64>,
    /// Random sparse invalid pixels (about 12 %).
    mask_sparse: Vec<u8>,
    /// Random rare invalid pixels (about 1.5 %), so that large kernels
    /// still produce valid outputs.
    mask_rare: Vec<u8>,
    /// Invalid pixels only in an outer band, used with a safe window.
    mask_band: Vec<u8>,
    /// Inclusive safe zone `[row_min, row_max, col_min, col_max]` of
    /// `mask_band`.
    safe_zone: [usize; 4],
}

const BAND: usize = 5;

fn fixture() -> Fixture {
    let mut rng = Rng::new(0x5EED_0007);
    let data: Vec<f64> = (0..NVAR * NROW * NCOL)
        .map(|_| rng.uniform(-100.0, 100.0))
        .collect();
    let mask_sparse: Vec<u8> = (0..NROW * NCOL)
        .map(|_| if rng.uniform(0.0, 1.0) < 0.12 { 0 } else { 1 })
        .collect();
    let mask_rare: Vec<u8> = (0..NROW * NCOL)
        .map(|_| if rng.uniform(0.0, 1.0) < 0.015 { 0 } else { 1 })
        .collect();
    let safe_zone = [BAND, NROW - 1 - BAND, BAND, NCOL - 1 - BAND];
    let mask_band: Vec<u8> = (0..NROW * NCOL)
        .map(|k| {
            let (r, c) = (k / NCOL, k % NCOL);
            let inside = r >= safe_zone[0]
                && r <= safe_zone[1]
                && c >= safe_zone[2]
                && c <= safe_zone[3];
            if inside || rng.uniform(0.0, 1.0) < 0.7 { 1 } else { 0 }
        })
        .collect();
    Fixture { data, mask_sparse, mask_rare, mask_band, safe_zone }
}

/// Target positions `(row, col)` covering the phases and borders of interest.
fn positions_all(seed: u64) -> Vec<(f64, f64)> {
    let mut rng = Rng::new(seed);
    let (nr, nc) = (NROW as f64, NCOL as f64);
    let mut pts = Vec::new();

    // Random positions, inside and outside the array.
    for _ in 0..3000 {
        pts.push((rng.uniform(-3.0, nr + 2.0), rng.uniform(-3.0, nc + 2.0)));
    }
    // Integer positions on both axes (phase 0 : exercices the effective
    // policy for interpolating kernels).
    for r in -2..=(NROW as i64 + 1) {
        for c in -2..=(NCOL as i64 + 1) {
            pts.push((r as f64, c as f64));
        }
    }
    // Integer on one axis only.
    for _ in 0..800 {
        pts.push((rng.int(-2, NROW as i64 + 1) as f64, rng.uniform(-3.0, nc + 2.0)));
        pts.push((rng.uniform(-3.0, nr + 2.0), rng.int(-2, NCOL as i64 + 1) as f64));
    }
    // Half-integer phases (rounding tie of the centred anchoring).
    for _ in 0..500 {
        pts.push((
            rng.int(-2, NROW as i64 + 1) as f64 + 0.5,
            rng.int(-2, NCOL as i64 + 1) as f64 - 0.5,
        ));
    }
    // Near-integer phases: grid coordinates are rounded to 1e-12 in the
    // resampling loop, so offsets of that order do occur in practice. The
    // weights of the farthest taps are tiny but non-zero there; they must not
    // be computed as exact zeros.
    for &d in &[1e-12, -1e-12, 1e-9, -1e-9, 1e-6, -1e-6] {
        for _ in 0..150 {
            pts.push((
                rng.int(-1, NROW as i64) as f64 + d,
                rng.int(-1, NCOL as i64) as f64 - d,
            ));
        }
    }
    // Last row / last column and their neighbourhood.
    for &r in &[-0.5, -0.25, 0.0, nr - 1.5, nr - 1.0, nr - 0.75, nr - 0.5] {
        for _ in 0..40 {
            pts.push((r, rng.uniform(-1.0, nc)));
            pts.push((rng.uniform(-1.0, nr), r * nc / nr));
        }
    }
    pts
}

/// Positions whose whole stencil (centred or native) is inside the array,
/// for the `NoBoundsCheck` strategy.
fn positions_interior(seed: u64, margin: usize) -> Vec<(f64, f64)> {
    let mut rng = Rng::new(seed);
    let (lo_r, hi_r) = (margin as f64, (NROW - 1 - margin) as f64);
    let (lo_c, hi_c) = (margin as f64, (NCOL - 1 - margin) as f64);
    let mut pts = Vec::new();
    for _ in 0..2000 {
        pts.push((rng.uniform(lo_r, hi_r), rng.uniform(lo_c, hi_c)));
    }
    for _ in 0..500 {
        pts.push((
            rng.int(margin as i64, (NROW - 1 - margin) as i64) as f64,
            rng.uniform(lo_c, hi_c),
        ));
    }
    pts
}

// =============================================================================
// Reference evaluation
// =============================================================================

/// Brute-force reference. Returns `None` when the output must be invalid,
/// otherwise the value of every variable and the scale used for the
/// tolerance.
/// Weights equal to 0 are ignored - corresponds to the effective policy.
fn reference(
    k: Kernel,
    data: &[f64],
    mask: Option<&[u8]>,
    y: f64,
    x: f64,
) -> Option<([f64; NVAR], f64)> {
    let i0 = (y - k.radius).floor() as i64;
    let i1 = (y + k.radius).ceil() as i64;
    let j0 = (x - k.radius).floor() as i64;
    let j1 = (x + k.radius).ceil() as i64;

    let mut acc = [0.0; NVAR];
    let mut scale = 0.0;
    for i in i0..=i1 {
        let wy = (k.phi)(y - i as f64);
        if wy == 0.0 {
            continue;
        }
        let mut acc_row = [0.0; NVAR];
        for j in j0..=j1 {
            let wx = (k.phi)(x - j as f64);
            if wx == 0.0 {
                continue;
            }
            if i < 0 || i >= NROW as i64 || j < 0 || j >= NCOL as i64 {
                return None;
            }
            let flat = i as usize * NCOL + j as usize;
            if let Some(m) = mask {
                if m[flat] == 0 {
                    return None;
                }
            }
            for v in 0..NVAR {
                let a = data[v * NROW * NCOL + flat];
                acc_row[v] += wx * a;
                scale += (wy * wx * a).abs();
            }
        }
        for v in 0..NVAR {
            acc[v] += wy * acc_row[v];
        }
    }
    Some((acc, scale))
}

/// Full-support reference: every tap of the native stencil (even support of
/// `2 * radius` taps anchored on `floor`) must lie inside the array and be
/// valid in the input mask; the value is the same as the effective one.
fn reference_full(
    k: Kernel,
    data: &[f64],
    mask: Option<&[u8]>,
    y: f64,
    x: f64,
) -> Option<([f64; NVAR], f64)> {
    let half = k.radius as i64;
    let (fy, fx) = (y.floor() as i64, x.floor() as i64);
    for i in (fy - half + 1)..=(fy + half) {
        for j in (fx - half + 1)..=(fx + half) {
            if i < 0 || i >= NROW as i64 || j < 0 || j >= NCOL as i64 {
                return None;
            }
            if let Some(m) = mask {
                if m[i as usize * NCOL + j as usize] == 0 {
                    return None;
                }
            }
        }
    }
    reference(k, data, mask, y, x)
}

/// Reference for the nearest neighbour interpolator.
fn reference_nearest(
    data: &[f64],
    mask: Option<&[u8]>,
    y: f64,
    x: f64,
) -> Option<([f64; NVAR], f64)> {
    let i = (y + 0.5).floor() as i64;
    let j = (x + 0.5).floor() as i64;
    if i < 0 || i >= NROW as i64 || j < 0 || j >= NCOL as i64 {
        return None;
    }
    let flat = i as usize * NCOL + j as usize;
    if let Some(m) = mask {
        if m[flat] == 0 {
            return None;
        }
    }
    let mut out = [0.0; NVAR];
    for v in 0..NVAR {
        out[v] = data[v * NROW * NCOL + flat];
    }
    Some((out, 0.0))
}

// =============================================================================
// Runner
// =============================================================================

/// Calls `array1_interp2` for every position with the given input mask,
/// bounds-check strategy and support policy. Returns the output values
/// (`NVAR` planes) and the output mask.
fn run<I, IM, BC, SP>(
    interp: &I,
    data: &[f64],
    pts: &[(f64, f64)],
    input_mask: IM,
    bounds: BC,
    policy: SP,
) -> (Vec<f64>, Vec<u8>)
where
    I: GxArrayViewInterpolator,
    IM: GxArrayViewInterpolatorInputMaskStrategy,
    BC: GxArrayViewInterpolatorBoundsCheckStrategy,
    SP: GxArrayViewInterpolatorSupportPolicy,
{
    let n = pts.len();
    let mut out = vec![f64::NAN; NVAR * n];
    let mut mask_out = vec![77u8; n];
    {
        let arr_in = GxArrayView::new(data, NVAR, NROW, NCOL);
        let mut arr_out = GxArrayViewMut::new(&mut out, NVAR, 1, n);
        let mut mask_view = GxArrayViewMut::new(&mut mask_out, 1, 1, n);
        let mut ctx = GxArrayViewInterpolationContext::with_policy(
            input_mask,
            BinaryOutputMask { mask: &mut mask_view },
            bounds,
            policy,
        );
        let mut buf = interp.allocate_kernel_buffer();
        for (k, &(y, x)) in pts.iter().enumerate() {
            interp
                .array1_interp2(&mut buf, y, x, k, &arr_in, &mut arr_out, NODATA, &mut ctx)
                .expect("array1_interp2 returned an error");
        }
    }
    (out, mask_out)
}

/// Compares an interpolation run with the reference, panics with a summary
/// of the first failures.
fn compare(
    label: &str,
    pts: &[(f64, f64)],
    out: &[f64],
    mask_out: &[u8],
    expected: impl Fn(f64, f64) -> Option<([f64; NVAR], f64)>,
    rtol: f64,
) {
    let n = pts.len();
    let mut failures = Vec::new();
    let mut max_rel = 0.0f64;
    let mut n_valid = 0usize;

    for (k, &(y, x)) in pts.iter().enumerate() {
        let exp = expected(y, x);
        let got_valid = mask_out[k];
        match exp {
            None => {
                let nodata_ok = (0..NVAR).all(|v| out[v * n + k] == NODATA);
                if got_valid != 0 || !nodata_ok {
                    failures.push(format!(
                        "({y:?}, {x:?}): expected invalid, got mask={got_valid} value={}",
                        out[k]
                    ));
                }
            }
            Some((vals, scale)) => {
                n_valid += 1;
                if got_valid != 1 {
                    failures.push(format!(
                        "({y:?}, {x:?}): expected valid {:?}, got mask={got_valid} value={}",
                        vals, out[k]
                    ));
                    continue;
                }
                for v in 0..NVAR {
                    let got = out[v * n + k];
                    let err = (got - vals[v]).abs();
                    let tol = rtol * scale + 1e-300;
                    if scale > 0.0 {
                        max_rel = max_rel.max(err / scale);
                    }
                    if err > tol && !(scale == 0.0 && got == vals[v]) {
                        failures.push(format!(
                            "({y:?}, {x:?}) var {v}: got {got:e} expected {:e} \
                             err={err:.2e} tol={tol:.2e}",
                            vals[v]
                        ));
                    }
                }
            }
        }
    }
    println!(
        "{label:<46} points={n:<6} valid={n_valid:<6} max_rel_err={max_rel:.2e}"
    );
    if !failures.is_empty() {
        let shown: Vec<_> = failures.iter().take(15).cloned().collect();
        panic!(
            "{label}: {} mismatches out of {n} points\n{}",
            failures.len(),
            shown.join("\n")
        );
    }
}

/// Reference evaluation of a support policy.
type Reference = fn(Kernel, &[f64], Option<&[u8]>, f64, f64) -> Option<([f64; NVAR], f64)>;

/// Runs the full suite for one separable interpolator under both policies.
fn check_separable<I: GxArrayViewInterpolator>(name: &str, interp: &I, kernel: Kernel, rtol: f64) {
    check_separable_policy(name, interp, kernel, rtol, Effective, "effective", reference);
    check_separable_policy(name, interp, kernel, rtol, Full, "full", reference_full);
}

/// Runs the full suite for one separable interpolator and one policy.
fn check_separable_policy<I, SP>(
    name: &str,
    interp: &I,
    kernel: Kernel,
    rtol: f64,
    policy: SP,
    policy_name: &str,
    expected: Reference,
)
where
    I: GxArrayViewInterpolator,
    SP: GxArrayViewInterpolatorSupportPolicy + Copy,
{
    let fx = fixture();
    let pts = positions_all(0xA11);
    let name = format!("{name} / {policy_name}");

    // Bounds check, no input mask.
    let (out, m) = run(interp, &fx.data, &pts, NoInputMask, BoundsCheck, policy);
    compare(&format!("{name} / bounds / no mask"), &pts, &out, &m,
        |y, x| expected(kernel, &fx.data, None, y, x), rtol);

    // Bounds check, binary input mask.
    let mask_view = GxArrayView::new(&fx.mask_sparse, 1, NROW, NCOL);
    let (out, m) = run(interp, &fx.data, &pts,
        BinaryInputMask { mask: &mask_view }, BoundsCheck, policy);
    compare(&format!("{name} / bounds / binary mask"), &pts, &out, &m,
        |y, x| expected(kernel, &fx.data, Some(&fx.mask_sparse), y, x), rtol);

    let rare_view = GxArrayView::new(&fx.mask_rare, 1, NROW, NCOL);
    let (out, m) = run(interp, &fx.data, &pts,
        BinaryInputMask { mask: &rare_view }, BoundsCheck, policy);
    compare(&format!("{name} / bounds / binary mask (rare)"), &pts, &out, &m,
        |y, x| expected(kernel, &fx.data, Some(&fx.mask_rare), y, x), rtol);

    // Bounds check, binary input mask with safe window. The convolution
    // bounds are derived exactly as in `gx_grid_resampling`.
    let band_view = GxArrayView::new(&fx.mask_band, 1, NROW, NCOL);
    let [r0, r1, c0, c1] = fx.safe_zone;
    let krow = interp.kernel_row_size();
    let kcol = interp.kernel_col_size();
    if let (Some(conv_r1), Some(conv_c1)) = (r1.checked_sub(krow), c1.checked_sub(kcol)) {
        let safe = BinaryInputMaskWithSafeWindow {
            inner: BinaryInputMask { mask: &band_view },
            safe_min_row: r0,
            safe_max_row: r1,
            safe_min_col: c0,
            safe_max_col: c1,
            conv_safe_min_row: r0,
            conv_safe_max_row: conv_r1,
            conv_safe_min_col: c0,
            conv_safe_max_col: conv_c1,
        };
        let (out, m) = run(interp, &fx.data, &pts, safe, BoundsCheck, policy);
        compare(&format!("{name} / bounds / safe window mask"), &pts, &out, &m,
            |y, x| expected(kernel, &fx.data, Some(&fx.mask_band), y, x), rtol);
    }

    // No bounds check: interior positions only. A margin of K / 2 + 1 keeps
    // the whole stencil inside the array.
    let margin = interp.kernel_row_size().max(interp.kernel_col_size()) / 2 + 1;
    let pts_in = positions_interior(0xB22, margin);
    let (out, m) = run(interp, &fx.data, &pts_in, NoInputMask, NoBoundsCheck, policy);
    compare(&format!("{name} / no bounds / no mask"), &pts_in, &out, &m,
        |y, x| expected(kernel, &fx.data, None, y, x), rtol);
    let (out, m) = run(interp, &fx.data, &pts_in,
        BinaryInputMask { mask: &rare_view }, NoBoundsCheck, policy);
    compare(&format!("{name} / no bounds / binary mask (rare)"), &pts_in, &out, &m,
        |y, x| expected(kernel, &fx.data, Some(&fx.mask_rare), y, x), rtol);
}


// =============================================================================
// Tests
// =============================================================================

/// Relative tolerance on values. The reference evaluates the kernels with
/// different (but equivalent) expressions, so a few ulps of the absolute sum
/// are expected.
const RTOL: f64 = 1e-12;

#[test]
fn linear_matches_reference() {
    let interp = GxLinearInterpolator::new(&GxArrayViewInterpolatorNoArgs {});
    check_separable("linear", &interp, LINEAR, RTOL);
}

#[test]
fn cubic_matches_reference() {
    let interp = GxOptimizedBicubicInterpolator::new(&GxArrayViewInterpolatorNoArgs {});
    check_separable("cubic", &interp, KEYS, RTOL);
}

#[test]
fn bspline3_matches_reference() {
    check_separable("bspline3", &bspline::<3>(), BSPLINE3, RTOL);
}

#[test]
fn bspline5_matches_reference() {
    check_separable("bspline5", &bspline::<5>(), BSPLINE5, RTOL);
}

#[test]
fn bspline7_matches_reference() {
    check_separable("bspline7", &bspline::<7>(), BSPLINE7, RTOL);
}

#[test]
fn bspline9_matches_reference() {
    check_separable("bspline9", &bspline::<9>(), BSPLINE9, RTOL);
}

#[test]
fn bspline11_matches_reference() {
    check_separable("bspline11", &bspline::<11>(), BSPLINE11, RTOL);
}

#[test]
fn nearest_matches_reference() {
    let interp = GxNearestInterpolator::new(&GxArrayViewInterpolatorNoArgs {});
    let fx = fixture();
    let pts = positions_all(0xC33);
    let (out, m) = run(&interp, &fx.data, &pts, NoInputMask, BoundsCheck, Effective);
    compare("nearest / bounds / no mask", &pts, &out, &m,
        |y, x| reference_nearest(&fx.data, None, y, x), 0.0);
    let mask_view = GxArrayView::new(&fx.mask_sparse, 1, NROW, NCOL);
    let (out, m) = run(&interp, &fx.data, &pts,
        BinaryInputMask { mask: &mask_view }, BoundsCheck, Effective);
    compare("nearest / bounds / binary mask", &pts, &out, &m,
        |y, x| reference_nearest(&fx.data, Some(&fx.mask_sparse), y, x), 0.0);
    // The support of the nearest neighbour is a single tap: the full
    // policy gives the same results.
    let (out, m) = run(&interp, &fx.data, &pts,
        BinaryInputMask { mask: &mask_view }, BoundsCheck, Full);
    compare("nearest / full / bounds / binary mask", &pts, &out, &m,
        |y, x| reference_nearest(&fx.data, Some(&fx.mask_sparse), y, x), 0.0);
}
