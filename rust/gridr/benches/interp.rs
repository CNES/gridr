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

//! Micro-benchmark of `array1_interp2` for every interpolator and support
//! policy.
//!
//! ```text
//! cargo bench -p gridr --bench interp -- --save-baseline main
//! cargo bench -p gridr --bench interp -- --baseline main
//! cargo bench -p gridr --bench interp -- 'interp/linear/'
//! cargo bench -p gridr --bench interp -- '/full'
//! ```
//!
//! Benchmarks are named `interp/<method>/<scenario>/<policy>`. Each call
//! interpolates `NPTS` target positions; throughput is reported in
//! interpolated points per second. Scenarios:
//!
//! - `regular`: positions on a rotated and slightly zoomed regular grid, as in
//!   a typical resampling. Memory accesses are local, phases vary smoothly.
//! - `random`: uniformly random positions. Phases are unpredictable and
//!   memory accesses scattered.
//! - `regular_masked`: `regular` with about 2 % of invalid input pixels.
//! - `regular_mask30`: `regular` with about 30 % of invalid input pixels:
//!   many stencils hold a masked pixel, and many targets are invalid.
//! - `aligned_mask10`: integer rows (row phase 0) and arbitrary columns,
//!   about 10 % of invalid pixels. Exercises the reduced windows of the
//!   effective support policy.
//! - `integer_mask10`: integer rows and columns (a translation by an integer
//!   offset), about 10 % of invalid pixels.
//! - `border`: positions within a few pixels of the array borders, one in
//!   four on an integer row or column. Exercises the border path.
//! - `border_mask10`: `border` with about 10 % of invalid pixels.
//! - `nocheck_mask10`: `regular` with about 10 % of invalid pixels and
//!   without bounds check.
//!
//! All scenarios except `border*` keep the positions away from the borders.
//! The B-spline variants do not include the prefiltering.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use gridr::gx_array::{GxArrayView, GxArrayViewMut};
use gridr::interp::gx_array_view_interp::{
    BinaryInputMask, BinaryOutputMask, BoundsCheck, GxArrayViewInterpolationContext,
    GxArrayViewInterpolator, GxArrayViewInterpolatorBoundsCheckStrategy,
    GxArrayViewInterpolatorInputMaskStrategy, GxArrayViewInterpolatorNoArgs, NoBoundsCheck,
    NoInputMask,
};
use gridr::interp::gx_array_view_interp::{Effective, GxArrayViewInterpolatorSupportPolicy, Full};
use gridr::interp::gx_bspline_kernel::{GxBSplineInterpolator, GxBSplineInterpolatorArgs};
use gridr::interp::gx_linear_kernel::GxLinearInterpolator;
use gridr::interp::gx_nearest_kernel::GxNearestInterpolator;
use gridr::interp::gx_optimized_bicubic_kernel::GxOptimizedBicubicInterpolator;

/// Input image size (single variable, f64): 512 x 512 = 2 MiB.
const NROW: usize = 512;
const NCOL: usize = 512;
/// Number of target positions interpolated per benchmark iteration.
const NPTS: usize = 16_384;
/// Distance kept from the borders (larger than the widest stencil).
const MARGIN: f64 = 16.0;

/// Support policy marker accepted by the benchmark.
trait Policy: GxArrayViewInterpolatorSupportPolicy + Copy {}
impl<P: GxArrayViewInterpolatorSupportPolicy + Copy> Policy for P {}

/// Builds the interpolation context for a policy.
macro_rules! ctx {
    ($im:expr, $om:expr, $bc:expr, $p:expr) => {
        GxArrayViewInterpolationContext::with_policy($im, $om, $bc, $p)
    };
}

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

/// xorshift64* generator, fixtures must be identical from one run to another.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct Fixture {
    data: Vec<f64>,
    mask2: Vec<u8>,
    mask10: Vec<u8>,
    mask30: Vec<u8>,
    regular: Vec<(f64, f64)>,
    random: Vec<(f64, f64)>,
    aligned: Vec<(f64, f64)>,
    integer: Vec<(f64, f64)>,
    border: Vec<(f64, f64)>,
}

fn mask(rng: &mut Rng, invalid: f64) -> Vec<u8> {
    (0..NROW * NCOL).map(|_| if rng.next_f64() < invalid { 0 } else { 1 }).collect()
}

fn fixture() -> Fixture {
    let mut rng = Rng(0xBE7C_4000_0000_0001);
    let data: Vec<f64> = (0..NROW * NCOL).map(|_| 1000.0 * rng.next_f64()).collect();
    let mask2 = mask(&mut rng, 0.02);
    let mask10 = mask(&mut rng, 0.10);
    let mask30 = mask(&mut rng, 0.30);

    // 128 x 128 output grid, rotated by 17 degrees and zoomed by 1.37 around
    // the image centre.
    let side = (NPTS as f64).sqrt() as usize;
    let (cy, cx) = (NROW as f64 / 2.0, NCOL as f64 / 2.0);
    let (s, c) = 17f64.to_radians().sin_cos();
    let step = 1.37;
    let mut regular = Vec::with_capacity(NPTS);
    let mut aligned = Vec::with_capacity(NPTS);
    let mut integer = Vec::with_capacity(NPTS);
    for i in 0..side {
        for j in 0..side {
            let u = (i as f64 - side as f64 / 2.0) * step;
            let v = (j as f64 - side as f64 / 2.0) * step;
            regular.push((cy + c * u - s * v, cx + s * u + c * v));
            aligned.push((cy + (i as f64 - side as f64 / 2.0), cx + v));
            integer.push((cy + (i as f64 - side as f64 / 2.0), cx + (j as f64 - side as f64 / 2.0)));
        }
    }

    let span_r = NROW as f64 - 2.0 * MARGIN;
    let span_c = NCOL as f64 - 2.0 * MARGIN;
    let random = (0..NPTS)
        .map(|_| (MARGIN + span_r * rng.next_f64(), MARGIN + span_c * rng.next_f64()))
        .collect();

    // Positions within 5 pixels of a border; one in four is on an integer
    // row or column, where zero-weight taps may lie outside the array.
    let band = |rng: &mut Rng, n: usize| {
        let d = -0.5 + 5.0 * rng.next_f64();
        if rng.next_f64() < 0.5 { d } else { n as f64 - 1.0 - d }
    };
    let border = (0..NPTS)
        .map(|k| {
            let along = |rng: &mut Rng, n: usize| (n as f64 - 1.0) * rng.next_f64();
            let (mut y, mut x) = if k % 2 == 0 {
                (band(&mut rng, NROW), along(&mut rng, NCOL))
            } else {
                (along(&mut rng, NROW), band(&mut rng, NCOL))
            };
            if k % 4 < 2 {
                if k % 8 < 4 { y = y.round() } else { x = x.round() }
            }
            (y, x)
        })
        .collect();

    Fixture { data, mask2, mask10, mask30, regular, random, aligned, integer, border }
}

// -----------------------------------------------------------------------------
// Benchmark bodies
// -----------------------------------------------------------------------------

/// Interpolates every position with the given strategies and policy.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn sweep<I, IM, BC, P>(
    interp: &I,
    buf: &mut [f64],
    arr_in: &GxArrayView<'_, f64>,
    out: &mut [f64],
    mask_out: &mut [u8],
    pts: &[(f64, f64)],
    input_mask: IM,
    bounds: BC,
    policy: P,
) where
    I: GxArrayViewInterpolator,
    IM: GxArrayViewInterpolatorInputMaskStrategy,
    BC: GxArrayViewInterpolatorBoundsCheckStrategy,
    P: Policy,
{
    let n = pts.len();
    let mut arr_out = GxArrayViewMut::new(out, 1, 1, n);
    let mut mask_view = GxArrayViewMut::new(mask_out, 1, 1, n);
    let mut context = ctx!(input_mask, BinaryOutputMask { mask: &mut mask_view }, bounds, policy);
    for (k, &(y, x)) in pts.iter().enumerate() {
        let _ = interp.array1_interp2(
            buf, black_box(y), black_box(x), k, arr_in, &mut arr_out, -1.0, &mut context,
        );
    }
    black_box(&arr_out.data);
}

fn bench_interpolator<I: GxArrayViewInterpolator, P: Policy>(
    c: &mut Criterion,
    fx: &Fixture,
    name: &str,
    interp: &I,
    policy_name: &str,
    policy: P,
) {
    let mut group = c.benchmark_group(format!("interp/{name}"));
    group.throughput(Throughput::Elements(NPTS as u64));

    let arr_in = GxArrayView::new(&fx.data, 1, NROW, NCOL);
    let m2 = GxArrayView::new(&fx.mask2, 1, NROW, NCOL);
    let m10 = GxArrayView::new(&fx.mask10, 1, NROW, NCOL);
    let m30 = GxArrayView::new(&fx.mask30, 1, NROW, NCOL);
    let mut out = vec![0.0f64; NPTS];
    let mut mask_out = vec![0u8; NPTS];
    let mut buf = interp.allocate_kernel_buffer();

    let id = |scenario: &str| BenchmarkId::from_parameter(format!("{scenario}/{policy_name}"));

    group.bench_function(id("regular"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.regular,
        NoInputMask, BoundsCheck, policy)));
    group.bench_function(id("random"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.random,
        NoInputMask, BoundsCheck, policy)));
    group.bench_function(id("regular_masked"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.regular,
        BinaryInputMask { mask: &m2 }, BoundsCheck, policy)));
    group.bench_function(id("regular_mask30"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.regular,
        BinaryInputMask { mask: &m30 }, BoundsCheck, policy)));
    group.bench_function(id("aligned_mask10"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.aligned,
        BinaryInputMask { mask: &m10 }, BoundsCheck, policy)));
    group.bench_function(id("integer_mask10"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.integer,
        BinaryInputMask { mask: &m10 }, BoundsCheck, policy)));
    group.bench_function(id("border"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.border,
        NoInputMask, BoundsCheck, policy)));
    group.bench_function(id("border_mask10"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.border,
        BinaryInputMask { mask: &m10 }, BoundsCheck, policy)));
    group.bench_function(id("nocheck_mask10"), |b| b.iter(|| sweep(
        interp, &mut buf, &arr_in, &mut out, &mut mask_out, &fx.regular,
        BinaryInputMask { mask: &m10 }, NoBoundsCheck, policy)));

    group.finish();
}

fn bspline<const N: usize>() -> GxBSplineInterpolator<N> {
    let args = GxBSplineInterpolatorArgs { epsilon: 1e-6, mask_influence_threshold: 1e-3 };
    let mut interp = GxBSplineInterpolator::<N>::new(&args);
    interp.initialize().expect("B-spline initialisation failed");
    interp
}

fn bench_all<P: Policy>(c: &mut Criterion, fx: &Fixture, policy_name: &str, policy: P) {
    let no_args = GxArrayViewInterpolatorNoArgs {};
    bench_interpolator(c, fx, "nearest", &GxNearestInterpolator::new(&no_args), policy_name, policy);
    bench_interpolator(c, fx, "linear", &GxLinearInterpolator::new(&no_args), policy_name, policy);
    bench_interpolator(c, fx, "cubic", &GxOptimizedBicubicInterpolator::new(&no_args), policy_name, policy);
    bench_interpolator(c, fx, "bspline3", &bspline::<3>(), policy_name, policy);
    bench_interpolator(c, fx, "bspline5", &bspline::<5>(), policy_name, policy);
    bench_interpolator(c, fx, "bspline7", &bspline::<7>(), policy_name, policy);
    bench_interpolator(c, fx, "bspline9", &bspline::<9>(), policy_name, policy);
    bench_interpolator(c, fx, "bspline11", &bspline::<11>(), policy_name, policy);
}

fn benches(c: &mut Criterion) {
    let fx = fixture();
    bench_all(c, &fx, "effective", Effective);
    bench_all(c, &fx, "full", Full);
}

criterion_group! {
    name = interp;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = benches
}
criterion_main!(interp);
