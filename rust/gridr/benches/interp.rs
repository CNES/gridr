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

//! Micro-benchmark of `array1_interp2` for every interpolator.
//!
//! ```text
//! cargo bench -p gridr --bench interp -- --save-baseline v0.7
//! cargo bench -p gridr --bench interp -- --baseline v0.7
//! cargo bench -p gridr --bench interp -- linear
//! ```
//!
//! Scenarios (each call interpolates `NPTS` target positions, throughput is
//! reported in interpolated points per second):
//!
//! - `regular`: positions on a rotated and slightly zoomed regular grid, as in
//!   a typical resampling. Memory accesses are local, phases vary smoothly.
//! - `random`: uniformly random positions. Phases are unpredictable and
//!   memory accesses scattered, which exposes branch mispredictions in the
//!   weight computation and cache misses.
//! - `regular_masked`: `regular` with a binary input mask holding about 2 %
//!   of invalid pixels and a binary output mask, which exercises the masked
//!   convolution paths (window count, weighted validation).
//!
//! All positions are kept away from the borders
//!
//! The B-Spline variants do not integrate the prefiltering

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use gridr::gx_array::{GxArrayView, GxArrayViewMut};
use gridr::interp::gx_array_view_interp::{
    BinaryInputMask, BinaryOutputMask, BoundsCheck, GxArrayViewInterpolationContext,
    GxArrayViewInterpolator, GxArrayViewInterpolatorNoArgs, NoInputMask, NoOutputMask,
};
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
    mask: Vec<u8>,
    regular: Vec<(f64, f64)>,
    random: Vec<(f64, f64)>,
}

fn fixture() -> Fixture {
    let mut rng = Rng(0xBE7C_4000_0000_0001);
    let data: Vec<f64> = (0..NROW * NCOL).map(|_| 1000.0 * rng.next_f64()).collect();
    let mask: Vec<u8> = (0..NROW * NCOL)
        .map(|_| if rng.next_f64() < 0.02 { 0 } else { 1 })
        .collect();

    // 128 x 128 output grid, rotated by 17 degrees and zoomed by 1.37 around
    // the image centre.
    let side = (NPTS as f64).sqrt() as usize;
    let (cy, cx) = (NROW as f64 / 2.0, NCOL as f64 / 2.0);
    let (s, c) = 17f64.to_radians().sin_cos();
    let step = 1.37;
    let mut regular = Vec::with_capacity(NPTS);
    for i in 0..side {
        for j in 0..side {
            let u = (i as f64 - side as f64 / 2.0) * step;
            let v = (j as f64 - side as f64 / 2.0) * step;
            regular.push((cy + c * u - s * v, cx + s * u + c * v));
        }
    }

    let span_r = NROW as f64 - 2.0 * MARGIN;
    let span_c = NCOL as f64 - 2.0 * MARGIN;
    let random = (0..NPTS)
        .map(|_| (MARGIN + span_r * rng.next_f64(), MARGIN + span_c * rng.next_f64()))
        .collect();

    Fixture { data, mask, regular, random }
}

// -----------------------------------------------------------------------------
// Benchmark bodies
// -----------------------------------------------------------------------------

fn bench_interpolator<I: GxArrayViewInterpolator>(
    c: &mut Criterion,
    fx: &Fixture,
    name: &str,
    interp: &I,
) {
    let mut group = c.benchmark_group(format!("interp/{name}"));
    group.throughput(Throughput::Elements(NPTS as u64));

    let arr_in = GxArrayView::new(&fx.data, 1, NROW, NCOL);
    let mut out = vec![0.0f64; NPTS];
    let mut buf = interp.allocate_kernel_buffer();

    // unmasked regular and random grid
    for (scenario, pts) in [("regular", &fx.regular), ("random", &fx.random)] {
        group.bench_function(BenchmarkId::from_parameter(scenario), |b| {
            b.iter(|| {
                let mut arr_out = GxArrayViewMut::new(&mut out, 1, 1, NPTS);
                let mut ctx =
                    GxArrayViewInterpolationContext::new(NoInputMask, NoOutputMask, BoundsCheck);
                // Note : we use black_box to avoid compilation time optimization
                // that could biased the benchmark.
                for (k, &(y, x)) in pts.iter().enumerate() {
                    let _ = interp.array1_interp2(
                        &mut buf, black_box(y), black_box(x), k,
                        &arr_in, &mut arr_out, -1.0, &mut ctx,
                    );
                }
                black_box(&arr_out.data);
            })
        });
    }

    // regular grid with mask on data
    let mask_in = GxArrayView::new(&fx.mask, 1, NROW, NCOL);
    let mut mask_out = vec![0u8; NPTS];
    group.bench_function(BenchmarkId::from_parameter("regular_masked"), |b| {
        b.iter(|| {
            let mut arr_out = GxArrayViewMut::new(&mut out, 1, 1, NPTS);
            let mut mask_view = GxArrayViewMut::new(&mut mask_out, 1, 1, NPTS);
            let mut ctx = GxArrayViewInterpolationContext::new(
                BinaryInputMask { mask: &mask_in },
                BinaryOutputMask { mask: &mut mask_view },
                BoundsCheck,
            );
            for (k, &(y, x)) in fx.regular.iter().enumerate() {
                let _ = interp.array1_interp2(
                    &mut buf, black_box(y), black_box(x), k,
                    &arr_in, &mut arr_out, -1.0, &mut ctx,
                );
            }
            black_box(&arr_out.data);
        })
    });

    group.finish();
}

fn bspline<const N: usize>() -> GxBSplineInterpolator<N> {
    let args = GxBSplineInterpolatorArgs { epsilon: 1e-6, mask_influence_threshold: 1e-3 };
    let mut interp = GxBSplineInterpolator::<N>::new(&args);
    interp.initialize().expect("B-spline initialisation failed");
    interp
}

fn benches(c: &mut Criterion) {
    let fx = fixture();
    let no_args = GxArrayViewInterpolatorNoArgs {};
    bench_interpolator(c, &fx, "nearest", &GxNearestInterpolator::new(&no_args));
    bench_interpolator(c, &fx, "linear", &GxLinearInterpolator::new(&no_args));
    bench_interpolator(c, &fx, "cubic", &GxOptimizedBicubicInterpolator::new(&no_args));
    bench_interpolator(c, &fx, "bspline3", &bspline::<3>());
    bench_interpolator(c, &fx, "bspline5", &bspline::<5>());
    bench_interpolator(c, &fx, "bspline7", &bspline::<7>());
    bench_interpolator(c, &fx, "bspline9", &bspline::<9>());
    bench_interpolator(c, &fx, "bspline11", &bspline::<11>());
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
