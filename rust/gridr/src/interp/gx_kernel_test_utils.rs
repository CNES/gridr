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

//! Test helpers shared by the kernel unit tests.
//!
//! [`check_native_weights`] validates a weight function on its native
//! support against the continuous kernel it implements. Reference kernels
//! that have no public scalar implementation in the crate are defined here.
#![cfg(test)]

/// Weight function on a native support: phase `t`, `K` weights in inverted
/// order (`weights[i]` is the weight of tap `first + K - 1 - i`).
pub type WeightFn = fn(f64, &mut [f64]);

/// Linear kernel `max(0, 1 - |x|)`.
pub fn phi_linear(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// Keys cubic convolution kernel (`a = -0.5`), written in factored form so
/// that it stays accurate next to `|x| = 1` and `|x| = 2`.
pub fn phi_keys(x: f64) -> f64 {
    let x = x.abs();
    if x <= 1.0 {
        (x - 1.0) * (1.5 * x * x - x - 1.0)
    } else if x < 2.0 {
        let d = x - 2.0;
        -0.5 * (x - 1.0) * d * d
    } else {
        0.0
    }
}

/// Phases used by [`check_native_weights`]: a regular grid of `[0, 1)` plus
/// the values next to both ends, where cancellation errors show up.
fn phases() -> Vec<f64> {
    let mut v: Vec<f64> = (0..200).map(|i| i as f64 / 200.0).collect();
    v.extend_from_slice(&[1e-15, 1e-12, 1e-9, 1.0 - 1e-9, 1.0 - 1e-12, 1.0 - f64::EPSILON]);
    v
}

/// Checks a weight function on its even native support of `k` taps
/// (anchor = `floor(x)`, phase `t` in `[0, 1)`) against the continuous kernel
/// `phi`, whose support is exactly `k` samples wide.
///
/// For every phase:
/// - each weight matches `phi(t - d)`, `d` being the tap offset from the
///   anchor, within `tol` times the largest weight;
/// - the weights sum to `sum` within `tol` relative;
/// - at `t = 0`, a weight is an exact zero if and only if `phi` is;
/// - for `t` in `(0, 1)`, no weight is an exact zero. This is the kernel
///   contract the core relies on: the conservative mask policy ignores
///   zero-weight samples, and the anchor tap must never vanish.
pub fn check_native_weights(w: WeightFn, k: usize, phi: fn(f64) -> f64, sum: f64, tol: f64) {
    assert!(k % 2 == 0, "check_native_weights expects an even support");
    let left = (k - 1) / 2;
    let mut buf = vec![f64::NAN; k];
    for t in phases() {
        w(t, &mut buf);
        let scale = buf.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for i in 0..k {
            let d = (k - 1 - i) as f64 - left as f64;
            let expected = phi(t - d);
            let err = (buf[i] - expected).abs();
            assert!(
                err <= tol * scale,
                "t={t:e} weights[{i}] (offset {d}): got {:e}, expected {expected:e}, err {err:.2e}",
                buf[i]
            );
            if t == 0.0 {
                assert_eq!(
                    buf[i] == 0.0, expected == 0.0,
                    "t=0 weights[{i}] (offset {d}): zero pattern differs ({:e} vs {expected:e})",
                    buf[i]
                );
            } else {
                assert!(
                    buf[i] != 0.0,
                    "t={t:e} weights[{i}] (offset {d}): exact zero inside the support",
                );
            }
        }
        let total: f64 = buf.iter().sum();
        assert!((total - sum).abs() <= tol * sum, "t={t:e}: weights sum to {total}, expected {sum}");
    }
}
