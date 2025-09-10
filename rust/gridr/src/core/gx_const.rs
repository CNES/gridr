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
//! # GridR constants definition
//!

/// Tolerance used when comparing two `f64` values for approximate equality.
///
/// This constant defines the acceptable maximum absolute difference between two `f64`
/// values for them to be considered equal in floating-point comparisons. It is typically
/// used in relative or absolute error checks to account for precision limitations
/// inherent in floating-point arithmetic.
///
/// # Value `1e-5` (i.e., 0.00001) is a reasonable default for many geometric and scientific
/// computations where a tolerance within five decimal places is acceptable.
pub const F64_TOLERANCE: f64 = 1e-5;