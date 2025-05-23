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