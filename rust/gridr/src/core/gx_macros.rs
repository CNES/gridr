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
//! GridR macros

/// Asserts that two `Option` values are either both `Some` or both `None`.
///
/// This macro is useful for enforcing consistency between optional arguments
/// that must be provided together or omitted together.
///
/// # Arguments
/// - `$opt1` - The first `Option` to check.
/// - `$opt2` - The second `Option` to check.
/// - `$err` - The error to return if the options do not match.
///
/// # Example
/// ```rust
/// assert_options_match!(some_option, other_option, MyError::new("Mismatch"));
/// ```
///
/// # Behavior
/// - If both options are `Some`, it passes silently.
/// - If both are `None`, it passes silently.
/// - If only one is `Some`, it returns the given error.
#[macro_export]
macro_rules! assert_options_match {
    ($opt1:expr, $opt2:expr, $err:expr) => {{
        match ($opt1.is_some(), $opt2.is_some()) {
            (true, true) | (false, false) => {}
            _ => return Err($err),
        }
    }};
}

/// Asserts that at most one of two `Option` values is `Some`.
///
/// This macro enforces exclusivity between two optional arguments,
/// allowing at most one to be present. If both are `Some`, it returns the provided error.
///
/// # Arguments
/// - `$opt1` - The first `Option` to check.
/// - `$opt2` - The second `Option` to check.
/// - `$err` - The error to return if both options are `Some`.
///
/// # Example
/// ```rust
/// assert_options_exclusive!(opt_a, opt_b, MyError::new("Only one option may be set"));
/// ```
///
/// # Behavior
/// - If both options are `None`, it passes silently.
/// - If exactly one is `Some`, it passes silently.
/// - If both are `Some`, it returns the given error.
#[macro_export]
macro_rules! assert_options_exclusive {
    ($opt1:expr, $opt2:expr, $err:expr) => {{
        match ($opt1.is_some(), $opt2.is_some()) {
            (true, false) | (false, true) | (false, false) => {}
            (true, true) => return Err($err),
        }
    }};
}