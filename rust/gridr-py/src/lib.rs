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
//! PyO3 binding layer for [gridr](https://github.com/CNES/gridr), exposed
//! to Python as the private `_libgridr` extension module.
//!
//! Exposes `__pyapi_native_version__` (this crate's own `CARGO_PKG_VERSION`)
//! and `__native_version__` (the `gridr` engine crate's version) at the
//! Python level, so a given wheel's embedded Rust versions can always be
//! checked at runtime - see py_bindings.rs.
pub mod pyapi;

