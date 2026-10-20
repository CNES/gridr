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

//! Integration tests of the interpolators (`gridr::interp`).
//!
//! Cargo builds this directory as a single test crate, `interp`:
//!
//! ```text
//! cargo test -p gridr --test interp                    # all tests
//! cargo test -p gridr --test interp array1_interp2::   # one module
//! ```

mod common;

mod array1_interp2;
mod margins;
