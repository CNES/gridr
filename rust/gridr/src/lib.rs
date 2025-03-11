#![warn(missing_docs)]
//! Crate doc
mod core;
mod pyapi;

// Instead of declaring "mod lib_py" from another file you can also directly
// define the pymodule here ; see bellow.
//use pyo3::prelude::*;
// 
// /// A Python module implemented in Rust. The name of this function must match
// /// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
// /// import the module.
//#[pymodule]
//mod _mylib {
//    use pyo3::prelude::*;
//    use crate::lib_core::mymodule as core_mymodule;
//    
//    /// Formats the sum of two numbers as string.
//    #[pyfunction]
//    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//        core_mymodule::sum_a_b_str(a as f64,b as f64)
//        //Ok((a + b).to_string())
//    }
//}
