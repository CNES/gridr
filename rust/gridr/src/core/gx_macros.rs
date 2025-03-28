#![warn(missing_docs)]
//! GridR macros

#[macro_export]
macro_rules! assert_options_match {
    ($opt1:expr, $opt2:expr, $err:expr) => {{
        match ($opt1.is_some(), $opt2.is_some()) {
            (true, true) | (false, false) => {}
            _ => return Err($err),
        }
    }};
}