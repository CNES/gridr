#![warn(missing_docs)]
//! GridR core error's definitions

#[derive(Debug, PartialEq)]
pub enum GxError {
    /// Two options must be both Some or both None.
    OptionsMismatch { field1: &'static str, field2: &'static str,},
    
    /// Two arrays must have the same shapes.
    ShapesMismatch { field1: &'static str, field2: &'static str,},
    
    /// Resolution factor must be strictly positiv
    ZeroResolution,
    
    /// Generic error with a message
    ErrMessage(String),
}

impl std::error::Error for GxError {}

impl std::fmt::Display for GxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GxError::OptionsMismatch { field1, field2 } => {
                write!(
                    f,
                    "Options '{}' and '{}' must be both Some or both None",
                    field1, field2
                )
            },
            GxError::ShapesMismatch { field1, field2 } => {
                write!(
                    f,
                    "Both '{}' and '{}' must have the same shape",
                    field1, field2
                )
            },
            GxError::ZeroResolution => write!(f, "Resolution cannot be zero !"),
            GxError::ErrMessage(msg) => write!(f, "{msg}"),
        }
    }
}

