// use crate::stack::{Decreasing, Decreasing, OrderMarker};
use std::fmt;

/// Represents possible validation and construction errors for a `Stack`.
///
/// # Variants
/// - `LengthMismatch(usize, usize)`: Input vectors `x` and `y` have differing lengths.  
///   - Parameters: `(x_len, y_len)` indicating each vector’s length.
/// - `NegativeMass`: Encountered a negative mass value in `x`.  
///   Indicates violation of the non-negativity invariant.
/// - `NonMonotonicLevels`: The levels vector `y` failed strict monotonicity  
///   (increasing or decreasing) as required by the order marker.
#[derive(Debug, PartialEq)]
pub enum Error {
    LengthMismatch(usize, usize),
    NegativeMass,
    // NonMonotonicLevels<O: OrderMarker>(O), not sure how to work this
    NonMonotonicLevels,
}

impl fmt::Display for Error {
    /// Formats the error for human-readable output.
    ///
    /// # Behavior
    /// - `LengthMismatch(x_len, y_len)`:  
    ///   Prints "Length mismatch: x has length {x_len}, y has length {y_len}"  
    /// - `NegativeMass`: Prints "Encountered negative mass"  
    /// - `NonMonotonicLevels`: Prints "Levels vector is not monotonic"
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::LengthMismatch(x_len, y_len) => {
                write!(
                    f,
                    "Length mismatch: x has length {}, y has length {}",
                    x_len, y_len
                )
            }
            Error::NegativeMass => {
                write!(f, "Encountered negative mass")
            }
            Error::NonMonotonicLevels => {
                write!(f, "Levels vector is not monotonic")
            }
        }
    }
}

/// Marker implementation integrating with the standard `Error` trait.
impl std::error::Error for Error {}
