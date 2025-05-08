pub mod error;
pub mod provenance;
pub mod stack;
pub mod transform;

pub type SourceID = u16;

pub use error::Error;
pub use provenance::{Provenance, record::Record};
pub use stack::{
    CumulativeSum, DecStack, IncStack, IteratorCumulativeSum, OrderMarker, SparseIter, Stack,
};
