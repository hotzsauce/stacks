use crate::X;
use pyo3::prelude::*;

use rusty_stacks::{Record, SourceID};

#[pyclass(name = "Record")]
#[derive(Clone)]
pub struct PyRecord {
    inner: Record<X>,
}

#[pymethods]
impl PyRecord {
    // instantiation methods

    #[new]
    fn new() -> Self {
        let inner = Record::empty();
        Self { inner }
    }

    #[staticmethod]
    fn from_entry(src: SourceID, mass: X) -> Self {
        let inner = Record::new(src, mass);
        Self { inner }
    }

    // dunder methods

    /// Returns the number of provenance entries in the `Record`
    ///
    /// # Returns
    /// - `0` for `Empty`
    /// - `1` for `One`
    /// - The length of the vector for `Many`
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// User-friendly print
    fn __str__(&self) -> String {
        // format!("Record({:?}, {:?})", self.inner.x, self.inner.y)
        match &self.inner {
            Record::Empty => "Record()".into(),
            Record::One(rec) => format!("Record({}, {})", rec.0, rec.1),
            Record::Many(vec) => format!("Record({:?})", vec),
        }
    }

    // interface methods

    /// Add an entry to the `Record`
    ///
    /// # Parameters
    /// - `src`: SourceID
    /// - `mass`: X
    fn append(&mut self, src: SourceID, mass: X) {
        self.inner.push(src, mass);
    }

    /// Append all entries from another `Record` to this one.
    ///
    /// # Parameters
    /// - `other`: Another `Record` whose entries will be merged into `self.`
    fn extend(&mut self, other: PyRecord) {
        self.inner.extend(other.inner);
    }

    /// Checks whether the `Record` contains exactly one entry
    ///
    /// # Returns
    /// - `true` if `self` is `One`.
    /// - `false` otherwise.
    fn is_one(&self) -> bool {
        self.inner.is_one()
    }
}
