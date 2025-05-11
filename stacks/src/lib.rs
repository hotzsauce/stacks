use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusty_stacks::IncStack;

pub mod record;
use record::PyRecord;

pub type X = f64;
pub type Y = f64;

#[pyclass]
struct OfferStack {
    inner: IncStack<X, Y>,
}

#[pymethods]
impl OfferStack {
    #[new]
    fn new(x: Vec<X>, y: Vec<Y>) -> PyResult<Self> {
        IncStack::try_from_vectors(x, y)
            .map(|inner| Self { inner })
            .map_err(|err| PyRuntimeError::new_err(format!("{}", err)))
    }

    // dunder methods

    fn __repr__(&self) -> String {
        // impractical for large Stacks but sufficient for prototyping lol
        format!("Stack({:?}, {:?})", self.inner.x, self.inner.y)
    }

    // interface methods

    fn cumulate(&self) -> Vec<X> {
        self.inner.cumulate()
    }

    #[getter]
    fn masses(&self) -> &[X] {
        &self.inner.x
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    #[getter]
    fn levels(&self) -> &[Y] {
        &self.inner.y
    }
}

#[pymodule]
fn stacks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRecord>()?;
    m.add_class::<OfferStack>()?;
    Ok(())
}
