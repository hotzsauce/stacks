use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rusty_stacks::IncStack;

#[pyclass]
struct OfferStack {
    inner: IncStack<f64, f64>,
}

#[pymethods]
impl OfferStack {
    #[new]
    fn new(x: Vec<f64>, y: Vec<f64>) -> PyResult<Self> {
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

    fn cumulate(&self) -> Vec<f64> {
        self.inner.cumulate()
    }

    #[getter]
    fn x(&self) -> &[f64] {
        &self.inner.x
    }

    #[getter]
    fn y(&self) -> &[f64] {
        &self.inner.y
    }
}

#[pymodule]
fn stacks(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OfferStack>()?;
    Ok(())
}
