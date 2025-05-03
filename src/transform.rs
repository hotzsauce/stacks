use std::cmp::PartialOrd;
use std::marker::PhantomData;

use num_traits::{Num, Zero};

use crate::provenance::{Provenance, record::Record};
use crate::stack::{OrderMarker, Stack};

/// Trait defining a transformation on a `Stack`, producing a new `Stack` of the same ordering.
///
/// # Type Parameters
/// - `X`: Numeric type for mass values.  
/// - `Y`: Numeric type for level values.  
/// - `O`: Marker type implementing `OrderMarker` that enforces `y` ordering.
pub trait Transform<X, Y, O> {
    /// Applies this transformation to `stack`, returning a new `Stack` instance.
    ///
    /// # Parameters
    /// - `stack`: Reference to the original `Stack` to transform.
    ///
    /// # Returns
    /// A new `Stack<X, Y, O>` containing transformed `x`, `y`, and provenance data.
    fn transform(&self, stack: &Stack<X, Y, O>) -> Stack<X, Y, O>;
}

/// Transformation that clips a `Stack` by a cumulative mass threshold.
///
/// Retains all entries up to (but not including) the first cumulative mass in `values`
/// exceeding `threshold`.
///
/// # Type Parameters
/// - `T`: Numeric type for the threshold and cumulative mass values.
pub struct Clip<'a, T> {
    threshold: T,
    values: &'a [T],
}

impl<'a, T> Clip<'a, T> {
    /// Creates a new `Clip` transformation.
    ///
    /// # Parameters
    /// - `threshold`: Mass threshold at which to clip the stack.
    ///   Any cumulative mass value `> threshold` triggers the cut.
    /// - `values`: Slice of cumulative mass values corresponding to the input `Stack`.
    pub fn new(threshold: T, values: &'a [T]) -> Self {
        Self { threshold, values }
    }
}

impl<'a, X, Y, O> Transform<X, Y, O> for Clip<'a, X>
where
    X: Copy + Num + PartialOrd + Zero + std::iter::Sum + Default,
    Y: Copy + Num + PartialOrd,
    O: OrderMarker,
{
    /// Applies the clip operation to `stack`.
    ///
    /// Finds the first index where `values[i] > threshold` and delegates
    /// to `chop_at` to produce the truncated `Stack`.
    fn transform(&self, stack: &Stack<X, Y, O>) -> Stack<X, Y, O> {
        let index = self.values.iter().position(|&v| v > self.threshold);
        chop_at(stack, index)
    }
}

// `Projection` creates a new `Stack`

/// Project the `Stack` onto a new levels vector
pub struct Projection<'a, T> {
    range: &'a [T],
}

impl<'a, T> Projection<'a, T> {
    pub fn new(range: &'a [T]) -> Self {
        Self { range }
    }
}

impl<'a, X, Y, O> Transform<X, Y, O> for Projection<'a, Y>
where
    X: Copy + Num + PartialOrd + Zero + std::iter::Sum + Default,
    Y: Copy + Num + PartialOrd,
    O: OrderMarker,
{
    fn transform(&self, stack: &Stack<X, Y, O>) -> Stack<X, Y, O> {
        let map = digitize_sorted::<Y, O>(&stack.y, &self.range, true);
        println!("{:?}", &map);

        if let Some(&last_index) = map.last() {
            let n_out = last_index + 1;

            let pool = stack.prov.pool.clone();
            let stack_records = &stack.prov.records;

            let mut x = vec![X::zero(); n_out];
            let mut records = vec![Record::default(); n_out];

            for (i, &o) in map.iter().enumerate() {
                x[o] = x[o] + stack.x[i];
                records[o].extend(stack_records[i].clone());
            }

            let prov = Provenance { records, pool };
            Stack {
                x,
                y: self.range.to_owned(),
                prov,
                _ord: PhantomData,
            }
        } else {
            Stack::new()
        }
    }
}

fn digitize_sorted<T, O>(data: &[T], bins: &[T], right: bool) -> Vec<usize>
where
    T: PartialOrd + Copy,
    O: OrderMarker,
{
    let n = bins.len();
    let mut out = Vec::with_capacity(bins.len());

    for x in data {
        // partition_point returns the first index `i` such that the
        // predicate is false; [0..i) all satisfy it.
        let i = if right {
            // count of bins ≤ x
            bins.partition_point(|b| O::is_in_order(&b, &x))
        } else {
            // count of bins < x
            bins.partition_point(|b| !O::is_in_order(&x, &b))
        };
        if i == n {
            // drop all the trailing data once we run off the end of the bins
            break;
        }
        out.push(i);
    }

    // this route produces the same output but I'm unsure about the relative performance

    // let n = bins.len();
    // let mut j = 0usize;
    // let mut out = Vec::with_capacity(n);
    //
    // for &x in data {
    //     if right {
    //         while j < bins.len() && O::is_in_order(&bins[j], &x) {
    //             j += 1;
    //         }
    //     } else {
    //         while j < bins.len() && !O::is_in_order(&x, &bins[j]) {
    //             j += 1;
    //         }
    //     }
    //     if j == n {
    //         break;
    //     }
    //     out.push(j);
    // }
    // out
    out
}

/// Transformation that truncates a `Stack` by a level threshold.
///
/// Retains all entries up to the point where levels satisfy the order marker
/// comparison against `threshold`.
///
/// # Type Parameters
/// - `T`: Numeric type for the threshold and level values.
pub struct Truncate<'a, T> {
    threshold: T,
    values: &'a [T],
}

impl<'a, T> Truncate<'a, T> {
    /// Creates a new `Truncate` transformation.
    ///
    /// # Parameters
    /// - `threshold`: Level threshold at which to truncate.
    /// - `values`: Slice of level values corresponding to the input `Stack`.
    pub fn new(threshold: T, values: &'a [T]) -> Self {
        Self { threshold, values }
    }
}

impl<'a, X, Y, O> Transform<X, Y, O> for Truncate<'a, Y>
where
    X: Copy + Num + PartialOrd + Zero + std::iter::Sum + Default,
    Y: Copy + Num + PartialOrd,
    O: OrderMarker,
{
    /// Applies the truncate operation to `stack`.
    ///
    /// Locates the first index where `O::is_in_order(&threshold, &values[i])`
    /// is true, then calls `chop_at` to build the resulting `Stack`.
    fn transform(&self, stack: &Stack<X, Y, O>) -> Stack<X, Y, O> {
        let index = self
            .values
            .iter()
            .position(|v| O::is_in_order(&self.threshold, v));
        chop_at(stack, index)
    }
}

// Utility functions

/// Internal utility to construct a truncated `Stack` at a given index.
///
/// # Parameters
/// - `stack`: Reference to the original `Stack<X, Y, O>`.
/// - `index`: `Option<usize>` indicating the cut-off position:
///    - `Some(i)`: Retain elements `[0..i]`.
///    - `None`: Retain the entire stack.
///
/// # Returns
/// A new `Stack<X, Y, O>` containing cloned `x`, `y`, and provenance data
/// up to the specified `index`.
///
/// # Type Constraints
/// - `X`: Must implement `Copy`, `Num`, `PartialOrd`, `Zero`, and `Sum`.
/// - `Y`: Must implement `Copy`, `Num`, and `PartialOrd`.
/// - `O`: Must implement `OrderMarker`.
fn chop_at<X, Y, O>(stack: &Stack<X, Y, O>, index: Option<usize>) -> Stack<X, Y, O>
where
    X: Copy + Num + PartialOrd + Zero + std::iter::Sum + Default,
    Y: Copy + Num + PartialOrd,
    O: OrderMarker,
{
    // if no cutoff, we copy the full length
    let len = index.unwrap_or(stack.len());
    if len == 0 {
        return Stack::new();
    }

    // avoid multiple allocations by reserving capacity once
    let mut x = Vec::with_capacity(len);
    x.extend_from_slice(&stack.x[..len]);

    let mut y = Vec::with_capacity(len);
    y.extend_from_slice(&stack.y[..len]);

    let mut records = Vec::with_capacity(len);
    records.extend_from_slice(&stack.prov.records[..len]);

    let pool = stack.prov.pool.clone();
    let prov = Provenance { records, pool };

    Stack {
        x,
        y,
        prov,
        _ord: PhantomData,
    }
}
