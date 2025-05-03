// methods that already exist in the python implementation:
// [x] clip(X) -> Stack<X, Y>
// [x] cumulate() -> Vec<X>
// [ ] mean() -> [UNKONOWN?]
// [ ] project_onto(Vec<Y>) -> Stack<X, Y>
// [x] rename(&str) -> Stack<X, Y> {here, called `alias`}
// [x] truncate(Y) -> Stack<X, Y>
// [x] total() -> X
// [x] total_above(Y) -> X
// [x] total_below(Y) -> X
// [ ] X() -> Vec<X> {should be renamed to mass()}
// [ ] Y() -> Vec<Y> {should be renamed to level()}
//
// [x] __add__(Stack<X, Y>, Stack<W, Z>) -> Stack<(X, W), (Y, Z)>
// [ ] __radd__(Stack<X, Y>, Stack<W, Z>) -> Stack<(W, X), (Z, Y)>
// [ ] __ge__(Stack<X, Y>, Stack<W, Z>) -> bool
// [x] __len__(Stack<X, Y>) -> usize
//
// [ ] wipe(&str) -> Stack<X, Y>
//      I'm not sure what this method name should actually be, but
//      it's meant to "wipe" the existing provenance vector into
//      the single provided source

use either::Either;
use num_traits::{Num, Zero};

use std::cmp::PartialOrd;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Add;

use crate::error::Error;
use crate::provenance::Provenance;
use crate::transform::{Clip, Projection, Transform, Truncate};

/// Marker type indicating that the levels vector `y` of a `Stack` is strictly increasing.
///
/// This zero-sized type is used as the default `OrderMarker` parameter for `Stack`,
/// enforcing the invariant `y[i] < y[i + 1]` for all valid adjacent index pairs.
pub enum Increasing {}

/// Marker type indicating that the levels vector `y` of a `Stack` is strictly decreasing.
///
/// Use this as the `OrderMarker` parameter for `Stack` to enforce
/// `y[i] > y[i + 1]` for each adjacent element in `y`.
pub enum Decreasing {}

/// Trait that defines compile-time markers for the ordering of values in a `Stack`.
///
/// Implementors specify a `LABEL` for runtime matching ("inc" or "dec") and
/// provide the `is_in_order` method to validate strict monotonicity.
///
/// # Associated Constants
///
/// - `LABEL`: Short identifier used at runtime to distinguish the ordering mode.
///
/// # Methods
///
/// - `is_in_order(prev, next) -> bool`
///    Returns `true` if `prev` and `next` satisfy the implementer’s order constraint.
pub trait OrderMarker {
    const LABEL: &'static str; // needed for run-time pattern matching
    fn is_in_order<T: PartialOrd>(prev: &T, next: &T) -> bool;
}

impl OrderMarker for Increasing {
    const LABEL: &'static str = "inc";
    #[inline]
    fn is_in_order<T: PartialOrd>(prev: &T, next: &T) -> bool {
        prev < next
    }
}

impl OrderMarker for Decreasing {
    const LABEL: &'static str = "dec";
    #[inline]
    fn is_in_order<T: PartialOrd>(prev: &T, next: &T) -> bool {
        prev > next
    }
}

/// Represents a 2-tuple of vectors `(x, y)` modeling discrete masses `x`
/// at corresponding levels `y`, with provenance tracking.
///
/// # Type Parameters
///
/// - `X`: Numeric type for mass values. Must implement `Num`, `Zero`, `PartialOrd`, and `Copy`.
/// - `Y`: Numeric type for level values. Must implement `Num`, `PartialOrd`, and `Copy`.
/// - `O`: Ordering marker (`Increasing` or `Decreasing`) controlling strict monotonicity of `y`.
///
/// # Invariants
///
/// 1. `x.len() == y.len()`
/// 2. All elements in `x` are non-negative.
/// 3. The `y` vector is strictly monotonic according to `O`.
///
/// # Fields
///
/// - `x`: `Vec<X>` storing mass values.
/// - `y`: `Vec<Y>` storing level values.
/// - `prov`: `Provenance<X>` tracking sources for each mass entry.
/// - `_ord`: `PhantomData<O>` preserving the compile-time ordering marker.
pub struct Stack<X, Y, O = Increasing> {
    pub x: Vec<X>,
    pub y: Vec<Y>,
    pub prov: Provenance<X>,
    pub _ord: PhantomData<O>,
}

impl<X, Y, O> Stack<X, Y, O>
where
    X: Copy + Num + PartialOrd + Zero + std::iter::Sum + Default,
    Y: Copy + Num + PartialOrd,
    O: OrderMarker,
{
    // instantiation methods

    /// Constructs an empty `Stack` with no mass, levels, or provenance entries.
    ///
    /// # Returns
    ///
    /// A `Stack` instance with:
    /// - `x = []`
    /// - `y = []`
    /// - `prov` initialized empty
    /// - default order marker `O = Increasing`
    pub fn new() -> Self {
        let x = Vec::<X>::new();
        let y = Vec::<Y>::new();

        // let prov = Provenance::from_masses(&x);
        let prov = Provenance::new();
        Self {
            x,
            y,
            prov,
            _ord: PhantomData,
        }
    }

    /// Attempts to build a `Stack` from provided mass and level vectors,
    /// validating length equality, non-negativity, and monotonicity.
    ///
    /// # Parameters
    ///
    /// - `x`: Vector of mass values.
    /// - `y`: Vector of level values.
    ///
    /// # Errors
    ///
    /// - `Error::LengthMismatch(x_len, y_len)`: if `x.len() != y.len()`.
    /// - `Error::NegativeMass`: if any `x` element < 0.
    /// - `Error::NonMonotonicLevels`: if `y` fails strict ordering.
    ///
    /// # Returns
    ///
    /// `Ok(Stack)` on success, containing `x`, `y`, and provenance derived from `x`.
    pub fn try_from_vectors(x: Vec<X>, y: Vec<Y>) -> Result<Self, Error> {
        validate_stack_vectors::<_, _, O>(&x, &y)?;
        let prov = Provenance::from_mass(&x);
        Ok(Self {
            x,
            y,
            prov,
            _ord: PhantomData,
        })
    }

    /// Builds a `Stack` from mass and level vectors and associates all masses
    /// with a single provenance source string.
    ///
    /// # Type Parameters
    ///
    /// - `S`: A type convertible into `String` for the initial source label.
    ///
    /// # Parameters
    ///
    /// - `x`: Vector of mass values.
    /// - `y`: Vector of level values.
    /// - `src`: Initial source identifier.
    ///
    /// # Errors
    ///
    /// Same as `try_from_vectors`, validating invariants before assignment.
    ///
    /// # Returns
    ///
    /// `Ok(Stack)` with provenance recording `src` for each mass element.
    pub fn try_from_vectors_and_source<S>(x: Vec<X>, y: Vec<Y>, src: S) -> Result<Self, Error>
    where
        S: Into<String>,
    {
        validate_stack_vectors::<_, _, O>(&x, &y)?;
        let prov = Provenance::from_source_and_mass(src, &x);

        Ok(Self {
            x,
            y,
            prov,
            _ord: PhantomData,
        })
    }

    // interface methods

    /// Adds a provenance source to an otherwise unlabelled `Stack`.
    ///
    /// If the internal provenance pool is empty, this method registers `src`.
    /// Otherwise, it leaves existing provenance intact.
    ///
    /// # Type Parameters
    ///
    /// - `S`: Convertible into `String` source identifier.
    ///
    /// # Parameters
    ///
    /// - `&mut self`: The `Stack` to label.
    /// - `src`: Identifier for the provenance source.
    ///
    /// # Returns
    ///
    /// A mutable reference to `self`, enabling method chaining.
    pub fn alias<S>(&mut self, src: S) -> &mut Self
    where
        S: Into<String>,
    {
        if self.prov.pool().is_empty() {
            let _ = self.prov.add_source(src);
        }
        self
    }

    /// Produces a truncated `Stack` by mass threshold, keeping all levels
    /// up to the cumulative mass `threshold`.
    ///
    /// # Parameters
    ///
    /// - `&self`: Source `Stack`.
    /// - `threshold`: Mass threshold `X` beyond which levels are discarded.
    ///
    /// # Returns
    ///
    /// A new `Stack` containing only the prefix of `(x, y)` where the
    /// running sum of `x` does not exceed `threshold`.
    pub fn clip(&self, threshold: X) -> Self {
        let cumulative = self.cumulate();
        let clip = Clip::new(threshold, &cumulative);
        clip.transform(&self)
    }

    /// Computes the prefix-sum (cumulative mass) vector of `self.x`.
    ///
    /// # Returns
    ///
    /// A `Vec<X>` of the same length as `x` where each element `i`
    /// is the sum of `x[0] + x[1] + ... + x[i]`.
    pub fn cumulate(&self) -> Vec<X> {
        let n = self.x.len();
        if n == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(n);
        let mut sum = X::zero();

        for &x in &self.x {
            sum = sum + x;
            result.push(sum);
        }

        result
    }

    /// Returns the number of level entries in the `Stack`.
    ///
    /// Equivalent to `self.y.len()`.
    ///
    /// # Returns
    ///
    /// `usize` length of the `y` vector.
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// project the stack onto a levels vector
    pub fn project_onto(&self, range: &[Y]) -> Self {
        // checks to make:
        // - `range` is shorter than `self`
        // - `range` is a valid levels vector
        // - for `O: Increasing`, that the maximal element of `range` is <= self.y.last()
        let proj = Projection::new(range);
        proj.transform(&self)
    }

    /// Computes the total mass of the `Stack`.
    ///
    /// # Returns
    ///
    /// Sum of all elements in `self.x`.
    pub fn total(&self) -> X {
        self.x.iter().fold(X::zero(), |acc, &x| acc + x)
    }

    /// Sums the mass entries corresponding to levels strictly above `threshold`.
    ///
    /// Iteration order depends on `O`:
    /// - For `Increasing`, scans from highest level downward.
    /// - For `Decreasing`, scans from lowest level upward.
    ///
    /// # Parameters
    ///
    /// - `threshold`: Level cutoff (exclusive).
    ///
    /// # Returns
    ///
    /// Total mass `X` above `threshold`.
    pub fn total_above(&self, threshold: Y) -> X {
        let base = self.x.iter().zip(self.y.iter());
        let iter: Either<_, _> = match O::LABEL {
            "inc" => Either::Left(base.rev()),
            "dec" => Either::Right(base),
            _ => unreachable!(),
        };

        iter.take_while(|&(_, y)| *y > threshold)
            .map(|(&x, _)| x)
            .sum()
    }

    /// Sums the mass entries for levels at or below `threshold`.
    ///
    /// Iteration order depends on `O`.
    ///
    /// # Parameters
    ///
    /// - `threshold`: Level cutoff (inclusive).
    ///
    /// # Returns
    ///
    /// Total mass `X` at or below `threshold`.
    pub fn total_below(&self, threshold: Y) -> X {
        let base = self.x.iter().zip(self.y.iter());
        let iter: Either<_, _> = match O::LABEL {
            "inc" => Either::Left(base),
            "dec" => Either::Right(base.rev()),
            _ => unreachable!(),
        };

        iter.take_while(|&(_, y)| *y <= threshold)
            .map(|(&x, _)| x)
            .sum()
    }

    /// Truncates the `Stack` to include only the prefix of `(x, y)`
    /// where `y <= threshold` (for `Increasing`) or `y >= threshold`
    /// (for `Decreasing`).
    ///
    /// # Parameters
    ///
    /// - `threshold`: Level cutoff `Y`.
    ///
    /// # Returns
    ///
    /// A new `Stack` containing only the truncated `(x, y, prov)` data.
    pub fn truncate(&self, threshold: Y) -> Self {
        let headsman = Truncate::new(threshold, &self.y);
        headsman.transform(&self)
    }
}

/// Alias for a `Stack` with strictly increasing levels.
pub type IncStack<X, Y> = Stack<X, Y, Increasing>;

/// Alias for a `Stack` with strictly decreasing levels.
pub type DecStack<X, Y> = Stack<X, Y, Decreasing>;

// `Trait`s for `Stack`s

/// Implements vectorized addition of two `Stack`s with the same ordering.
///
/// - Merges their level vectors into a unique sorted union.
/// - Aligns masses from both operands to the merged levels.
/// - Sums masses for identical levels.
/// - Merges provenance information accordingly.
impl<X, Y, O> Add for Stack<X, Y, O>
where
    X: Copy + Num + PartialOrd + Zero + std::iter::Sum + std::fmt::Debug,
    Y: Copy + Num + PartialOrd + std::fmt::Debug,
    O: OrderMarker,
{
    type Output = Self;

    fn add(self, rhs: Stack<X, Y, O>) -> Self::Output {
        let (y, li, ri) = merge_union_indices(&self.y, &rhs.y);
        let prov = Provenance::merge(&self.prov, &rhs.prov, &li, &ri, y.len());

        let n = y.len();
        let mut x = vec![X::zero(); n];

        for (src, out) in li.into_iter().enumerate() {
            x[out] = x[out] + self.x[src];
        }

        for (src, out) in ri.into_iter().enumerate() {
            x[out] = x[out] + rhs.x[src];
        }

        Self {
            x,
            y,
            prov,
            _ord: PhantomData,
        }
    }
}

/// Custom `Debug` to format `Stack` as a struct with fields `x`, `y`, and `prov`.
impl<X, Y, O> fmt::Debug for Stack<X, Y, O>
where
    X: fmt::Debug,
    Y: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stack")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("prov", &self.prov)
            .finish()
    }
}

// utility functions

/// Checks that `y` is strictly monotonic according to `O`.
///
/// # Errors
///
/// - `Error::NonMonotonicLevels` if any adjacent pair violates the order.
fn validate_level_vector<Y, O>(y: &[Y]) -> Result<(), Error>
where
    Y: PartialOrd,
    O: OrderMarker,
{
    if y.windows(2).all(|w| O::is_in_order(&w[0], &w[1])) {
        Ok(())
    } else {
        Err(Error::NonMonotonicLevels)
    }
}

/// Ensures all mass values are non-negative.
///
/// # Errors
///
/// - `Error::NegativeMass` if any entry < 0.
fn validate_mass_vector<X>(x: &[X]) -> Result<(), Error>
where
    X: PartialOrd + Zero,
{
    for mass in x.iter() {
        if *mass < X::zero() {
            return Err(Error::NegativeMass);
        }
    }

    Ok(())
}

/// Validates the mass and level vectors jointly:
/// - Checks non-negative mass.
/// - Checks strict monotonicity.
/// - Checks equal length.
///
/// # Errors
///
/// Propagates errors from `validate_mass_vector` or
/// `validate_level_vector`, or returns `Error::LengthMismatch`.
fn validate_stack_vectors<X, Y, O>(x: &[X], y: &[Y]) -> Result<(), Error>
where
    X: PartialOrd + Zero,
    Y: PartialOrd,
    O: OrderMarker,
{
    validate_mass_vector(x)?;
    validate_level_vector::<Y, O>(y)?;

    if x.len() != y.len() {
        return Err(Error::LengthMismatch(x.len(), y.len()));
    }

    Ok(())
}

/// Given two sorted slices `left` and `right`, computes:
/// 1. A sorted unique union of their elements.
/// 2. Index mapping from `left` entries into the union.
/// 3. Index mapping from `right` entries into the union.
///
/// Useful for merging two stacks with alignment of levels.
fn merge_union_indices<T>(left: &[T], right: &[T]) -> (Vec<T>, Vec<usize>, Vec<usize>)
where
    T: Copy + std::cmp::PartialOrd,
{
    // reserve the worst-case capacity (no reallocs)
    let max_union_size = left.len() + right.len();
    let mut union = Vec::with_capacity(max_union_size);
    let mut li = Vec::with_capacity(left.len());
    let mut ri = Vec::with_capacity(right.len());

    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        let left_val = &left[i];
        let right_val = &right[j];
        let current_idx = union.len();

        if left_val < right_val {
            union.push(*left_val);
            li.push(current_idx);
            i += 1;
        } else if right_val < left_val {
            union.push(*right_val);
            ri.push(current_idx);
            j += 1;
        } else {
            // equal values - only insert once, but record index for both
            union.push(*left_val);
            li.push(current_idx);
            ri.push(current_idx);
            i += 1;
            j += 1;
        }
    }

    // drain any leftovers
    while i < left.len() {
        let current_idx = union.len();
        union.push(left[i]);
        li.push(current_idx);
        i += 1;
    }
    while j < right.len() {
        let current_idx = union.len();
        union.push(right[j]);
        ri.push(current_idx);
        j += 1;
    }

    // shrink vectors to actual size if significantly smaller capacity
    if union.len() < max_union_size / 2 {
        union.shrink_to_fit();
        li.shrink_to_fit();
        ri.shrink_to_fit();
    }

    (union, li, ri)
}
