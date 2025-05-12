// methods that already exist in the python implementation:
// [x] clip(X) -> Stack<X, Y>
// [x] cumulate() -> Vec<X>
// [x] mean() -> [UNKONOWN?]
// [x] project_onto(Vec<Y>) -> Stack<X, Y>
// [x] rename(&str) -> Stack<X, Y> {here, called `alias`}
// [x] truncate(Y) -> Stack<X, Y>
// [x] total() -> X
// [x] total_above(Y) -> X
// [x] total_below(Y) -> X
// [x] X() -> Vec<X> {should be renamed to masses()?}
// [x] Y() -> Vec<Y> {should be renamed to levels()?}
//
// [x] __add__(Stack<X, Y>, Stack<W, Z>) -> Stack<(X, W), (Y, Z)>
// [x] __radd__(Stack<X, Y>, Stack<W, Z>) -> Stack<(W, X), (Z, Y)>
// [x] __ge__(Stack<X, Y>, Stack<W, Z>) -> Ordering
// [x] __len__(Stack<X, Y>) -> usize
//
// [x] wipe(&str) -> Stack<X, Y>
//      I'm not sure what this method name should actually be, but
//      it's meant to "wipe" the existing provenance vector into
//      the single provided source

use either::Either;
use num_traits::{Num, ToPrimitive, Zero};

use std::cmp::{Ordering, PartialOrd};
use std::fmt;
use std::iter::{DoubleEndedIterator, ExactSizeIterator, FusedIterator};
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

/// An iterator adaptor that yields the running (prefix) sums of an inner iterator.
///
/// Each time you call `next()`, the returned value is the sum of all items
/// seen so far, including the current one.
///
/// # Type parameters
///
/// - `I`: The inner iterator yielding items of type `T`.
/// - `T`: The element and accumulation type. Must implement `Default` (for the
///   initial zero), `Add<Output = T>` (for summation), and `Copy`.
///
/// # Examples
///
/// ```rust
/// use rusty_stacks::CumulativeSum;
///
/// let data = vec![1, 2, 3];
/// let mut cum = CumulativeSum::new(data.into_iter());
/// assert_eq!(cum.next(), Some(1));
/// assert_eq!(cum.next(), Some(3));
/// assert_eq!(cum.next(), Some(6));
/// assert_eq!(cum.next(), None);
/// ```
pub struct CumulativeSum<I, T> {
    iter: I,
    init: T,
}

impl<I, T> CumulativeSum<I, T>
where
    T: Default,
{
    /// Creates a new `CumulativeSum` iterator starting from zero.
    ///
    /// # Parameters
    ///
    /// - `iter`: The base iterator whose items will be accumulated.
    ///
    /// # Returns
    ///
    /// A `CumulativeSum` adaptor that yields prefix sums of the inner iterator.
    pub fn new(iter: I) -> Self {
        let init = T::default();
        Self { iter, init }
    }
}

impl<I, T> Iterator for CumulativeSum<I, T>
where
    I: Iterator<Item = T>,
    T: Add<Output = T> + Copy,
{
    type Item = T;

    /// Advances the iterator and returns the next cumulative total.
    ///
    /// Returns `None` when the inner iterator is exhausted.
    fn next(&mut self) -> Option<T> {
        self.iter.next().map(|x| {
            self.init = self.init + x;
            self.init
        })
    }
}

/// Extension trait providing a `cumulative_sum` adaptor for any iterator.
///
/// Once in scope, any `Iterator` whose `Item` implements `Default + Add + Copy`
/// gains a `cumulative_sum()` method.
///
/// # Examples
///
/// ```
/// use rusty_stacks::IteratorCumulativeSum;
///
/// let result: Vec<_> = vec![4, 5, 6]
///     .into_iter()
///     .cumulative_sum()
///     .collect();
/// assert_eq!(result, vec![4, 9, 15]);
/// ```
pub trait IteratorCumulativeSum: Iterator + Sized {
    fn cumulative_sum(self) -> CumulativeSum<Self, Self::Item>
    where
        Self::Item: Default + Add<Output = Self::Item> + Copy,
    {
        CumulativeSum::new(self)
    }
}

// blanket implementation for all iterators
impl<I: Iterator> IteratorCumulativeSum for I {}

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
        self.x.iter().copied().cumulative_sum().collect()
    }

    /// Returns an iterator over references to the masses and levels in this stack.
    ///
    /// Each element yielded is a pair (`&X`, `&Y`), corresponding to one
    /// entry in the `x` (mass) and `y` (level) vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![1, 2], vec![10.0, 20.0]).unwrap();
    /// let snapshot: Vec<_> = stack.iter().collect();
    /// assert_eq!(snapshot, vec![(&1, &10.0), (&2, &20.0)]);
    /// ```
    pub fn iter(&self) -> StackVectorIterator<'_, X, Y> {
        let iter = StackVectorIterator::new(&self.x, &self.y);
        iter
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

    /// Borrow a read-only slice of all the levels (y-values) in this stack.
    ///
    /// # Returns
    ///
    /// A slice `&[Y]` containing each level exactly once, in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2, 3, 4], vec![3.3, 4.4, 5.5]).unwrap();
    /// let lvls: &[f64] = stack.levels();
    /// println!("levels = {:?}", lvls);
    /// ```
    pub fn levels(&self) -> &[Y] {
        &self.y
    }

    /// Eagerly collect a “fine‐grained” list of levels into a `Vec<Y>`.
    ///
    /// Each level is duplicated according to the number of sources
    /// contributing at that level (as recorded in `self.prov.records`).
    ///
    /// # Returns
    ///
    /// A newly allocated `Vec<Y>` of length
    /// `sum(self.prov.records.iter().map(|r| r.len()))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2, 3, 4], vec![3.3, 4.4, 5.5]).unwrap();
    /// let fine: Vec<f64> = stack.levels_fine();
    /// println!("fine levels = {:?}", fine);
    /// ```
    pub fn levels_fine(&self) -> Vec<Y> {
        self.levels_fine_iter().collect()
    }

    /// Lazily iterate over a fine-grained view of levels.
    ///
    /// This zero-allocation iterator emits each level `y` exactly `n` times,
    /// where `n` is `self.prov.records[i].len()` for the corresponding index `i`.
    ///
    /// # Returns
    ///
    /// An `impl Iterator<Item = Y>` producing owned `Y` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2, 3, 4], vec![3.3, 4.4, 5.5]).unwrap();
    /// for lvl in stack.levels_fine_iter() {
    ///     // lvl appears once per source
    ///     println!("{:?}", lvl);
    /// }
    /// ```
    pub fn levels_fine_iter(&self) -> impl Iterator<Item = Y> + '_ {
        self.y
            .iter()
            .copied()
            .zip(self.prov.records.iter())
            .flat_map(|(y, rec)| std::iter::repeat(y).take(rec.len()))
    }

    /// Lazily iterate over each level exactly once.
    ///
    /// This zero-allocation iterator simply yields each `Y` in `self.y`
    /// by value (via `Copy`).
    ///
    /// # Returns
    ///
    /// An `impl Iterator<Item = Y>` producing each level in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2, 3, 4], vec![3.3, 4.4, 5.5]).unwrap();
    /// for lvl in stack.levels_iter() {
    ///     println!("level = {:?}", lvl);
    /// }
    /// ```
    pub fn levels_iter(&self) -> impl Iterator<Item = Y> + '_ {
        self.y.iter().copied()
    }

    /// Borrow a read‐only slice of all the masses (x‐values) in this stack.
    ///
    /// # Returns
    ///
    /// A slice `&[X]` containing each mass exactly once, in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2.0, 3.0, 4.0], vec![3.3, 4.4, 5.5]).unwrap();
    /// let ms: &[f64] = stack.masses();
    /// println!("masses = {:?}", ms);
    /// ```
    pub fn masses(&self) -> &[X] {
        &self.x
    }

    /// Eagerly collect a “fine‐grained” list of masses into a `Vec<X>`.
    ///
    /// Each mass value is duplicated according to the number of sources
    /// contributing at that index (as recorded in `self.prov.records`).
    ///
    /// # Returns
    ///
    /// A newly allocated `Vec<X>` of length
    /// `sum(self.prov.records.iter().map(|r| r.len()))`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2.0, 3.0, 4.0], vec![3.3, 4.4, 5.5]).unwrap();
    /// let fine: Vec<f64> = stack.masses_fine();
    /// println!("fine masses = {:?}", fine);
    /// ```
    pub fn masses_fine(&self) -> Vec<X> {
        self.masses_fine_iter().collect()
    }

    /// Lazily iterate over a fine‐grained view of masses.
    ///
    /// This zero‐allocation iterator emits each mass `x` exactly `n` times,
    /// where `n` is `self.prov.records[i].len()` for the corresponding index `i`.
    ///
    /// # Returns
    ///
    /// An `impl Iterator<Item = X>` producing owned `X` values.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2, 3, 4], vec![3.3, 4.4, 5.5]).unwrap();
    /// for m in stack.masses_fine_iter() {
    ///     // each mass appears once per contributing source
    ///     println!("{:?}", m);
    /// }
    /// ```
    pub fn masses_fine_iter(&self) -> impl Iterator<Item = X> + '_ {
        self.x
            .iter()
            .copied()
            .zip(self.prov.records.iter())
            .flat_map(|(x, rec)| std::iter::repeat(x).take(rec.len()))
    }

    /// Lazily iterate over each mass exactly once.
    ///
    /// This zero‐allocation iterator simply yields each `X` in `self.x`
    /// by value (via the `Copy` bound).
    ///
    /// # Returns
    ///
    /// An `impl Iterator<Item = X>` producing each mass in order.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(vec![2, 3, 4], vec![3.3, 4.4, 5.5]).unwrap();
    /// for m in stack.masses_iter() {
    ///     println!("mass = {:?}", m);
    /// }
    /// ```
    pub fn masses_iter(&self) -> impl Iterator<Item = X> + '_ {
        self.x.iter().copied()
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

    /// Resets the provenance of every mass entry in the `Stack` to the given source.
    ///
    /// Discards any existing provenance information and creates a new provenance
    ///     pool where each mass value is associated *only* with `src`.
    ///
    /// # Type Parameters
    ///
    /// - `S`: A type that can be converted into a `String`, representing the new source
    ///     label.
    ///
    /// # Parameters
    ///
    /// - `src`: Identifier for the new provenance source.
    ///
    /// # Returns
    ///
    /// A mutable reference to `self`
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_stacks::IncStack;
    /// let mut stack = IncStack::try_from_vectors_and_source(
    ///     vec![1, 2, 3],
    ///     vec![10.0, 20.0, 30.0],
    ///     "initial"
    /// ).unwrap();
    ///
    /// // Now reassign every entry’s provenance to "reset_source":
    /// stack.wipe("reset_source");
    /// ```
    pub fn wipe<S>(&mut self, src: S) -> &mut Self
    where
        S: Into<String>,
    {
        let src = src.into();
        let prov = Provenance::from_source_and_mass(src, &self.x);

        self.prov = prov;
        self
    }
}

impl<X, Y, O> Stack<X, Y, O>
where
    X: Copy + Zero + ToPrimitive,
    Y: Copy + Zero + ToPrimitive,
    O: OrderMarker,
{
    /// Computes the weighted mean of this stack’s levels, using the masses as weights.
    ///
    /// Each mass (`x`) and level (`y`) is converted to `f64`; any entries that
    /// fail to convert are silently skipped. The result is
    ///
    /// ```text
    ///     mean = ∑(mass_i * level_i) / ∑(mass_i)
    /// ```
    ///
    /// # Returns
    ///
    /// - A `f64` containing the weighted average of the levels.
    /// - If the total mass is zero (or there are no valid entries), the result
    ///   will be `NaN`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rusty_stacks::IncStack;
    /// let stack = IncStack::try_from_vectors(
    ///     vec![1, 2, 3],
    ///     vec![10.0, 20.0, 30.0],
    /// ).unwrap();
    /// assert_eq!(
    ///     stack.mean(),
    ///     (1.0*10.0 + 2.0*20.0 + 3.0*30.0) / (1.0 + 2.0 + 3.0)
    /// );
    /// ```
    pub fn mean(&self) -> f64 {
        let mut num = f64::zero();
        let mut den = f64::zero();

        for (mass, level) in self
            .x
            .iter()
            .filter_map(|&x| x.to_f64())
            .zip(self.y.iter().filter_map(|&y| y.to_f64()))
        {
            num += mass * level;
            den += mass;
        }

        num / den
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
        let (y, li, ri) = merge_union_indices::<Y, O>(&self.y, &rhs.y);
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

/// Comparing `Stack` equality. Provenances are ignored entirely.
impl<X, Y, O> PartialEq for Stack<X, Y, O>
where
    X: PartialEq,
    Y: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y)
    }
}

/// Compare two `Stack`s by their cumulative mass distributions (CDFs),
/// implementing a variant of second-order stochastic dominance.
///
/// This method:
/// 1. Aligns both stacks’ levels by taking the sorted union of their `y` vectors.
/// 2. Uses `SparseIter` to fill each stack’s mass vector to that union length,
///    inserting zeros where a level is absent.
/// 3. Computes the running total (prefix sums) of each extended mass vector.
/// 4. At each level, records whether `self`’s cumulative mass is strictly less
///    or strictly greater than `other`’s.
///
/// The final decision is:
/// - `Some(Ordering::Greater)` if `self` has ≥ cumulative mass at every level
///    and strictly greater at least one level.
/// - `Some(Ordering::Less)`    if `self` has ≤ cumulative mass at every level
///    and strictly less    at least one level.
/// - `Some(Ordering::Equal)`   if both cumulative distributions are identical.
/// - `None`                    otherwise (the CDFs cross and no dominance holds).
///
/// # Type constraints
/// - `X`: Mass type. Must implement `Copy`, `PartialOrd`, `Default`, and `Add<Output = X>`.
/// - `Y`: Level type. Must implement `Copy` and `PartialOrd`.
/// - `O`: `OrderMarker` (`Increasing` or `Decreasing`), determining whether “below”
///   and “above” are interpreted in ascending or descending order of `y`.
///
/// # Performance
/// Runs in O(n + m) time (where n, m are the lengths of the two stacks) and
/// O(n + m) memory to hold the merged level list and intermediate prefix sums.
///
/// # Notes
/// - Provenance is ignored in comparisons.
/// - Uses `merge_union_indices` + `SparseIter` internally.
impl<X, Y, O> PartialOrd for Stack<X, Y, O>
where
    X: Copy + PartialOrd + Default + Add<Output = X> + std::fmt::Debug,
    Y: Copy + PartialOrd,
    O: OrderMarker,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (y, li, ri) = merge_union_indices::<Y, O>(&self.y, &other.y);
        let n = y.len();

        let lx = SparseIter::new(&self.x, &li, n).cumulative_sum();
        let rx = SparseIter::new(&other.x, &ri, n).cumulative_sum();

        let mut any_lt = false;
        let mut any_gt = false;
        let mut all_ge = true;
        let mut all_le = true;

        for (a, b) in lx.zip(rx) {
            if a < b {
                any_lt = true;
                all_ge = false;
            }
            if b < a {
                any_gt = true;
                all_le = false;
            }
        }

        match (all_le, all_ge, any_lt, any_gt) {
            (true, true, false, false) => Some(Ordering::Equal),
            (true, false, true, false) => Some(Ordering::Less),
            (false, true, false, true) => Some(Ordering::Greater),
            _ => None, // crosses
        }
    }
}

// `Stack` iterators

/// Iterator over pairs of mass and level references in a `Stack`.
///
/// Yields one tuple `(&X, &Y)` per index, walking the `x` and `y` slices
/// in lockstep. Supports forward and backward traversal, exact size
/// reporting, and fused termination.
pub struct StackVectorIterator<'a, X, Y> {
    x: std::slice::Iter<'a, X>,
    y: std::slice::Iter<'a, Y>,
}

impl<'a, X, Y> StackVectorIterator<'a, X, Y> {
    /// Creates a new `StackVectorIterator` from parallel slices.
    ///
    /// # Parameters
    ///
    /// - `x`: Slice of mass values.
    /// - `y`: Slice of level values.
    ///
    /// # Returns
    ///
    /// An iterator that yields `( &X, &Y )` for each corresponding index.
    pub fn new(x: &'a [X], y: &'a [Y]) -> Self {
        Self {
            x: x.iter(),
            y: y.iter(),
        }
    }
}

impl<'a, X, Y> Iterator for StackVectorIterator<'a, X, Y> {
    type Item = (&'a X, &'a Y);

    /// Advances the iterator and returns the next (`&mass`, `&level`) pair.
    ///
    /// Returns `None` when either slice is exhausted.
    fn next(&mut self) -> Option<Self::Item> {
        match (self.x.next(), self.y.next()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        }
    }

    /// Returns an exact lower and upper bound on the remaining length.
    ///
    /// Both bounds are the same (`lo == hi`), equal to the number of
    /// elements left in the shorter of the two underlying slices.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lo1, hi1) = self.x.size_hint();
        let (lo2, hi2) = self.y.size_hint();

        let lo = lo1.min(lo2);
        let hi = match (hi1, hi2) {
            (Some(h1), Some(h2)) => Some(h1.min(h2)),
            _ => None,
        };
        (lo, hi)
    }
}

// implement DoubleEndedIterator to get `.rev()
impl<'a, X, Y> DoubleEndedIterator for StackVectorIterator<'a, X, Y> {
    /// Advances the iterator from the back, returning the last (`&mass`, `&level`) pair.
    ///
    /// Returns `None` when no elements remain.
    fn next_back(&mut self) -> Option<Self::Item> {
        match (self.x.next_back(), self.y.next_back()) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        }
    }
}

// because `size_hint()` is exact
impl<'a, X, Y> ExactSizeIterator for StackVectorIterator<'a, X, Y> {
    /// Returns the exact number of elements remaining in the iterator.
    fn len(&self) -> usize {
        self.x.len()
    }
}

// mark as Fuxed so that once it returns None it never again yields Some
impl<'a, X, Y> FusedIterator for StackVectorIterator<'a, X, Y> {}

// utility functions

/// An iterator that interpolates a sparse set of values over a fixed length,
/// filling all other positions with `T::default()`.
///
/// Given
///  - a slice `x` of length `k`,
///  - a sorted index slice `index` of length `k` with values in `[0..n)`, and
///  - a target length `n`,
///
/// `SparseIter::new(&x, &index, n)` will yield `n` items:
/// at each position `j`:
///  - if `j` appears in `index[i]`, it yields `x[i]`,
///  - otherwise it yields `T::default()`.
///
/// # Type parameters
///
/// - `T`: The element type. Must implement `Default` (for filler values) and `Copy`.
///
/// # Panics
///
/// - If `x.len() != index.len()`.
/// - If `index` is empty or its last element ≥ `n`.
///
/// # Examples
///
/// ```rust
/// use rusty_stacks::SparseIter;
///
/// let values = vec![10, 20];
/// let positions = vec![1, 3];
/// let expanded: Vec<_> = SparseIter::new(&values, &positions, 5).collect();
/// assert_eq!(expanded, vec![0, 10, 0, 20, 0]);
/// ```
pub struct SparseIter<'a, T> {
    x: &'a [T],
    index: &'a [usize],
    done: usize, // how many of the pairs we've already emitted
    pos: usize,  // current j in 0..n
    n: usize,    // total length
}

impl<'a, T> SparseIter<'a, T> {
    /// Constructs the sparse iterator.
    ///
    /// # Parameters
    ///
    /// - `x`: Slice of `k` values to emit.
    /// - `index`: Sorted slice of `k` distinct positions where `x` should be placed.
    /// - `n`: Total number of items the iterator will yield.
    ///
    /// # Panics
    ///
    /// - If `x.len() != index.len()`.
    /// - If `n <= *index.last().unwrap()`.
    pub fn new(x: &'a [T], index: &'a [usize], n: usize) -> Self {
        assert!(x.len() == index.len(), "length mismatch");
        if let Some(&last) = index.last() {
            assert!(n > last, "`n` must exceed the final index");
        }
        Self {
            x,
            index,
            done: 0,
            pos: 0,
            n,
        }
    }
}

impl<'a, T> Iterator for SparseIter<'a, T>
where
    T: Default + Copy,
{
    type Item = T;

    /// Yields exactly `n` values: each position matching `index` yields the
    /// corresponding element from `x`; all other positions yield `T::default()`.
    fn next(&mut self) -> Option<T> {
        if self.pos >= self.n {
            return None;
        }
        // if the next index matches this position, emit that x; otherwise default
        let out = if self.done < self.index.len() && self.index[self.done] == self.pos {
            let v = self.x[self.done];
            self.done += 1;
            v
        } else {
            T::default()
        };
        self.pos += 1;
        Some(out)
    }
}

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
fn merge_union_indices<T, O>(left: &[T], right: &[T]) -> (Vec<T>, Vec<usize>, Vec<usize>)
where
    T: Copy + std::cmp::PartialOrd,
    O: OrderMarker,
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

        if O::is_in_order(left_val, right_val) {
            union.push(*left_val);
            li.push(current_idx);
            i += 1;
        } else if O::is_in_order(right_val, left_val) {
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
