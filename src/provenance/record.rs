/// A single `Record` needs to be light-weight in terms of memory and easily
/// extendable/appendable
use std::convert::From;
use std::mem;
use std::slice::{Iter as SliceIter, IterMut as SliceIterMut};

use crate::SourceID;

/// Compact container for one or more `(SourceID, X)` entries at a single level.
///
/// - `Empty`: No entries.
/// - `One((SourceID, X))`: Exactly one entry.
/// - `Many(Vec<(SourceID, X)>)`: Multiple entries.
#[derive(Clone, Debug)]
pub enum Record<X> {
    Empty,
    One((SourceID, X)),
    Many(Vec<(SourceID, X)>),
}

impl<X> Record<X> {
    /// Create a new empty record.
    pub fn empty() -> Self {
        Record::Empty
    }

    /// Creates an empty record (`Record::Empty`).
    ///
    /// # Returns
    /// `Record::Empty`.
    pub fn new(src: SourceID, x: X) -> Self {
        Record::One((src, x))
    }

    /// Creates a record containing `(src, mass)` pairs for each element `mass` in `iter`.
    ///
    /// # Parameters
    /// - `src`: Source identifier.
    /// - `iter`: Iterable of masses.
    ///
    /// # Returns
    /// `Record`.
    pub fn from_iter_with_source<I>(src: SourceID, iter: I) -> Self
    where
        I: IntoIterator<Item = X>,
    {
        let mut vec = iter.into_iter().map(|mass| (src, mass)).collect::<Vec<_>>();

        match vec.len() {
            0 => Self::Empty,
            1 => Self::One(vec.pop().unwrap()),
            _ => Self::Many(vec),
        }
    }

    /// Appends all entries from another record onto this one.
    ///
    /// # Parameters
    /// - `other`: Another `Record<X>` whose entries will be merged into `self`.
    ///
    /// # Behavior
    /// - Does nothing if `other` is `Empty`.
    /// - Pushes the single entry if `other` is `One`.
    /// - Iterates and pushes each entry if `other` is `Many`.
    pub fn extend(&mut self, other: Record<X>) {
        // drain out `self` into `old_self`, leaving a temporary "empty" `Many`
        let old = std::mem::replace(self, Record::Empty);

        // collect all entries (old + new) into one Vec
        let mut combined = match old {
            Record::Empty => Vec::new(),
            Record::One(rec) => vec![rec],
            Record::Many(vec) => vec,
        };

        // extend by consuming `other`
        combined.extend(other);

        // put in back in the same spot in memory, ensuring
        *self = match combined.len() {
            0 => Record::Empty,
            1 => Record::One(combined.pop().unwrap()),
            _ => Record::Many(combined),
        };
    }

    /// Converts this record into a single `(SourceID, X)` pair, if exactly one exists.
    ///
    /// # Returns
    /// - `Some((SourceID, X))` if the record is `One`.
    /// - `None` otherwise.
    pub fn into_one(self) -> Option<(SourceID, X)> {
        if let Record::One((src, x)) = self {
            Some((src, x))
        } else {
            None
        }
    }

    /// Checks whether the record contains exactly one entry.
    ///
    /// # Returns
    /// - `true` if `self` is `One`.
    /// - `false` otherwise.
    pub fn is_one(&self) -> bool {
        matches!(self, Record::One(_))
    }

    /// Returns the number of provenance entries in the record.
    ///
    /// # Returns
    /// - `0` for `Empty`.
    /// - `1` for `One`.
    /// - The length of the vector for `Many`.
    pub fn len(&self) -> usize {
        match self {
            Record::Empty => 0,
            Record::One(_) => 1,
            Record::Many(vec) => vec.len(),
        }
    }

    /// Adds a `(SourceID, X)` entry to the record.
    ///
    /// # Parameters
    /// - `src`: The `SourceID` of the new entry.
    /// - `x`: The mass value for the new entry.
    ///
    /// # Behavior
    /// - Transforms `Empty` into `One`.
    /// - Transforms `One` into `Many`, preserving the existing entry.
    /// - Pushes onto the `Many` variant.
    pub fn push(&mut self, src: SourceID, x: X) {
        match self {
            Record::Empty => *self = Record::One((src, x)),
            Record::One(_) => {
                // take the existing entry out...
                let (prev_src, prev_x) = mem::replace(self, Record::Many(Vec::new()))
                    .into_one()
                    .expect("just replaced with One");
                // ...and build a new Vec
                let mut v = Vec::new();
                v.push((prev_src, prev_x));
                v.push((src, x));
                *self = Record::Many(v);
            }
            Record::Many(vec) => {
                vec.push((src, x));
            }
        }
    }

    /// Returns an iterator over immutable references to `(SourceID, X)` pairs.
    ///
    /// # Returns
    /// A `RecordIter<'_, X>` yielding `&(SourceID, X)`.
    pub fn iter(&self) -> RecordIter<'_, X> {
        match self {
            Record::Empty => RecordIter::Empty,
            Record::One(rec) => RecordIter::One(Some(rec)),
            Record::Many(vec) => RecordIter::Many(vec.iter()),
        }
    }

    /// Returns an iterator over mutable references to `(SourceID, X)` pairs.
    ///
    /// # Returns
    /// A `RecordIterMut<'_, X>` yielding `&mut (SourceID, X)`.
    pub fn iter_mut(&mut self) -> RecordIterMut<'_, X> {
        match self {
            Record::Empty => RecordIterMut::Empty,
            Record::One(rec) => RecordIterMut::One(Some(rec)),
            Record::Many(vec) => RecordIterMut::Many(vec.iter_mut()),
        }
    }
}

// interface

/// Allows conversion from a `(SourceID, X)` tuple into a `Record<X>`.
///
/// # Example
/// ```
/// let record: Record<_> = (42, value).into();
/// ```
impl<X> From<(SourceID, X)> for Record<X> {
    fn from(item: (SourceID, X)) -> Self {
        Self::One(item)
    }
}

/// Allows conversion from a value `X` into a `Record<X>` by assigning a default
/// `SourceID` of `0`.
///
/// # Warning
/// This may be unsafe or semantically incorrect if `0 as SourceID` is not valid
///
/// # Example
/// ```
/// let record: Record<_> = value.into();
/// ```
impl<X> From<X> for Record<X> {
    // this could cause potential problems w/o constraints on `X`
    fn from(mass: X) -> Self {
        let src = 0 as SourceID;
        Self::One((src, mass))
    }
}

/// Provides `Record::empty()` as the default `Default` implementation.
///
/// # Type Parameters
/// - `X`: Must implement `Default`, though not used to populate data.
impl<X> Default for Record<X>
where
    X: Default,
{
    fn default() -> Self {
        Self::empty()
    }
}

/// Provides equality comparison between `Record<X>` values.
///
/// # Equality Rules
/// - Two `Empty` records are equal.
/// - Two `One` records are equal if their contents are equal.
/// - Two `Many` records are equal if their vectors are element-wise equal.
/// - Any other variant combinations are not equal.
///
/// # Type Parameters
/// - `X` must implement `PartialEq`.
impl<X> PartialEq for Record<X>
where
    X: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Record::Empty, Record::Empty) => true,
            (Record::One(l), Record::One(r)) => l == r,
            (Record::Many(l), Record::Many(r)) => l == r,
            _ => false,
        }
    }
}

// iteration interface

/// Creating a `Record` from an iterator (`source` forced to `0 as SourceID`)
impl<X> FromIterator<X> for Record<X> {
    // Private helper: build from a bare iterator of mass values under a single source.
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = X>,
    {
        let src = 0 as SourceID;
        let mut vec = iter.into_iter().map(|mass| (src, mass)).collect::<Vec<_>>();

        match vec.len() {
            0 => Self::Empty,
            1 => Self::One(vec.pop().unwrap()),
            _ => Self::Many(vec),
        }
    }
}

/// Consumes the record, yielding owned `(SourceID, X)` pairs.
///
/// - `Empty` yields nothing.
/// - `One` yields the single `(SourceID, X)`.
/// - `Many` yields all stored pairs in order.
impl<X> IntoIterator for Record<X> {
    type Item = (SourceID, X);
    type IntoIter = std::vec::IntoIter<(SourceID, X)>;

    // Private helper: convert into an iterator consuming self.
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Record::Empty => Vec::new().into_iter(),
            Record::One(rec) => vec![rec].into_iter(),
            Record::Many(vec) => vec.into_iter(),
        }
    }
}

/// Iterator over immutable references to `(SourceID, X)` in a `Record<X>`.
///
/// Supports efficient dispatch for each variant:
/// - `Empty`: yields no items.
/// - `One`: yields the single reference once.
/// - `Many`: wraps a slice iterator over a `Vec`.
pub enum RecordIter<'a, X> {
    Empty,
    One(Option<&'a (SourceID, X)>),
    Many(SliceIter<'a, (SourceID, X)>),
}

impl<'a, X> Iterator for RecordIter<'a, X> {
    type Item = &'a (SourceID, X);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RecordIter::Empty => None,
            RecordIter::One(opt) => opt.take(),
            RecordIter::Many(iter) => iter.next(),
        }
    }
}

/// Iterator over mutable references to `(SourceID, X)` in a `Record<X>`.
///
/// Variants mirror `RecordIter` but yield `&mut` references.
pub enum RecordIterMut<'a, X> {
    Empty,
    One(Option<&'a mut (SourceID, X)>),
    Many(SliceIterMut<'a, (SourceID, X)>),
}

impl<'a, X> Iterator for RecordIterMut<'a, X> {
    type Item = &'a mut (SourceID, X);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RecordIterMut::Empty => None,
            RecordIterMut::One(opt) => opt.take(),
            RecordIterMut::Many(iter) => iter.next(),
        }
    }
}
