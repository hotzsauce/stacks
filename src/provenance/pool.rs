use rustc_hash::FxHashMap;
use std::convert::From;

use crate::SourceID;

/// String interner managing a mapping between source labels and numeric `SourceID`s.
///
/// Each `SourcePool` maintains:
/// - `lookup`: `FxHashMap<String, SourceID>` for fast ID lookup.
/// - `sources`: `Vec<String>` storing labels by their ID index.
#[derive(Clone, Debug)]
pub struct SourcePool {
    pub lookup: FxHashMap<String, SourceID>,
    pub sources: Vec<String>,
}

impl SourcePool {
    // instantiation methods

    /// Creates an empty `SourcePool` with no interned sources.
    ///
    /// # Returns
    /// A blank `SourcePool`.
    pub fn new() -> Self {
        let lookup = FxHashMap::default();
        let sources = Vec::new();
        SourcePool { lookup, sources }
    }

    /// Builds a `SourcePool` by interning each label in an iterable.
    ///
    /// # Type Parameters
    /// - `I`: Iterable of source labels.
    /// - `S`: Convertible into `String`.
    ///
    /// # Parameters
    /// - `iter`: Iterable of sources.
    ///
    /// # Returns
    /// A `SourcePool` containing all provided labels.
    pub fn from_sources<I, S>(iter: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut pool = Self::new();
        for source in iter {
            pool.intern(source);
        }
        pool
    }

    // interface methods

    /// Interns a single source label, returning its `SourceID`.
    ///
    /// If the label is already interned, returns the existing ID; otherwise assigns a new ID.
    ///
    /// # Type Parameters
    /// - `S`: Convertible into `String`.
    ///
    /// # Parameters
    /// - `source`: Label to intern.
    ///
    /// # Returns
    /// `SourceID` corresponding to `source`.
    pub fn intern<S>(&mut self, source: S) -> SourceID
    where
        S: Into<String>,
    {
        let key = source.into();

        if let Some(&id) = self.lookup.get(&key) {
            id
        } else {
            let id = self.sources.len() as SourceID;
            self.sources.push(key.clone());
            self.lookup.insert(key, id);
            id
        }
    }

    /// Checks whether the `SourcePool` has any interned labels.
    ///
    /// # Returns
    /// `true` if no labels have been interned; `false` otherwise.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Resolves a `SourceID` back to its interned label.
    ///
    /// # Parameters
    /// - `id`: `SourceID` to look up.
    ///
    /// # Returns
    /// `Option<&str>` of the label if `id` exists.
    pub fn resolve(&self, id: SourceID) -> Option<&str> {
        self.sources.get(id as usize).map(|s| s.as_str())
    }

    /// Merges two `SourcePool`s into one, returning remapping tables.
    ///
    /// Interns all labels from `a` and `b` into a new pool, producing
    /// two `FxHashMap<SourceID, SourceID>` maps to translate IDs from each original pool.
    ///
    /// # Parameters
    /// - `a`: First `SourcePool`.
    /// - `b`: Second `SourcePool`.
    ///
    /// # Returns
    /// `(merged_pool, a_map, b_map)`.
    pub fn union(
        a: &SourcePool,
        b: &SourcePool,
    ) -> (
        SourcePool,
        FxHashMap<SourceID, SourceID>,
        FxHashMap<SourceID, SourceID>,
    ) {
        let mut pool = SourcePool::new();
        let mut a_map = FxHashMap::default();
        let mut b_map = FxHashMap::default();

        for (id, name) in a.sources.iter().enumerate() {
            let new_id = pool.intern(name);
            a_map.insert(id as SourceID, new_id);
        }

        for (id, name) in b.sources.iter().enumerate() {
            let new_id = pool.intern(name);
            b_map.insert(id as SourceID, new_id);
        }

        (pool, a_map, b_map)
    }
}

/// Converts a single label into a `SourcePool` containing just that label.
///
/// # Type Parameters
/// - `S`: Convertible into `String`.
///
/// # Parameters
/// - `string`: Label to intern.
///
/// # Returns
/// A singleton `SourcePool`.
impl<S> From<S> for SourcePool
where
    S: Into<String>,
{
    fn from(string: S) -> Self {
        Self::from_sources([string])
    }
}

/// Provides `SourcePool::new()` as the default constructor.
impl Default for SourcePool {
    fn default() -> Self {
        Self::new()
    }
}
