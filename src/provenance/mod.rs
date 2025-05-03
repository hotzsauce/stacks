pub mod pool;
pub mod record;

use crate::provenance::pool::SourcePool;
use crate::provenance::record::Record;

/// Tracks the origin (`SourcePool`) and per-level records (`Record<X>`) of mass values in a `Stack`.
///
/// # Type Parameters
/// - `X`: Numeric type for mass entries; used in each `Record<X>`.
///
/// # Fields
/// - `records`: `Vec<Record<X>>` mapping each level to its provenance entries.
/// - `pool`: `SourcePool` interning the unique sources referenced by `records`.
#[derive(Clone, Debug)]
pub struct Provenance<X> {
    pub records: Vec<Record<X>>,
    pub pool: SourcePool,
}

impl<X> Provenance<X>
where
    X: Copy,
{
    // instantiation methods
    /// Constructs an empty `Provenance` with no sources and no mass records.
    ///
    /// # Returns
    /// A `Provenance<X>` with:
    /// - `records = []`
    /// - `pool` initialized empty.
    pub fn new() -> Self {
        let records = Vec::<Record<X>>::new();
        let pool = SourcePool::new();
        Self { records, pool }
    }

    /// Creates provenance for a single source across multiple mass entries.
    ///
    /// # Type Parameters
    /// - `S`: Convertible into `String` to label the source.
    /// - `M`: Iterable of mass values (`X`).
    ///
    /// # Parameters
    /// - `src`: Initial source label.
    /// - `xs`: Mass values to associate with `src`.
    ///
    /// # Returns
    /// A `Provenance<X>` where each mass in `xs` is recorded under `src`.
    pub fn from_source_and_mass<S, M>(src: S, xs: M) -> Self
    where
        S: Into<String>,
        M: AsRef<[X]>,
    {
        let mut pool = SourcePool::new();
        let id = pool.intern(src);

        let records = xs
            .as_ref()
            .iter()
            .copied()
            .map(|mass| Record::new(id, mass))
            .collect();

        Self { records, pool }
    }

    /// Creates provenance entries for mass values without assigning a named source.
    ///
    /// # Type Parameters
    /// - `M`: Iterable of mass values (`X`).
    ///
    /// # Parameters
    /// - `xs`: Mass values to record.
    ///
    /// # Returns
    /// A `Provenance<X>` with each mass recorded under an unnamed default source (ID `0`).
    pub fn from_mass<M>(xs: M) -> Self
    where
        M: AsRef<[X]>,
    {
        let pool = SourcePool::new();
        let records = xs
            .as_ref()
            .iter()
            .copied()
            .map(|mass| Record::new(0, mass))
            .collect();

        Self { records, pool }
    }

    /// Merges two provenance sets into one, aligning records by output indices.
    ///
    /// Performs a union of the two `SourcePool`s and remaps source IDs accordingly,
    /// then consolidates per-level `Record<X>` entries into a new `Provenance<X>`.
    ///
    /// # Parameters
    /// - `left`: First provenance backer.
    /// - `right`: Second provenance backer.
    /// - `li`: Mapping from left records to output indices.
    /// - `ri`: Mapping from right records to output indices.
    /// - `output_len`: Number of output levels.
    ///
    /// # Returns
    /// A merged `Provenance<X>` sharing a unified `SourcePool` and consolidated `records`.
    pub fn merge(
        left: &Provenance<X>,
        right: &Provenance<X>,
        li: &[usize],
        ri: &[usize],
        output_len: usize,
    ) -> Provenance<X> {
        // union the pools and get back the id-maps
        let (mut pool, mut lmap, mut rmap) = SourcePool::union(&left.pool, &right.pool);
        let unnamed = "<unnamed>";

        // pre-allocate the blank record for each level
        let mut records: Vec<Record<X>> = (0..output_len).map(|_| Record::empty()).collect();

        // pour in the left provenance
        for (src, &out) in li.iter().enumerate() {
            for &(old_src, mass) in left.records[src].iter() {
                let new_src = *lmap.entry(old_src).or_insert_with(|| {
                    let name = left.pool.resolve(old_src).unwrap_or(unnamed);
                    pool.intern(name)
                });
                records[out].push(new_src, mass);
            }
        }

        // pour in the right provenance
        for (src, &out) in ri.iter().enumerate() {
            for &(old_src, mass) in right.records[src].iter() {
                let new_src = *rmap.entry(old_src).or_insert_with(|| {
                    let name = right.pool.resolve(old_src).unwrap_or(unnamed);
                    pool.intern(name)
                });
                records[out].push(new_src, mass);
            }
        }

        Provenance { records, pool }
    }

    // Interface methods

    /// Interns a new source label into the `SourcePool`.
    ///
    /// # Type Parameters
    /// - `S`: Convertible into `String`.
    ///
    /// # Parameters
    /// - `src`: Label to add.
    ///
    /// # Returns
    /// Mutable reference to `self` for method chaining.
    pub fn add_source<S>(&mut self, src: S) -> &mut Self
    where
        S: Into<String>,
    {
        self.pool.intern(src);
        self
    }

    /// Returns an immutable reference to the underlying `SourcePool`.
    ///
    /// # Returns
    /// `&SourcePool` containing all interned sources.
    pub fn pool(&self) -> &SourcePool {
        &self.pool
    }

    /// Retrieves the list of source labels in the `SourcePool`.
    ///
    /// # Returns
    /// `&[String]` of all interned source names.
    pub fn sources(&self) -> &[String] {
        &self.pool.sources
    }
}
