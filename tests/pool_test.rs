use stacks::provenance::pool::SourcePool;

#[test]
fn init_pool_from_sources() {
    let sources = vec!["rincewind", "mortimer", "death"];
    let _pool = SourcePool::from_sources(sources);
}

#[test]
fn init_pool_from_str() {
    let _pool: SourcePool = "logan".into();
}

#[test]
fn init_pool_from_string() {
    let _pool: SourcePool = "logan".to_string().into();
}

#[test]
fn pool_union() {
    let left = SourcePool::from_sources(vec!["rincewind", "mortimer", "death"]);
    let right = SourcePool::from_sources(vec!["roo", "ginger", "mortimer"]);
    let (union, _, _) = SourcePool::union(&left, &right);

    assert_eq!(
        union.sources,
        vec!["rincewind", "mortimer", "death", "roo", "ginger"]
    );
}

#[test]
fn pool_clone() {
    let left = SourcePool::from_sources(vec!["rincewind", "mortimer", "death"]);
    let _ = left.clone();
}
