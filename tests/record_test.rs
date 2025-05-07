use rusty_stacks::provenance::record::Record;

#[test]
fn init_record_tests() {
    let _record = Record::new(0, 3);
    let _record = Record::from_iter([3, 4, 5, 1]);
    let _record = Record::from_iter_with_source(3, [3, 4, 5, 1]);

    let iter = ["rince", "wind", "mortimer", "death"];
    let _record: Record<_> = iter.iter().collect();
}

#[test]
fn extend_method() {
    // One + One
    let mut left = Record::new(0, 3);
    let right = Record::new(1, 76);
    let _record = left.extend(right);

    // One + Many
    let mut left = Record::new(0, 3);
    let right = Record::from_iter_with_source(1, [9, 43, 1, 2]);
    let _record = left.extend(right);

    // Many + One
    let mut left = Record::from_iter([9, 43, 1, 2]);
    let right = Record::new(1, 3);
    let _record = left.extend(right);

    // Many + Many
    let mut left = Record::from_iter([9, 43, 1, 2]);
    let right = Record::from_iter_with_source(312, [4, 6, 7, 10]);
    let _record = left.extend(right);
}

#[test]
fn record_equality() {
    let l = Record::One((0, 45));
    let r = Record::One((0, 45));
    assert_eq!(l, r);

    let l = Record::Empty;
    let r: Record<_> = (1, "rincewind").into();
    assert_ne!(l, r);
}
