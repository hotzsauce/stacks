use rusty_stacks::provenance::record::Record;

#[test]
fn init_record_tests() {
    let _record = Record::new(0, 3);
    let _record = Record::from(0, [3, 4, 5, 1]);

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
    let right = Record::from(1, [9, 43, 1, 2]);
    let _record = left.extend(right);

    // Many + One
    let mut left = Record::from(1, [9, 43, 1, 2]);
    let right = Record::new(0, 3);
    let _record = left.extend(right);

    // Many + Many
    let mut left = Record::from(1, [9, 43, 1, 2]);
    let right = Record::from(312, [4, 6, 7, 10]);
    let _record = left.extend(right);
}
