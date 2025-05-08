use rusty_stacks::stack::{DecStack, Decreasing, IncStack, Stack};

fn three_stack() -> IncStack<i32, f64> {
    let x = vec![3, 4, 5];
    let y = vec![-2.0, 0.5, 200.4];
    IncStack::try_from_vectors(x, y).unwrap()
}

fn four_stack_dec() -> DecStack<i32, f64> {
    DecStack::try_from_vectors(vec![2, 2, 4, 8], vec![6.7, 0.5, -0.4, -2.3]).unwrap()
}

fn four_stack_inc() -> IncStack<i32, f64> {
    IncStack::try_from_vectors(vec![2, 2, 4, 8], vec![-2.3, -0.4, 0.5, 6.7]).unwrap()
}

fn five_stack_dec() -> DecStack<i32, f64> {
    DecStack::try_from_vectors(vec![2, 3, 5, 8, 13], vec![6.7, 0.5, 0.0, -0.4, -2.3]).unwrap()
}

fn five_stack_inc() -> IncStack<i32, f64> {
    IncStack::try_from_vectors(vec![2, 3, 5, 8, 13], vec![-2.3, -0.4, 0.0, 0.5, 6.7]).unwrap()
}

#[test]
fn init_stack_new() {
    let _blank_stack = Stack::<f64, f64>::new();
}

#[test]
fn init_stack_inc() {
    let x = vec![3, 4, 5];
    let y = vec![-12, 0, 4];
    let _simple_stack = IncStack::try_from_vectors(x, y).unwrap();
}

#[test]
fn init_stack_dec() {
    let x = vec![3, 4, 5];
    let y = vec![-2, -3, -10];
    let _simple_stack = Stack::<_, _, Decreasing>::try_from_vectors(x, y).unwrap();
}

#[test]
#[should_panic]
fn init_stack_length_mismatch() {
    let x = vec![3, 4, 5];
    let y = vec![-12, 0, 4, 30];
    let _simple_stack = Stack::<_, _, Decreasing>::try_from_vectors(x, y).unwrap();
}

#[test]
fn stack_inc_addition() {
    let x = vec![3, 4, 6];
    let y = vec![0.1, 0.3, 0.6];
    let left = IncStack::try_from_vectors_and_source(x, y, "rincewind").unwrap();

    let x = vec![2, 3, 5];
    let y = vec![0.1, 0.3, 0.6];
    let right = IncStack::try_from_vectors_and_source(x, y, "mortimer").unwrap();

    let sum = left + right;
    println!("{:?}", &sum);
}

#[test]
fn stack_cumulate() {
    let stack = three_stack();
    assert_eq!(stack.cumulate(), vec![3, 7, 12]);
}

#[test]
fn stack_inc_total_above() {
    let stack = four_stack_inc();
    let ta = stack.total_above(0.0);

    assert_eq!(ta, 12);
}

#[test]
fn stack_dec_total_above() {
    let stack = four_stack_dec();
    let ta = stack.total_above(0.0);

    assert_eq!(ta, 4);
}

#[test]
fn stack_inc_total_below() {
    let stack = four_stack_inc();
    let ta = stack.total_below(0.0);

    assert_eq!(ta, 4);
}

#[test]
fn stack_dec_total_below() {
    let stack = four_stack_dec();
    let ta = stack.total_below(0.0);

    assert_eq!(ta, 12);
}

#[test]
fn stack_inc_clip() {
    let stack = four_stack_inc();
    let clipped = stack.clip(10);

    assert_eq!(clipped.x, vec![2, 2, 4]);
    assert_eq!(clipped.y, vec![-2.3, -0.4, 0.5]);
}

#[test]
fn stack_inc_truncate() {
    let stack = five_stack_inc();
    let clipped = stack.truncate(0.0);

    assert_eq!(clipped.x, vec![2, 3, 5]);
    assert_eq!(clipped.y, vec![-2.3, -0.4, 0.0]);
}

#[test]
fn stack_dec_truncate() {
    let stack = five_stack_dec();
    let clipped = stack.truncate(0.0);

    assert_eq!(clipped.x, vec![2, 3, 5]);
    assert_eq!(clipped.y, vec![6.7, 0.5, 0.0]);
}

#[test]
fn incstack_gt_cmp() {
    let x = vec![2, 3, 5, 8];
    let y = vec![-2, 1, 3, 6];
    let l = IncStack::try_from_vectors(x, y).unwrap();

    let x = vec![2, 3, 5, 8];
    let y = vec![-1, 2, 4, 7];
    let r = IncStack::try_from_vectors(x, y).unwrap();

    let gt = l.partial_cmp(&r);
    assert_eq!(gt, Some(std::cmp::Ordering::Greater));
    assert_ne!(l, r);
}

#[test]
fn incstack_lt_cmp() {
    let x = vec![2, 3, 5, 8];
    let y = vec![-1, 2, 4, 7];
    let l = IncStack::try_from_vectors(x, y).unwrap();

    let x = vec![2, 3, 5, 8];
    let y = vec![-2, 1, 3, 6];
    let r = IncStack::try_from_vectors(x, y).unwrap();

    let lt = l.partial_cmp(&r);
    assert_eq!(lt, Some(std::cmp::Ordering::Less));
    assert_ne!(l, r);
}

#[test]
fn incstack_no_cmp() {
    let x = vec![2, 3, 5, 8];
    let y = vec![-1, 2, 4, 7];
    let l = IncStack::try_from_vectors(x, y).unwrap();

    let x = vec![2, 3, 5, 8];
    let y = vec![-2, 1, 5, 8];
    let r = IncStack::try_from_vectors(x, y).unwrap();

    let no = l.partial_cmp(&r);
    assert_eq!(no, None);
    assert_ne!(l, r);
}

#[test]
fn decstack_gt_cmp() {
    let x = vec![2, 3, 5, 8];
    let y = vec![7, 4, 2, -1];
    let l = DecStack::try_from_vectors(x, y).unwrap();

    let x = vec![2, 3, 5, 8];
    let y = vec![6, 3, 1, -2];
    let r = DecStack::try_from_vectors(x, y).unwrap();

    let lt = l.partial_cmp(&r);
    assert_eq!(lt, Some(std::cmp::Ordering::Greater));
    assert_ne!(l, r);
}

#[test]
fn decstack_lt_cmp() {
    let x = vec![2, 3, 5, 8];
    let y = vec![6, 3, 1, -2];
    let l = DecStack::try_from_vectors(x, y).unwrap();

    let x = vec![2, 3, 5, 8];
    let y = vec![7, 4, 2, -1];
    let r = DecStack::try_from_vectors(x, y).unwrap();

    let lt = l.partial_cmp(&r);
    assert_eq!(lt, Some(std::cmp::Ordering::Less));
    assert_ne!(l, r);
}

#[test]
fn decstack_no_cmp() {
    let x = vec![2, 3, 5, 8];
    let y = vec![6, 3, 1, -2];
    let l = DecStack::try_from_vectors(x, y).unwrap();

    let x = vec![2, 3, 5, 8];
    let y = vec![7, 4, -1, -3];
    let r = DecStack::try_from_vectors(x, y).unwrap();

    let lt = l.partial_cmp(&r);
    assert_eq!(lt, None);
    assert_ne!(l, r);
}
