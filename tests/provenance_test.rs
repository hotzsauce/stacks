use rusty_stacks::provenance::Provenance;

fn simple_prov() -> Provenance<i32> {
    Provenance::from_source_and_mass("rincewind", vec![3, 4, 5])
}

#[test]
fn init_prov_tests() {
    let _prov = Provenance::<f64>::new();
    let _prov = Provenance::from_source_and_mass("rincewind", vec![4, 3, 5]);
}

#[test]
fn prov_clone() {
    let prov = simple_prov();
    let _clone = prov.clone();

    // for (p, c) in prov.records.iter().zip(clone.records.iter()) {
    //     assert_eq!(p, c);
    // }
}
