use ndarray::array;
use phylo2vec::tree_vec::ops::matrix::pre_precision;

fn main() {
    let m = array![[0.0, 0.5, 0.7], [0.0, 0.1, 0.2], [1.0, 0.3, 0.4]];

    let p = pre_precision(&m.view());

    println!("Precision: {:?}", p);
}
