use criterion::criterion_main;

mod benchmarks;

// criterion_main!(benchmarks::core::core);
criterion_main!(benchmarks::ndarray_vs_vec::ndarray_vs_vec);
