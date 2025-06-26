use criterion::{criterion_group, BenchmarkId, Criterion};
use phylo2vec::tree_vec::ops::matrix::{pre_precision, pre_precision_old};
use phylo2vec::utils::{sample_matrix, sample_matrix_old, sample_vector, sample_vector_ndarray};
use std::ops::RangeInclusive;
use std::time::Duration;

const SAMPLE_SIZES: RangeInclusive<usize> = 1..=10;

fn bench_sample_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_vector");
    // Set logarithmic scale for plot
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Linear),
    );

    for i in SAMPLE_SIZES {
        let sample_size = 10000 * i;

        group.bench_with_input(
            BenchmarkId::new("ndarray", sample_size),
            &sample_size,
            |b, &size| {
                b.iter(|| sample_vector_ndarray(size, false));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec", sample_size),
            &sample_size,
            |b, &size| {
                b.iter(|| sample_vector(size, false));
            },
        );
    }
    group.finish();
}

fn bench_sample_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_matrix");
    // Set logarithmic scale for plot
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Linear),
    );

    for i in SAMPLE_SIZES {
        let sample_size = 100 * i;

        group.bench_with_input(
            BenchmarkId::new("ndarray", sample_size),
            &sample_size,
            |b, &size| {
                b.iter(|| sample_matrix(size, false));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec", sample_size),
            &sample_size,
            |b, &size| {
                b.iter(|| sample_matrix_old(size, false));
            },
        );
    }
    group.finish();
}

fn bench_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision");
    // Set logarithmic scale for plot
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for i in SAMPLE_SIZES {
        let sample_size = 10 * i;

        group.bench_with_input(
            BenchmarkId::new("ndarray", sample_size),
            &sample_size,
            |b, &size| {
                let m = sample_matrix(size, false);

                b.iter(|| pre_precision(&m.view()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec", sample_size),
            &sample_size,
            |b, &size| {
                let m = sample_matrix_old(size, false);

                b.iter(|| pre_precision_old(&m));
            },
        );
    }
    group.finish();
}

fn bench_ancestry(c: &mut Criterion) {
    let mut group = c.benchmark_group("ancestry");
    // Set logarithmic scale for plot
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for i in SAMPLE_SIZES {
        let sample_size = 100 * i;

        group.bench_with_input(
            BenchmarkId::new("ndarray", sample_size),
            &sample_size,
            |b, &sample_size| {
                let v = sample_vector(sample_size, false);

                b.iter(|| phylo2vec::tree_vec::ops::vector::get_ancestry_ndarray(&v));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec", sample_size),
            &sample_size,
            |b, &sample_size| {
                let v = sample_vector(sample_size, false);

                b.iter(|| phylo2vec::tree_vec::ops::vector::get_ancestry(&v));
            },
        );
    }
    group.finish();
}

fn bench_make_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("make_tree");
    // Set logarithmic scale for plot
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    for i in SAMPLE_SIZES {
        let sample_size = 1000 * i;

        group.bench_with_input(
            BenchmarkId::new("ndarray", sample_size),
            &sample_size,
            |b, &size| {
                let v = sample_vector_ndarray(size, false);

                b.iter(|| phylo2vec::tree_vec::ops::vector::make_tree_ndarray(&v.view()));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("vec", sample_size),
            &sample_size,
            |b, &size| {
                let v = sample_vector(size, false);

                b.iter(|| phylo2vec::tree_vec::ops::vector::make_tree(&v));
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = ndarray_vs_vec;
    config = Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(1000));
    targets = bench_make_tree
}
