use std::collections::HashMap;
use std::iter::FromIterator;

use criterion::{self, criterion_group, criterion_main, Criterion};

use boomphf::hashmap::BoomHashMap;
use boomphf::Mphf;

const UPPER_BOUND: u64 = 1000000;

fn build(c: &mut Criterion) {
    let mut group = c.benchmark_group("build");
    let keys: Vec<u64> = (0..UPPER_BOUND).map(|x| x * 2).collect();
    let values: Vec<u64> = (0..UPPER_BOUND).collect();

    group.bench_function("boomphf_serial", |b| {
        b.iter(|| {
            criterion::black_box(Mphf::new(2.0, &keys));
        });
    });

    group.bench_function("boomphf_parallel", |b| {
        b.iter(|| {
            criterion::black_box(Mphf::new_parallel(2.0, &keys, None));
        });
    });

    group.bench_function("boomphf_hashmap", |b| {
        b.iter(|| {
            criterion::black_box({
                let _: BoomHashMap<u64, u64> = BoomHashMap::new(keys.clone(), values.clone());
            });
        });
    });

    group.bench_function("hashmap", |b| {
        b.iter(|| {
            criterion::black_box({
                let _: HashMap<u64, u64> =
                    HashMap::from_iter(keys.iter().cloned().zip(values.iter().cloned()));
            });
        });
    });
}

fn scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan_hashes");
    let items: Vec<u64> = (0..UPPER_BOUND).map(|x| x * 2).collect();
    let phf = Mphf::new(2.0, &items);

    group.bench_function("boomphf_scan", |b| {
        b.iter(|| {
            for i in (0..UPPER_BOUND).map(|x| x * 2) {
                phf.hash(&i);
            }
        });
    });
}

fn lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup");
    let keys: Vec<u64> = (0..UPPER_BOUND).map(|x| x * 2).collect();
    let values: Vec<u64> = (0..UPPER_BOUND).collect();

    let hashmap: HashMap<u64, u64> =
        HashMap::from_iter(keys.iter().cloned().zip(values.iter().cloned()));
    let boom_hashmap = BoomHashMap::new(keys, values);

    group.bench_function("hashmap", |b| {
        b.iter(|| {
            for k in (0..UPPER_BOUND).map(|x| x * 2) {
                criterion::black_box(hashmap.get(&k));
            }
        });
    });

    group.bench_function("boomphf_hashmap", |b| {
        b.iter(|| {
            for k in (0..UPPER_BOUND).map(|x| x * 2) {
                criterion::black_box(boom_hashmap.get(&k));
            }
        });
    });
}

criterion_group!(benches, build, scan, lookup,);
criterion_main!(benches);
