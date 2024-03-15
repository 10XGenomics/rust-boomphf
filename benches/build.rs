#[cfg(test)]
#[macro_use]
extern crate bencher;

use bencher::Bencher;

use boomphf::{ExternallyHashed, Mphf};

fn build1_ser_u64(bench: &mut Bencher) {
    let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
    bench.iter(|| {
        std::hint::black_box(Mphf::new(2.0, &items));
    });
}

fn build1_ser_externally_hashed(bench: &mut Bencher) {
    let items: Vec<ExternallyHashed> = (0..1000000u64)
        .map(|x| ExternallyHashed(wyhash::wyrng(&mut (x * 2))))
        .collect();
    bench.iter(|| {
        std::hint::black_box(Mphf::new(2.0, &items));
    });
}

fn build1_ser_slices(bench: &mut Bencher) {
    let items: Vec<[u8; 8]> = (0..1000000u64).map(|x| (x * 2).to_le_bytes()).collect();
    bench.iter(|| {
        std::hint::black_box(Mphf::new(2.0, &items));
    });
}

fn build1_ser_long_slices(bench: &mut Bencher) {
    let items = (0..1000000u64)
        .map(|x| {
            let mut long_key = [0u8; 128];
            long_key[0..8].copy_from_slice(&(x * 2).to_le_bytes());
            long_key
        })
        .collect::<Vec<_>>();
    bench.iter(|| {
        std::hint::black_box(Mphf::new(2.0, &items));
    });
}

fn build1_ser_long_slices_externally_hashed(bench: &mut Bencher) {
    let items = (0..1000000u64)
        .map(|x| {
            let mut long_key = [0u8; 128];
            long_key[0..8].copy_from_slice(&(x * 2).to_le_bytes());
            ExternallyHashed(wyhash::wyhash(&long_key, 0))
        })
        .collect::<Vec<_>>();
    bench.iter(|| {
        std::hint::black_box(Mphf::new(2.0, &items));
    });
}

#[allow(dead_code)]
fn build1_par_u64(bench: &mut Bencher) {
    let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
    #[cfg(feature = "parallel")]
    bench.iter(|| {
        std::hint::black_box(Mphf::new_parallel(2.0, &items, None));
    });
}

#[allow(dead_code)]
fn build1_par_slices(bench: &mut Bencher) {
    let items: Vec<[u8; 8]> = (0..1000000u64).map(|x| (x * 2).to_le_bytes()).collect();
    #[cfg(feature = "parallel")]
    bench.iter(|| {
        std::hint::black_box(Mphf::new_parallel(2.0, &items, None));
    });
}

fn scan1_ser_u64(bench: &mut Bencher) {
    let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
    let phf = Mphf::new(2.0, &items);

    bench.iter(|| {
        for i in &items {
            std::hint::black_box(phf.hash(&i));
        }
    });
}

fn scan1_ser_slice(bench: &mut Bencher) {
    let items: Vec<[u8; 8]> = (0..1000000u64).map(|x| (x * 2).to_le_bytes()).collect();
    let phf = Mphf::new(2.0, &items);

    bench.iter(|| {
        for i in &items {
            std::hint::black_box(phf.hash(i));
        }
    });
}

fn scan1_ser_externally_hashed(bench: &mut Bencher) {
    let items: Vec<ExternallyHashed> = (0..1000000u64)
        .map(|x| ExternallyHashed(wyhash::wyrng(&mut (x * 2))))
        .collect();
    let phf = Mphf::new(2.0, &items);

    bench.iter(|| {
        for i in &items {
            std::hint::black_box(phf.hash(i));
        }
    });
}

fn scan1_ser_long_key(bench: &mut Bencher) {
    let items = (0..1000000u64)
        .map(|x| {
            let mut long_key = [0u8; 128];
            long_key[0..8].copy_from_slice(&(x * 2).to_le_bytes());
            long_key
        })
        .collect::<Vec<_>>();
    let phf = Mphf::new(2.0, &items);

    bench.iter(|| {
        for i in &items {
            std::hint::black_box(phf.hash(i));
        }
    });
}

fn scan1_ser_long_key_externally_hashed(bench: &mut Bencher) {
    let items: Vec<ExternallyHashed> = (0..1000000u64)
        .map(|x| {
            let mut long_key = [0u8; 128];
            long_key[0..8].copy_from_slice(&(x * 2).to_le_bytes());
            ExternallyHashed(wyhash::wyhash(&long_key, 0))
        })
        .collect();
    let phf = Mphf::new(2.0, &items);

    bench.iter(|| {
        for i in &items {
            std::hint::black_box(phf.hash(i));
        }
    });
}

benchmark_group!(
    benches,
    build1_ser_externally_hashed,
    build1_ser_u64,
    build1_ser_slices,
    build1_ser_long_slices,
    build1_ser_long_slices_externally_hashed,
    build1_par_u64,
    build1_par_slices,
    scan1_ser_u64,
    scan1_ser_slice,
    scan1_ser_externally_hashed,
    scan1_ser_long_key,
    scan1_ser_long_key_externally_hashed
);
benchmark_main!(benches);
