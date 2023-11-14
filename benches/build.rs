#[cfg(test)]
#[macro_use]
extern crate bencher;

use bencher::Bencher;

use boomphf::Mphf;

fn build1_ser_u64(bench: &mut Bencher) {
    let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
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

fn scan1_ser(bench: &mut Bencher) {
    let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
    let phf = Mphf::new(2.0, &items);

    bench.iter(|| {
        for i in (0..1000000u64).map(|x| x * 2) {
            std::hint::black_box(phf.hash(&i));
        }
    });
}

benchmark_group!(benches, build1_ser_u64, build1_ser_slices, build1_par_u64, build1_par_slices, scan1_ser);
benchmark_main!(benches);
