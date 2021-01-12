#[cfg(test)]
#[macro_use]
extern crate bencher;

extern crate boomphf;

use bencher::Bencher;

use boomphf::*;

fn build1_ser(bench: &mut Bencher) {
    bench.iter(|| {
        let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
        let _ = Mphf::new(2.0, &items);
    });
}

fn build1_par(bench: &mut Bencher) {
    bench.iter(|| {
        let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
        let _ = Mphf::new_parallel(2.0, &items, None);
    });
}

fn scan1_ser(bench: &mut Bencher) {
    let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
    let phf = Mphf::new(2.0, &items);

    bench.iter(|| {
        for i in (0..1000000u64).map(|x| x * 2) {
            phf.hash(&i);
        }
    });
}

benchmark_group!(benches, build1_ser, build1_par, scan1_ser);
benchmark_main!(benches);
