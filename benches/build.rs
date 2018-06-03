#[cfg(test)]
#[macro_use]
extern crate bencher;

extern crate boomphf;

use bencher::Bencher;

use boomphf::*;

fn build1(bench: &mut Bencher) {
    bench.iter(|| {
        let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
        let phf = Mphf::new(2.0, &items, None);
    });
}

fn build1_par(bench: &mut Bencher) {
    bench.iter(|| {
        let items: Vec<u64> = (0..1000000u64).map(|x| x * 2).collect();
        let phf = Mphf::new_parallel(2.0, &items, None);
    });
}

benchmark_group!(benches, build1, build1_par);
benchmark_main!(benches);
