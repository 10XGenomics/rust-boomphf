
#[cfg(test)]
#[macro_use]
extern crate bencher;

extern crate boomphf;

use bencher::Bencher;

use boomphf::*;


fn build1(bench: &mut Bencher) {
    bench.iter(|| {
        let items: Vec<u64> = (0..1000000u64).map(|x| x*2).collect();
        let phf = Mphf::new(2.0, &items, None);
    });
}


fn a(bench: &mut Bencher) {
    bench.iter(|| {
        (0..1000).fold(0, |x, y| x + y)
    })
}

fn b(bench: &mut Bencher) {
    const N: usize = 1024;
    bench.iter(|| {
        vec![0u8; N]
    });

    bench.bytes = N as u64;
}

benchmark_group!(benches, a, b, build1);
benchmark_main!(benches);
