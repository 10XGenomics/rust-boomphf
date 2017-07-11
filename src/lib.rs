extern crate fxhash;

mod bit_vector;
use bit_vector::*;

use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;

// Fast and scalable minimal perfect hashing for massive key sets
// https://arxiv.org/abs/1702.03154


pub const SEEDS: [u64; 10] = [0,1,2,3,4,5,6,7,8,9];

pub struct Mphf<T: Hash + Clone> {
	bitvecs: Vec<BitVector>,
	ranks: Vec<Vec<u64>>,
	phantom: PhantomData<T>
}


pub fn hash<T: Hash>(mut x: u64, v: &T) -> u64 {
    let mut state = fxhash::FxHasher::default();

	x = x ^ (x >> 12); // a
	x = x ^ (x << 25); // b
	x = x ^ (x >> 27); // c
	x = x.wrapping_mul(2685821657736338717);

	state.write_u64(x);
    v.hash(&mut state);
    state.finish()
}

impl<T: Hash + Clone> Mphf<T> {
	pub fn new(gamma: f64, start_keys: Vec<T>) -> Mphf<T> {
		// FIXME - don't require owned Vec
		// be more memory efficient.
		let mut bitvecs = Vec::new();
		let mut keys = start_keys;
		let mut iter = 0;

		while keys.len() > 0 {

			let size = std::cmp::max(255, (gamma * keys.len() as f64) as u64);
			let mut a = BitVector::new(size as usize);
			let mut collide = BitVector::new(size as usize);

			let seed = SEEDS[iter];

			for v in keys.iter() {
				let idx = hash(seed, v) % size;

				if collide.contains(idx as usize) {
					continue;
				}

				let a_was_set = !a.insert(idx as usize);
				if a_was_set {
					collide.insert(idx as usize);
				}
			}

			let mut redo_keys = Vec::new();
			for v in keys.iter() {
				let idx = hash(seed, v) % size;

				if collide.contains(idx as usize) {
					redo_keys.push(v.clone());
					a.remove(idx as usize);
				}
			}

			bitvecs.push(a);
			keys = redo_keys;
			iter += 1;
		}

		let ranks = Self::compute_ranks(&bitvecs);
		Mphf { bitvecs: bitvecs, ranks: ranks, phantom: PhantomData }
	}

	fn compute_ranks(bvs: &Vec<BitVector>) -> Vec<Vec<u64>> {
		let mut ranks = Vec::new();
		let mut pop = 0 as u64;

		for bv in bvs {
			let mut rank: Vec<u64> = Vec::new();
			for i in 0 .. bv.num_words() {
				let v = bv.get_word(i);

				if i % 8 == 0 {
					rank.push(pop)
				}

				pop += v.count_ones() as u64;
			}

			ranks.push(rank)
		}

		ranks 
	}

	#[inline]
	fn get_rank(&self, hash: u64, i: usize) -> u64 {

		let idx = hash as usize;
		let bv = self.bitvecs.get(i).expect("that level doesn't exist");
		let ranks = self.ranks.get(i).expect("that level doesn't exist");

		// Last pre-computed rank
		let mut rank = ranks[idx/512];

		// Add rank of intervening words
		for j in (idx / 64) & !7 .. idx/64 {
			rank += bv.get_word(j).count_ones() as u64;
		}

		// Add rank of final word up to hash
		let final_word = bv.get_word(idx/64);
		if idx % 64 > 0 {
			rank += (final_word << (64 - (idx % 64))).count_ones() as u64;
		}
		rank
	}


	pub fn query(&self, v: &T) -> u64 {

		for (iter, bv) in self.bitvecs.iter().enumerate() {
			let hash = hash(SEEDS[iter], v) % (bv.capacity() as u64);

			if bv.contains(hash as usize) {
				return self.get_rank(hash, iter);
			}
		}

		unreachable!("must find a hash value");
	}
}


#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
mod tests {
	use super::*;
	use std::collections::HashSet;

	fn check_mphf<T>(xs: HashSet<T>) -> bool where T: Hash + PartialEq + Eq + Clone {

		let mut xsv: Vec<T> = Vec::new();
		xsv.extend(xs);
		let n = xsv.len();

		let phf = Mphf::new(1.5, xsv.clone());

		let mut hashes = Vec::new();

		for v in xsv {
			hashes.push(phf.query(&v));
		}

		hashes.sort();

		let gt: Vec<u64> = (0 .. n as u64).collect();
		hashes == gt
	}

	quickcheck! {
		fn check_u32(v: HashSet<u32>) -> bool {
			check_mphf(v)
		}
	}

	quickcheck! {
		fn check_isize(v: HashSet<isize>) -> bool {
			check_mphf(v)
		}
	}

	quickcheck! {
		fn check_u64(v: HashSet<u64>) -> bool {
			check_mphf(v)
		}
	}

	quickcheck! {
		fn check_vec_u8(v: HashSet<Vec<u8>>) -> bool {
			check_mphf(v)
		}
	}

	quickcheck! {
		fn check_string(v: HashSet<Vec<String>>) -> bool {
			check_mphf(v)
		}
	}
}