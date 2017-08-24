// Copyright (c) 2014 10X Genomics, Inc. All rights reserved.

//! ### boomphf - Fast and scalable minimal perfect hashing for massive key sets
//! A Rust implementation of "Fast and scalable minimal perfect hashing for massive key sets"
//! [https://arxiv.org/abs/1702.03154](https://arxiv.org/abs/1702.03154). The library generates
//! a minimal perfect hash function (MPHF) for a collection of hashable objects. Note: minimal
//! perfect hash functions can only be used with the set of objects used when hash function
//! was created. Hashing a new object will return an arbitrary hash value. If your use case
//! may result in hashing new values, you will need an auxiliary scheme to detect this condition.
//!
//! ```
//! use boomphf::*;
//! // Generate MPHF
//! let possible_objects = vec![1, 10, 1000, 23, 457, 856, 845, 124, 912];
//! let n = possible_objects.len();
//! let phf = Mphf::new(1.7, possible_objects.clone(), None);
//! // Get hash value of all objects
//! let mut hashes = Vec::new();
//! for v in possible_objects {
//!		hashes.push(phf.hash(&v));
//!	}
//!	hashes.sort();
//!
//! // Expected hash output is set of all integers from 0..n
//! let expected_hashes: Vec<u64> = (0 .. n as u64).collect();
//!	assert!(hashes == expected_hashes)
//! ```

extern crate fnv;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

mod bitvector;
use bitvector::*;

use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::fmt::Debug;

/// A minimal perfect hash function over a set of objects of type `T`.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mphf<T: Hash + Clone + Debug> {
	bitvecs: Vec<BitVector>,
	ranks: Vec<Vec<u64>>,
	phantom: PhantomData<T>
}


fn hash_with_seed<T: Hash>(iter: u64, v: &T) -> u64 {
	let mut state = fnv::FnvHasher::with_key(iter);
    v.hash(&mut state);
    state.finish()
}

impl<T: Hash + Clone + Debug> Mphf<T> {
	/// Generate a minimal perfect hash function for the set of `objects`.
	/// `objects` must not contain any duplicate items.
        /// `gamma` controls the tradeoff between the construction-time and run-time speed,
        /// and the size of the datastructure representing the hash function. See the paper for details.
        /// `max_iters` - None to never stop trying to find a perfect hash (safe if no duplicates).

	pub fn new(gamma: f64, objects: Vec<T>, max_iters: Option<u64>) -> Mphf<T> {
		// FIXME - don't require owned Vec
		// be more memory efficient.
		let mut bitvecs = Vec::new();
		let mut keys = objects;
		let mut iter = 0;

		assert!(gamma > 1.01);

		while keys.len() > 0 {
			if max_iters.is_some() && iter > max_iters.unwrap() {
				println!("ran out of key space. items: {:?}", keys);
				panic!("counldn't find unique hashes");
			}

			let size = std::cmp::max(255, (gamma * keys.len() as f64) as u64);
			let mut a = BitVector::new(size as usize);
			let mut collide = BitVector::new(size as usize);

			let seed = iter;

			for v in keys.iter() {
				let idx = hash_with_seed(seed, v) % size;

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
				let idx = hash_with_seed(seed, v) % size;

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

	/// Compute the hash value of `item`. This method should only be used
	/// with items known to be in construction set. Use `try_hash` you cannot
	/// guarantee that `item` was in the construction set. If `item` was not present
	/// in the construction set this function may panic.
	pub fn hash(&self, item: &T) -> u64 {

		for (iter, bv) in self.bitvecs.iter().enumerate() {
			let hash = hash_with_seed(iter as u64, item) % (bv.capacity() as u64);

			if bv.contains(hash as usize) {
				return self.get_rank(hash, iter);
			}
		}

		unreachable!("must find a hash value");
	}

	/// Compute the hash value of `item`. If `item` was not present
	/// in the set of objects used to construct the hash function, the return
	/// value will an arbitrary value Some(x), or None.
	pub fn try_hash(&self, item: &T) -> Option<u64> {

		for (iter, bv) in self.bitvecs.iter().enumerate() {
			let hash = hash_with_seed(iter as u64, item) % (bv.capacity() as u64);

			if bv.contains(hash as usize) {
				return Some(self.get_rank(hash, iter))
			}
		}

		None
	}
}


#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
mod tests {

	use super::*;
	use std::collections::HashSet;
	use std::iter::FromIterator;

	fn check_mphf<T>(xs: HashSet<T>) -> bool where T: Hash + PartialEq + Eq + Clone + Debug {

		let mut xsv: Vec<T> = Vec::new();
		xsv.extend(xs);
		let n = xsv.len();

		let phf = Mphf::new(1.7, xsv.clone(), None);

		let mut hashes = Vec::new();

		for v in xsv {
			hashes.push(phf.hash(&v));
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


	#[test]
	fn from_ints() {
		let items = (0..1000000).map(|x| x*2);
		check_mphf(HashSet::from_iter(items));
	}
}
