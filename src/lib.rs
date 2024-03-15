// Copyright (c) 2017 10X Genomics, Inc. All rights reserved.
// Copyright (c) 2015 Guillaume Rizk
// Some portions of this code are derived from https://github.com/rizkg/BBHash (MIT license)

//! ### boomphf - Fast and scalable minimal perfect hashing for massive key sets
//! A Rust implementation of the BBHash method for constructing minimal perfect hash functions,
//! as described in "Fast and scalable minimal perfect hashing for massive key sets"
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
//! let phf = Mphf::new(1.7, &possible_objects);
//! // Get hash value of all objects
//! let mut hashes = Vec::new();
//! for v in possible_objects {
//!     hashes.push(phf.hash(&v));
//! }
//! hashes.sort();
//!
//! // Expected hash output is set of all integers from 0..n
//! let expected_hashes: Vec<u64> = (0 .. n as u64).collect();
//! assert!(hashes == expected_hashes)
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod bitvector;
pub mod hashmap;
#[cfg(feature = "parallel")]
mod par_iter;
use bitvector::BitVector;
#[cfg(target_endian = "little")]
use bitvector::BitVectorRef;

use log::error;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "parallel")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "serde")]
use serde::{self, Deserialize, Serialize};

/// fastmod used to construct the seed as 1 << (iters + iters). However, for external hashing
/// there's a faster path available via lookup tables if we just pass in iters. This method is
/// to ensure that pre-existing hashes continue to work as before when not using ExternallyHashed.
#[inline(always)]
fn default_seed_correction(seed: u64) -> u64 {
    1 << (seed + seed)
}

fn default_hash_with_seed<T: Hash>(value: &T, seed: u64) -> u64 {
    let mut state = wyhash::WyHash::with_seed(1 << (seed + seed));
    value.hash(&mut state);
    state.finish()
}

// This custom trait allows us to fast-path &[u8] to avoid constructing the temporary Hasher object.
// Can be simplified once specialization is stabilized.
pub trait SeedableHash {
    fn hash_with_seed(&self, seed: u64) -> u64;
}

impl SeedableHash for [u8] {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(self, default_seed_correction(seed))
    }
}

impl<const N: usize> SeedableHash for [u8; N] {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(self, default_seed_correction(seed))
    }
}

impl SeedableHash for u8 {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&[*self], default_seed_correction(seed))
    }
}

impl SeedableHash for i16 {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl SeedableHash for u16 {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl SeedableHash for i32 {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl SeedableHash for u32 {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl SeedableHash for i64 {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl SeedableHash for u64 {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl SeedableHash for isize {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl SeedableHash for usize {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        wyhash::wyhash(&self.to_le_bytes(), default_seed_correction(seed))
    }
}

impl<T: SeedableHash> SeedableHash for &T {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        (**self).hash_with_seed(seed)
    }
}

impl<T: Hash> SeedableHash for &[T] {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        default_hash_with_seed(self, seed)
    }
}

impl<T: Hash> SeedableHash for Vec<T> {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        default_hash_with_seed(self, seed)
    }
}

impl SeedableHash for &str {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        default_hash_with_seed(self, seed)
    }
}

impl SeedableHash for String {
    fn hash_with_seed(&self, seed: u64) -> u64 {
        default_hash_with_seed(self, seed)
    }
}

/// This is a fast-path where the hash for an entry is known externally. That way we can skip hashing the
/// key for building / lookups which provides savings as keys grow longer or you need to do a lookup of the
/// same key across multiple perfect hashes. It's the user's responsibility to construct this with a value
/// that is deterministically derived from a key.
#[derive(Debug, Clone, Copy)]
pub struct ExternallyHashed(pub u64);

impl ExternallyHashed {
    // Helper function for wyrng.
    const fn wymum(a: u64, b: u64) -> u64 {
        let mul = a as u128 * b as u128;
        ((mul >> 64) ^ mul) as u64
    }

    // wyrng except a constified version
    const fn wyrng(seed: u64) -> u64 {
        const P0: u64 = 0xa076_1d64_78bd_642f;
        const P1: u64 = 0xe703_7ed1_a0b4_28db;

        let seed = seed.wrapping_add(P0);
        Self::wymum(seed ^ P1, seed)
    }

    // Generate lookup tables to map the hash seed to a random value.
    const fn gen_seed_lookups() -> [u64; MAX_ITERS as usize] {
        let mut result = [0; MAX_ITERS as usize];
        let mut i = 0;
        while i < MAX_ITERS {
            result[i as usize] = Self::wyrng(i);
            i += 1;
        }
        result
    }
    const SEED_HASH_LOOKUP_TABLES: [u64; MAX_ITERS as usize] = Self::gen_seed_lookups();

    // Helper utility to convert the seed passed in from hashmod (which is in 0..MAX_ITERS) into a hash.
    fn fast_seed_hash(x: u64) -> u64 {
        debug_assert!(x <= MAX_ITERS);
        Self::SEED_HASH_LOOKUP_TABLES[x as usize]
    }

    // Quickly combine two hashes. Because .0 represents a hash, we know it's random and doesn't need to be
    // independently hashed again, so we just need to combine it uniquely with iters.
    fn hash_combine(h1: u64, h2: u64) -> u64 {
        // https://stackoverflow.com/questions/5889238/why-is-xor-the-default-way-to-combine-hashes
        h1 ^ (h2
            .wrapping_add(0x517cc1b727220a95)
            .wrapping_add(h1 << 6)
            .wrapping_add(h1 >> 2))
    }
}

impl SeedableHash for ExternallyHashed {
    #[inline(always)]
    fn hash_with_seed(&self, seed: u64) -> u64 {
        Self::hash_combine(self.0, Self::fast_seed_hash(seed))
    }
}

#[inline]
fn fold(v: u64) -> u32 {
    ((v & 0xFFFFFFFF) as u32) ^ ((v >> 32) as u32)
}

#[inline]
fn fastmod(hash: u32, n: u32) -> u64 {
    ((hash as u64) * (n as u64)) >> 32
}

#[inline]
fn hashmod<T: SeedableHash + ?Sized>(iter: u64, v: &T, n: u64) -> u64 {
    // when n < 2^32, use the fast alternative to modulo described here:
    // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    let h = v.hash_with_seed(iter);
    if n < (1 << 32) {
        fastmod(fold(h), n as u32) as u64
    } else {
        h % n
    }
}

#[cfg(target_endian = "little")]
pub(crate) fn u8_slice_cast<T>(s: &[u8]) -> &[T] {
    assert_eq!(s.len() % std::mem::size_of::<T>(), 0, "Invalid length");
    assert_eq!(
        s.as_ptr().align_offset(std::mem::size_of::<T>()),
        0,
        "Misaligned - not safe"
    );

    let converted_length = s.len() / std::mem::size_of::<T>();
    let converted: *const [u8] = s;
    unsafe { std::slice::from_raw_parts(converted as *const T, converted_length) }
}

/// Can only be used for lookups and same interface as Mphf. This is useful if you have the serialized
/// form stored somewhere in a raw uncompressed form and just want to reference it cheaply without copying
/// the bulk of the underlying data. In the future it might be possible to define an efficient contiguous
/// layout that lets us avoid even the heap allocation although it might be tricky since this is an array
/// of tuples where each tuple itself has 2 arrays (i.e. randomly accessing the BitVectorRef is tricky
/// without defining a fast serialized representation of an index to aide in that endeavour).
///
/// Note: the type parameter doesn't actually mean anything since you can
/// launder by round-tripping through serialization (but that's true for serde as well).
#[derive(Clone, Debug)]
pub struct MphfRef<'a, T> {
    bitvecs: Box<[(BitVectorRef<'a>, &'a [u64])]>,
    phantom: PhantomData<T>,
}

impl<'a, T> MphfRef<'a, T> {
    /// This takes an iterator of slices that compose all the parts of the original [Mphf] as returned by [Mphf::to_dma].
    /// This requires some memory allocations to set up the internal data structures, but the underlying data itself isn't
    /// copied at all.
    #[cfg(target_endian = "little")]
    pub fn from_scatter_dma(mut dma: impl ExactSizeIterator<Item = &'a [u8]>) -> Self {
        assert_eq!(dma.len() % 3, 0);
        let num_bitvecs = dma.len() / 3;
        let mut bitvecs = Vec::with_capacity(num_bitvecs);

        for _ in 0..num_bitvecs {
            let bitvec = BitVectorRef::from_dma([dma.next().unwrap(), dma.next().unwrap()]);
            let raw_ranks = dma.next().unwrap();
            bitvecs.push((bitvec, u8_slice_cast(raw_ranks)));
        }

        assert_eq!(dma.next(), None, "Didn't properly consume DMA stream");

        Self {
            bitvecs: bitvecs.into_boxed_slice(),
            phantom: Default::default(),
        }
    }

    #[inline]
    fn get_rank(&self, hash: u64, i: usize) -> u64 {
        let idx = hash as usize;
        let (bv, ranks) = self.bitvecs.get(i).expect("that level doesn't exist");

        // Last pre-computed rank
        let mut rank = ranks[idx / 512];

        // Add rank of intervening words
        for j in (idx / 64) & !7..idx / 64 {
            rank += bv.get_word(j).count_ones() as u64;
        }

        // Add rank of final word up to hash
        let final_word = bv.get_word(idx / 64);
        if idx % 64 > 0 {
            rank += (final_word << (64 - (idx % 64))).count_ones() as u64;
        }
        rank
    }

    /// Compute the hash value of `item`. This method should only be used
    /// with items known to be in construction set. Use `try_hash` if you cannot
    /// guarantee that `item` was in the construction set. If `item` was not present
    /// in the construction set this function may panic.
    pub fn hash<Q>(&self, item: &Q) -> u64
    where
        T: Borrow<Q>,
        Q: ?Sized + SeedableHash,
    {
        for i in 0..self.bitvecs.len() {
            let (bv, _) = &self.bitvecs[i];
            let hash = hashmod(i as u64, item, bv.capacity());

            if bv.contains(hash) {
                return self.get_rank(hash, i);
            }
        }

        unreachable!("must find a hash value");
    }

    /// Compute the hash value of `item`. If `item` was not present
    /// in the set of objects used to construct the hash function, the return
    /// value will an arbitrary value Some(x), or None.
    pub fn try_hash<Q>(&self, item: &Q) -> Option<u64>
    where
        T: Borrow<Q>,
        Q: ?Sized + SeedableHash,
    {
        for i in 0..self.bitvecs.len() {
            let (bv, _) = &(self.bitvecs)[i];
            let hash = hashmod(i as u64, item, bv.capacity());

            if bv.contains(hash) {
                return Some(self.get_rank(hash, i));
            }
        }

        None
    }
}

impl<'a, T> PartialEq<Mphf<T>> for MphfRef<'a, T> {
    fn eq(&self, other: &Mphf<T>) -> bool {
        self.bitvecs
            .iter()
            .zip(other.bitvecs.iter())
            .all(|(b1, b2)| b1.0 == b2.0 && b1.1 == &*b2.1)
    }
}

/// A minimal perfect hash function over a set of objects of type `T`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mphf<T> {
    bitvecs: Box<[(BitVector, Box<[u64]>)]>,
    phantom: PhantomData<T>,
}

impl<T> PartialEq for Mphf<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bitvecs == other.bitvecs
    }
}

impl<T> PartialEq<MphfRef<'_, T>> for Mphf<T> {
    fn eq(&self, other: &MphfRef<'_, T>) -> bool {
        self.bitvecs
            .iter()
            .zip(other.bitvecs.iter())
            .all(|(b1, b2)| b1.0 == b2.0 && &*b1.1 == b2.1)
    }
}

const MAX_ITERS: u64 = 100;

impl<T> Mphf<T> {
    pub fn num_dma_slices(&self) -> usize {
        // See to_dma.
        self.bitvecs.len() * 3
    }

    #[cfg(target_endian = "little")]
    pub fn to_dma(&self) -> Vec<&[u8]> {
        // Each bitvec has 3 DMA regions within it: the bits value within BitVector, the vector within BitVector and the ranks.
        let mut scattered = Vec::with_capacity(self.bitvecs.len() * 3);
        for (bitvec, ranks) in self.bitvecs.iter() {
            for piece in bitvec.dma() {
                scattered.push(piece)
            }

            let ranks = &**ranks;
            let ranks_len = std::mem::size_of_val(ranks);
            let ranks_ptr: *const [u64] = ranks;
            let ranks_ptr: *const u64 = ranks_ptr as *const u64;
            scattered
                .push(unsafe { std::slice::from_raw_parts(ranks_ptr as *const u8, ranks_len) });
        }

        assert_eq!(scattered.len() % 3, 0);
        scattered
    }

    #[cfg(target_endian = "little")]
    pub fn copy_from_scatter_dma<'a>(mut dma: impl ExactSizeIterator<Item = &'a [u8]>) -> Self {
        assert_eq!(dma.len() % 3, 0);
        let num_bitvecs = dma.len() / 3;
        let mut bitvecs = Vec::with_capacity(num_bitvecs);
        for _ in 0..num_bitvecs {
            let bitvec = BitVector::copy_from_dma([dma.next().unwrap(), dma.next().unwrap()]);
            let ranks = u8_slice_cast::<u64>(dma.next().unwrap()).into();
            bitvecs.push((bitvec, ranks));
        }
        assert_eq!(dma.next(), None, "Didn't fully consume DMA slices");

        Self {
            bitvecs: bitvecs.into_boxed_slice(),
            phantom: Default::default(),
        }
    }

    fn compute_ranks(bvs: Vec<BitVector>) -> Box<[(BitVector, Box<[u64]>)]> {
        let mut ranks = Vec::new();
        let mut pop = 0_u64;

        for bv in bvs {
            let mut rank: Vec<u64> = Vec::new();
            for i in 0..bv.num_words() {
                let v = bv.get_word(i);

                if i % 8 == 0 {
                    rank.push(pop)
                }

                pop += v.count_ones() as u64;
            }

            ranks.push((bv, rank.into_boxed_slice()))
        }

        ranks.into_boxed_slice()
    }

    #[inline]
    fn get_rank(&self, hash: u64, i: usize) -> u64 {
        let idx = hash as usize;
        let (bv, ranks) = self.bitvecs.get(i).expect("that level doesn't exist");

        // Last pre-computed rank
        let mut rank = ranks[idx / 512];

        // Add rank of intervening words
        for j in (idx / 64) & !7..idx / 64 {
            rank += bv.get_word(j).count_ones() as u64;
        }

        // Add rank of final word up to hash
        let final_word = bv.get_word(idx / 64);
        if idx % 64 > 0 {
            rank += (final_word << (64 - (idx % 64))).count_ones() as u64;
        }
        rank
    }
}

impl<'a, T: 'a + SeedableHash + Debug> Mphf<T> {
    /// Constructs an MPHF from a (possibly lazy) iterator over iterators.
    /// This allows construction of very large MPHFs without holding all the keys
    /// in memory simultaneously.
    /// `objects` is an `IntoInterator` yielding a stream of `IntoIterator`s that must not contain any duplicate items.
    /// `objects` must be able to be iterated over multiple times and yield the same stream of items each time.
    /// `gamma` controls the tradeoff between the construction-time and run-time speed,
    /// and the size of the datastructure representing the hash function. See the paper for details.
    /// `n` is the total number of items that will be produced by iterating over all the input iterators.
    /// NOTE: the inner iterator `N::IntoIter` should override `nth` if there's an efficient way to skip
    /// over items when iterating.  This is important because later iterations of the MPHF construction algorithm
    /// skip most of the items.
    pub fn from_chunked_iterator<I, N>(gamma: f64, objects: &'a I, n: u64) -> Mphf<T>
    where
        &'a I: IntoIterator<Item = N>,
        N: IntoIterator<Item = T> + Send,
        <N as IntoIterator>::IntoIter: ExactSizeIterator,
        <&'a I as IntoIterator>::IntoIter: Send,
        I: Sync,
    {
        let mut iter = 0;
        let mut bitvecs = Vec::new();
        #[allow(unused_mut)]
        let mut done_keys = BitVector::new(std::cmp::max(255, n));

        assert!(gamma > 1.01);

        loop {
            if iter > MAX_ITERS {
                error!("ran out of key space. items: {:?}", done_keys.len());
                panic!("couldn't find unique hashes");
            }

            let keys_remaining = if iter == 0 {
                n
            } else {
                n - (done_keys.len() as u64)
            };

            let size = std::cmp::max(255, (gamma * keys_remaining as f64) as u64);

            let mut a = BitVector::new(size);
            let mut collide = BitVector::new(size);

            let seed = iter;
            let mut offset = 0u64;

            for object in objects {
                let mut object_iter = object.into_iter();

                // Note: we will use Iterator::nth() to advance the iterator if
                // we've skipped over some items.
                let mut object_pos = 0;
                let len = object_iter.len() as u64;

                for object_index in 0..len {
                    let index = offset + object_index;

                    if !done_keys.contains(index) {
                        let key = match object_iter.nth((object_index - object_pos) as usize) {
                            None => panic!("ERROR: max number of items overflowed"),
                            Some(key) => key,
                        };

                        object_pos = object_index + 1;

                        let idx = hashmod(seed, &key, size);

                        if collide.contains(idx) {
                            continue;
                        }
                        let a_was_set = !a.insert_sync(idx);
                        if a_was_set {
                            collide.insert_sync(idx);
                        }
                    }
                } // end-window for

                offset += len;
            } // end-objects for

            let mut offset = 0u64;
            for object in objects {
                let mut object_iter = object.into_iter();

                // Note: we will use Iterator::nth() to advance the iterator if
                // we've skipped over some items.
                let mut object_pos = 0;
                let len = object_iter.len() as u64;

                for object_index in 0..len {
                    let index = offset + object_index;

                    if !done_keys.contains(index) {
                        // This will fast-forward the iterator over unneeded items.
                        let key = match object_iter.nth((object_index - object_pos) as usize) {
                            None => panic!("ERROR: max number of items overflowed"),
                            Some(key) => key,
                        };

                        object_pos = object_index + 1;

                        let idx = hashmod(seed, &&key, size);

                        if collide.contains(idx) {
                            a.remove(idx);
                        } else {
                            done_keys.insert(index);
                        }
                    }
                } // end-window for

                offset += len;
            } // end- objects for

            bitvecs.push(a);
            if done_keys.len() as u64 == n {
                break;
            }
            iter += 1;
        }

        Mphf {
            bitvecs: Self::compute_ranks(bitvecs),
            phantom: PhantomData,
        }
    }
}

impl<T: SeedableHash + Debug> Mphf<T> {
    /// Generate a minimal perfect hash function for the set of `objects`.
    /// `objects` must not contain any duplicate items.
    /// `gamma` controls the tradeoff between the construction-time and run-time speed,
    /// and the size of the datastructure representing the hash function. See the paper for details.
    /// `max_iters` - None to never stop trying to find a perfect hash (safe if no duplicates).
    pub fn new(gamma: f64, objects: &[T]) -> Mphf<T> {
        assert!(gamma > 1.01);
        let mut bitvecs = Vec::new();
        let mut iter = 0;

        let mut cx = Context::new(
            std::cmp::max(255, (gamma * objects.len() as f64) as u64),
            iter,
        );

        objects.iter().for_each(|v| cx.find_collisions_sync(v));
        let mut redo_keys = objects
            .iter()
            .filter_map(|v| cx.filter(v))
            .collect::<Vec<_>>();

        bitvecs.push(cx.a);
        iter += 1;

        while !redo_keys.is_empty() {
            let mut cx = Context::new(
                std::cmp::max(255, (gamma * redo_keys.len() as f64) as u64),
                iter,
            );

            redo_keys.iter().for_each(|&v| cx.find_collisions_sync(v));
            redo_keys = redo_keys.into_iter().filter_map(|v| cx.filter(v)).collect();

            bitvecs.push(cx.a);
            iter += 1;
            if iter > MAX_ITERS {
                error!("ran out of key space. items: {:?}", redo_keys);
                panic!("counldn't find unique hashes");
            }
        }

        Mphf {
            bitvecs: Self::compute_ranks(bitvecs),
            phantom: PhantomData,
        }
    }

    /// Compute the hash value of `item`. This method should only be used
    /// with items known to be in construction set. Use `try_hash` if you cannot
    /// guarantee that `item` was in the construction set. If `item` was not present
    /// in the construction set this function may panic.
    pub fn hash(&self, item: &T) -> u64 {
        for i in 0..self.bitvecs.len() {
            let (bv, _) = &self.bitvecs[i];
            let hash = hashmod(i as u64, item, bv.capacity());

            if bv.contains(hash) {
                return self.get_rank(hash, i);
            }
        }

        unreachable!("must find a hash value");
    }

    /// Compute the hash value of `item`. If `item` was not present
    /// in the set of objects used to construct the hash function, the return
    /// value will an arbitrary value Some(x), or None.
    pub fn try_hash<Q>(&self, item: &Q) -> Option<u64>
    where
        T: Borrow<Q>,
        Q: ?Sized + SeedableHash,
    {
        for i in 0..self.bitvecs.len() {
            let (bv, _) = &(self.bitvecs)[i];
            let hash = hashmod(i as u64, item, bv.capacity());

            if bv.contains(hash) {
                return Some(self.get_rank(hash, i));
            }
        }

        None
    }
}

#[cfg(feature = "parallel")]
impl<T: SeedableHash + Debug + Sync + Send> Mphf<T> {
    /// Same as `new`, but parallelizes work on the rayon default Rayon threadpool.
    /// Configure the number of threads on that threadpool to control CPU usage.
    #[cfg(feature = "parallel")]
    pub fn new_parallel(gamma: f64, objects: &[T], starting_seed: Option<u64>) -> Mphf<T> {
        assert!(gamma > 1.01);
        let mut bitvecs = Vec::new();
        let mut iter = 0;

        let cx = Context::new(
            std::cmp::max(255, (gamma * objects.len() as f64) as u64),
            starting_seed.unwrap_or(0) + iter,
        );

        objects.into_par_iter().for_each(|v| cx.find_collisions(v));
        let mut redo_keys = objects
            .into_par_iter()
            .filter_map(|v| cx.filter(v))
            .collect::<Vec<_>>();

        bitvecs.push(cx.a);
        iter += 1;

        while !redo_keys.is_empty() {
            let cx = Context::new(
                std::cmp::max(255, (gamma * redo_keys.len() as f64) as u64),
                starting_seed.unwrap_or(0) + iter,
            );

            (&redo_keys)
                .into_par_iter()
                .for_each(|&v| cx.find_collisions(v));
            redo_keys = (&redo_keys)
                .into_par_iter()
                .filter_map(|&v| cx.filter(v))
                .collect();

            bitvecs.push(cx.a);
            iter += 1;
            if iter > MAX_ITERS {
                println!("ran out of key space. items: {:?}", redo_keys);
                panic!("counldn't find unique hashes");
            }
        }

        Mphf {
            bitvecs: Self::compute_ranks(bitvecs),
            phantom: PhantomData,
        }
    }
}

struct Context {
    size: u64,
    seed: u64,
    a: BitVector,
    collide: BitVector,
}

impl Context {
    fn new(size: u64, seed: u64) -> Self {
        Self {
            size,
            seed,
            a: BitVector::new(size),
            collide: BitVector::new(size),
        }
    }

    #[cfg(feature = "parallel")]
    fn find_collisions<T: SeedableHash>(&self, v: &T) {
        let idx = hashmod(self.seed, v, self.size);
        if !self.collide.contains(idx) && !self.a.insert(idx) {
            self.collide.insert(idx);
        }
    }

    fn find_collisions_sync<T: SeedableHash>(&mut self, v: &T) {
        let idx = hashmod(self.seed, v, self.size);
        if !self.collide.contains(idx) && !self.a.insert_sync(idx) {
            self.collide.insert_sync(idx);
        }
    }

    #[cfg(feature = "parallel")]
    fn filter<'t, T: SeedableHash>(&self, v: &'t T) -> Option<&'t T> {
        let idx = hashmod(self.seed, v, self.size);
        if self.collide.contains(idx) {
            self.a.remove(idx);
            Some(v)
        } else {
            None
        }
    }

    #[cfg(not(feature = "parallel"))]
    fn filter<'t, T: SeedableHash>(&mut self, v: &'t T) -> Option<&'t T> {
        let idx = hashmod(self.seed, v, self.size);
        if self.collide.contains(idx) {
            self.a.remove(idx);
            Some(v)
        } else {
            None
        }
    }
}

#[cfg(feature = "parallel")]
struct Queue<'a, I: 'a, T>
where
    &'a I: IntoIterator,
    <&'a I as IntoIterator>::Item: IntoIterator<Item = T>,
{
    keys_object: &'a I,
    queue: <&'a I as IntoIterator>::IntoIter,

    num_keys: u64,
    last_key_index: u64,

    job_id: u8,

    phantom_t: PhantomData<T>,
}

#[cfg(feature = "parallel")]
impl<'a, I: 'a, N1, N2, T> Queue<'a, I, T>
where
    &'a I: IntoIterator<Item = N1>,
    N2: Iterator<Item = T> + ExactSizeIterator,
    N1: IntoIterator<Item = T, IntoIter = N2> + Clone,
{
    fn new(keys_object: &'a I, num_keys: u64) -> Queue<'a, I, T> {
        Queue {
            keys_object,
            queue: keys_object.into_iter(),

            num_keys,
            last_key_index: 0,

            job_id: 0,

            phantom_t: PhantomData,
        }
    }

    fn next(&mut self, done_keys_count: &AtomicU64) -> Option<(N2, u8, u64, u64)> {
        if self.last_key_index == self.num_keys {
            loop {
                let done_count = done_keys_count.load(Ordering::SeqCst);

                if self.num_keys == done_count {
                    self.queue = self.keys_object.into_iter();
                    done_keys_count.store(0, Ordering::SeqCst);
                    self.last_key_index = 0;
                    self.job_id += 1;

                    break;
                }
            }
        }

        if self.job_id > 1 {
            return None;
        }

        let node = self.queue.next().unwrap();
        let node_keys_start = self.last_key_index;

        let num_keys = node.clone().into_iter().len() as u64;

        self.last_key_index += num_keys;

        Some((node.into_iter(), self.job_id, node_keys_start, num_keys))
    }
}

#[cfg(feature = "parallel")]
impl<'a, T: 'a + SeedableHash + Debug + Send + Sync> Mphf<T>
where
    &'a T: SeedableHash,
{
    /// Same as to `from_chunked_iterator` but parallelizes work over `num_threads` threads.
    #[cfg(feature = "parallel")]
    pub fn from_chunked_iterator_parallel<I, N>(
        gamma: f64,
        objects: &'a I,
        max_iters: Option<u64>,
        n: u64,
        num_threads: usize,
    ) -> Mphf<T>
    where
        &'a I: IntoIterator<Item = N>,
        N: IntoIterator<Item = T> + Send + Clone,
        <N as IntoIterator>::IntoIter: ExactSizeIterator,
        <&'a I as IntoIterator>::IntoIter: Send,
        I: Sync,
    {
        // TODO CONSTANT, might have to change
        // Allowing atmost 381Mb for buffer
        const MAX_BUFFER_SIZE: u64 = 50000000;
        const ONE_PERCENT_KEYS: f32 = 0.01;
        let min_buffer_keys_threshold: u64 = (ONE_PERCENT_KEYS * n as f32) as u64;

        let mut iter: u64 = 0;
        let mut bitvecs = Vec::<BitVector>::new();

        assert!(gamma > 1.01);

        let global = Arc::new(GlobalContext {
            done_keys: BitVector::new(std::cmp::max(255, n)),
            buffered_keys: Mutex::new(Vec::new()),
            buffer_keys: AtomicBool::new(false),
        });
        loop {
            if max_iters.is_some() && iter > max_iters.unwrap() {
                error!("ran out of key space. items: {:?}", global.done_keys.len());
                panic!("couldn't find unique hashes");
            }

            let keys_remaining = if iter == 0 {
                n
            } else {
                n - global.done_keys.len()
            };
            if keys_remaining == 0 {
                break;
            }
            if keys_remaining < MAX_BUFFER_SIZE && keys_remaining < min_buffer_keys_threshold {
                global.buffer_keys.store(true, Ordering::SeqCst);
            }

            let size = std::cmp::max(255, (gamma * keys_remaining as f64) as u64);
            let cx = Arc::new(IterContext {
                done_keys_count: AtomicU64::new(0),
                work_queue: Mutex::new(Queue::new(objects, n)),
                collide: BitVector::new(size),
                a: BitVector::new(size),
            });

            crossbeam_utils::thread::scope(|scope| {
                for _ in 0..num_threads {
                    let global = global.clone();
                    let cx = cx.clone();

                    scope.spawn(move |_| {
                        loop {
                            let (mut node, job_id, offset, num_keys) =
                                match cx.work_queue.lock().unwrap().next(&cx.done_keys_count) {
                                    None => break,
                                    Some(val) => val,
                                };

                            let mut node_pos = 0;
                            for index in 0..num_keys {
                                let key_index = offset + index;
                                if global.done_keys.contains(key_index) {
                                    continue;
                                }

                                let key = node.nth((index - node_pos) as usize).unwrap();
                                node_pos = index + 1;

                                let idx = hashmod(iter, &key, size);
                                let collision = cx.collide.contains(idx);
                                if job_id == 0 {
                                    if !collision && !cx.a.insert(idx) {
                                        cx.collide.insert(idx);
                                    }
                                } else if collision {
                                    cx.a.remove(idx);
                                    if global.buffer_keys.load(Ordering::SeqCst) {
                                        global.buffered_keys.lock().unwrap().push(key);
                                    }
                                } else {
                                    global.done_keys.insert(key_index);
                                }
                            }

                            cx.done_keys_count.fetch_add(num_keys, Ordering::SeqCst);
                        } //end-loop
                    }); //end-scope
                } //end-threads-for
            })
            .unwrap(); //end-crossbeam

            match Arc::try_unwrap(cx) {
                Ok(cx) => bitvecs.push(cx.a),
                Err(_) => unreachable!(),
            }

            iter += 1;
            if global.buffer_keys.load(Ordering::SeqCst) {
                break;
            }
        } //end-loop

        let buffered_keys_vec = global.buffered_keys.lock().unwrap();
        if buffered_keys_vec.len() > 1 {
            let mut buffered_mphf = Mphf::new_parallel(1.7, &buffered_keys_vec, Some(iter));

            for i in 0..buffered_mphf.bitvecs.len() {
                let buff_vec =
                    std::mem::replace(&mut buffered_mphf.bitvecs[i].0, BitVector::new(0));
                bitvecs.push(buff_vec);
            }
        }

        Mphf {
            bitvecs: Self::compute_ranks(bitvecs),
            phantom: PhantomData,
        }
    }
}

#[cfg(feature = "parallel")]
struct IterContext<'a, I: 'a, N1, N2, T>
where
    &'a I: IntoIterator<Item = N1>,
    N2: Iterator<Item = T> + ExactSizeIterator,
    N1: IntoIterator<Item = T, IntoIter = N2> + Clone,
{
    done_keys_count: AtomicU64,
    work_queue: Mutex<Queue<'a, I, T>>,
    collide: BitVector,
    a: BitVector,
}

#[cfg(feature = "parallel")]
struct GlobalContext<T> {
    done_keys: BitVector,
    buffered_keys: Mutex<Vec<T>>,
    buffer_keys: AtomicBool,
}

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
mod tests {

    use super::*;
    use std::collections::HashSet;
    use std::iter::FromIterator;

    /// Check that a Minimal perfect hash function (MPHF) is generated for the set xs
    fn check_mphf<T>(xs: HashSet<T>) -> bool
    where
        T: Sync + SeedableHash + PartialEq + Eq + Debug + Send,
    {
        let xsv: Vec<T> = xs.into_iter().collect();

        // test single-shot data input
        check_mphf_serial(&xsv) && check_mphf_parallel(&xsv)
    }

    /// Check that a Minimal perfect hash function (MPHF) is generated for the set xs
    fn check_mphf_serial<T>(xsv: &[T]) -> bool
    where
        T: SeedableHash + PartialEq + Eq + Debug,
    {
        // Generate the MPHF
        let phf = Mphf::new(1.7, xsv);

        // Hash all the elements of xs
        let mut hashes: Vec<u64> = xsv.iter().map(|v| phf.hash(v)).collect();

        hashes.sort_unstable();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..xsv.len() as u64).collect();
        hashes == gt
    }

    /// Check that a Minimal perfect hash function (MPHF) is generated for the set xs
    #[cfg(feature = "parallel")]
    fn check_mphf_parallel<T>(xsv: &[T]) -> bool
    where
        T: Sync + SeedableHash + PartialEq + Eq + Debug + Send,
    {
        // Generate the MPHF
        let phf = Mphf::new_parallel(1.7, xsv, None);

        // Hash all the elements of xs
        let mut hashes: Vec<u64> = xsv.iter().map(|v| phf.hash(v)).collect();

        hashes.sort_unstable();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..xsv.len() as u64).collect();
        hashes == gt
    }

    #[cfg(not(feature = "parallel"))]
    fn check_mphf_parallel<T>(_xsv: &[T]) -> bool
    where
        T: SeedableHash + PartialEq + Eq + Debug,
    {
        true
    }

    fn check_chunked_mphf<T>(values: Vec<Vec<T>>, total: u64) -> bool
    where
        T: Sync + SeedableHash + PartialEq + Eq + Debug + Send,
    {
        let phf = Mphf::from_chunked_iterator(1.7, &values, total);

        // Hash all the elements of xs
        let mut hashes: Vec<u64> = values
            .iter()
            .flat_map(|x| x.iter().map(|v| phf.hash(&v)))
            .collect();

        hashes.sort_unstable();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..total as u64).collect();
        hashes == gt
    }

    #[cfg(feature = "parallel")]
    fn check_chunked_mphf_parallel<T>(values: Vec<Vec<T>>, total: u64) -> bool
    where
        T: Sync + SeedableHash + PartialEq + Eq + Debug + Send,
    {
        let phf = Mphf::from_chunked_iterator_parallel(1.7, &values, None, total, 2);

        // Hash all the elements of xs
        let mut hashes: Vec<u64> = values
            .iter()
            .flat_map(|x| x.iter().map(|v| phf.hash(&v)))
            .collect();

        hashes.sort_unstable();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..total as u64).collect();
        hashes == gt
    }

    #[cfg(not(feature = "parallel"))]
    fn check_chunked_mphf_parallel<T>(_values: Vec<Vec<T>>, _total: u64) -> bool
    where
        T: Sync + Hash + PartialEq + Eq + Debug + Send,
    {
        true
    }

    // this does not work under WASI.
    #[test]
    #[cfg(feature = "parallel")]
    fn check_crossbeam_scope() {
        crossbeam_utils::thread::scope(|scope| {
            let mut handles = vec![];
            for i in 0..2 {
                let h = scope.spawn(move |_| i * i);
                handles.push(h);
            }

            for (i, h) in handles.into_iter().enumerate() {
                assert_eq!(i * i, h.join().unwrap());
            }
        })
        .unwrap()
    }

    quickcheck! {
        fn check_int_slices(v: HashSet<u64>, lens: Vec<usize>) -> bool {

            let mut lens = lens;

            let items: Vec<u64> = v.iter().cloned().collect();
            if lens.is_empty() || lens.iter().all(|x| *x == 0) {
                lens.clear();
                lens.push(items.len())
            }

            let mut slices: Vec<Vec<u64>> = Vec::new();

            let mut total = 0_usize;
            for slc_len in lens {
                let end = std::cmp::min(items.len(), total.saturating_add(slc_len));
                let slc = Vec::from(&items[total..end]);
                slices.push(slc);
                total = end;

                if total == items.len() {
                    break;
                }
            }

            check_chunked_mphf(slices.clone(), total as u64) && check_chunked_mphf_parallel(slices, total as u64)
        }
    }

    quickcheck! {
        fn check_string(v: HashSet<Vec<String>>) -> bool {
            check_mphf(v)
        }
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

    #[test]
    fn from_ints_serial() {
        let items = (0..1000000).map(|x| x * 2);
        assert!(check_mphf(HashSet::from_iter(items)));
    }

    #[test]
    fn externally_hashed() {
        let total = 1000000;
        // User gets to pick the hash function.
        let entries = (0..total)
            .map(|x| ExternallyHashed(wyhash::wyrng(&mut (x * 2))))
            .collect::<Vec<ExternallyHashed>>();
        let phf = Mphf::new(1.7, &entries);

        let mut hashes = entries.iter().map(|eh| phf.hash(eh)).collect::<Vec<u64>>();
        hashes.sort_unstable();

        let gt = (0..total as u64).collect::<Vec<u64>>();
        assert_eq!(hashes, gt);

        // Hand-picked a value that fails to hash since it's not in the original set that it's built from.
        // It's not ideal that this assertion is sensitive to the implementation details internal to Mphf.
        assert_eq!(
            phf.try_hash(&ExternallyHashed(wyhash::wyrng(&mut 1000129))),
            None
        );
    }

    #[test]
    fn dma_serde() {
        // User gets to pick the hash function.
        let items = (0..1000000).map(|x| x * 2).collect::<Vec<_>>();
        let phf = Mphf::new(1.7, &items);

        let serialized = phf.to_dma();

        let phf2 = Mphf::<i32>::copy_from_scatter_dma(serialized.iter().copied());
        let phf3 = MphfRef::<i32>::from_scatter_dma(serialized.iter().copied());

        assert_eq!(phf, phf2);
        assert_eq!(phf2, phf);

        assert_eq!(phf, phf3);
        assert_eq!(phf3, phf);

        for i in items {
            assert_eq!(phf.try_hash(&i), phf2.try_hash(&i));
            assert_eq!(phf.try_hash(&i), phf3.try_hash(&i));
        }

        for i in 1000000..1000000 * 2 {
            assert_eq!(phf.try_hash(&i), phf2.try_hash(&i));
            assert_eq!(phf.try_hash(&i), phf3.try_hash(&i));
        }
    }
}
