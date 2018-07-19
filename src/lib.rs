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
//!		hashes.push(phf.hash(&v));
//!	}
//!	hashes.sort();
//!
//! // Expected hash output is set of all integers from 0..n
//! let expected_hashes: Vec<u64> = (0 .. n as u64).collect();
//!	assert!(hashes == expected_hashes)
//! ```

extern crate fnv;
extern crate heapsize;
extern crate rayon;
extern crate crossbeam;

#[macro_use]
extern crate log;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

use heapsize::HeapSizeOf;
use rayon::prelude::*;

pub mod hashmap;
mod bitvector;
use bitvector::*;

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::{Arc, Mutex};
use std::marker::PhantomData;

use std::sync::atomic::{AtomicUsize, Ordering, AtomicBool};


#[inline]
fn hash_with_seed<T: Hash>(iter: u64, v: &T) -> u64 {
    let mut state = fnv::FnvHasher::with_key(iter);
    v.hash(&mut state);
    state.finish()
}

/// Trait for iterators that can skip values efficiently if the client knows they aren't needed.
pub trait FastIterator: Iterator {
    /// Skip the next item in the iterator, without returning it.
    fn skip_next(&mut self);
}

/// A minimal perfect hash function over a set of objects of type `T`.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mphf<T> {
    bitvecs: Vec<BitVector>,
    ranks: Vec<Vec<u64>>,
    phantom: PhantomData<T>,
}

impl<T> HeapSizeOf for Mphf<T> {
    fn heap_size_of_children(&self) -> usize {
        self.bitvecs.heap_size_of_children() + self.ranks.heap_size_of_children()
    }
}

const MAX_ITERS: u64 = 100;

#[cfg(feature = "fast-constructors")]
impl<'a, T: 'a + Hash + Clone + Debug> Mphf<T> {
    pub fn new_with_key<I, N>(gamma: f64, objects: &'a I, n: usize) -> Mphf<T>
    where
        &'a I: IntoIterator<Item = N>, N: IntoIterator<Item = T>,
        <N as IntoIterator>::IntoIter: FastIterator,
        <&'a I as IntoIterator>::IntoIter: Send, N: Send + ExactSizeIterator, I: Sync
    {
        let mut iter = 0;
        let mut bitvecs = Vec::new();
        let done_keys = BitVector::new(std::cmp::max(255, n));

        assert!(gamma > 1.01);

        loop {
            if iter > MAX_ITERS {
                error!("ran out of key space. items: {:?}", done_keys.len());
                panic!("counldn't find unique hashes");
            }

            let keys_remaining = if iter == 0 { n } else { n - done_keys.len() };

            let size = std::cmp::max(255, (gamma * keys_remaining as f64) as u64);

            let a = BitVector::new(size as usize);
            let collide = BitVector::new(size as usize);

            let seed = iter;
            let mut offset = 0;

            for object in objects {
                let len = object.len();
                let mut object_iter = object.into_iter();

                for object_index in 0..len {
                    let index = offset + object_index;

                    if !done_keys.contains(index) {
                        let key = match object_iter.next() {
                            None => panic!("ERROR: max number of items overflowed"),
                            Some(key) => key,
                        };

                        let idx = hash_with_seed(seed, &key) % size;

                        if collide.contains(idx as usize) {
                            continue;
                        }
                        let a_was_set = !a.insert(idx as usize);
                        if a_was_set {
                            collide.insert(idx as usize);
                        }
                    }
                    else{
                        object_iter.skip_next();
                    }
                }// end-window for

                offset += len;
            }// end-objects for

            let mut offset = 0;
            for object in objects {
                let len = object.len();
                let mut object_iter = object.into_iter();

                for object_index in 0..len {
                    let index = offset + object_index;

                    if !done_keys.contains(index) {
                        let key = match object_iter.next() {
                            None => panic!("ERROR: max number of items overflowed"),
                            Some(key) => key,
                        };

                        let idx = hash_with_seed(seed, &key) % size;

                        if collide.contains(idx as usize) {
                            a.remove(idx as usize);
                        } else {
                            done_keys.insert(index as usize);
                        }
                    }
                    else{
                        object_iter.skip_next();
                    }
                } // end-window for

                offset += len;
            } // end- objects for

            bitvecs.push(a);
            if done_keys.len() == n {
                break;
            }
            iter += 1;
        }

        let ranks = Self::compute_ranks(&bitvecs);
        let r = Mphf {
            bitvecs: bitvecs,
            ranks: ranks,
            phantom: PhantomData,
        };
        let sz = r.heap_size_of_children();
        info!(
            "\nItems: {}, Mphf Size: {}, Bits/Item: {}",
            n,
            sz,
            (sz * 8) as f32 / n as f32
        );
        r
    }
}

impl<T: Hash + Clone + Debug> Mphf<T> {
    /// Generate a minimal perfect hash function for the set of `objects`.
    /// `objects` must not contain any duplicate items.
    /// `gamma` controls the tradeoff between the construction-time and run-time speed,
    /// and the size of the datastructure representing the hash function. See the paper for details.
    /// `max_iters` - None to never stop trying to find a perfect hash (safe if no duplicates).
    pub fn new(gamma: f64, objects: &Vec<T>) -> Mphf<T> {
        let n = objects.len();
        let mut bitvecs = Vec::new();
        let mut iter = 0;
        let mut redo_keys = Vec::new();

        assert!(gamma > 1.01);

        loop {
            // this scope structure is needed to allow keys to be
            // the 'objects' reference on the first loop, and a reference
            // to 'redo_keys' on subsequent iterations.
            redo_keys = {
                let keys = if iter == 0 { objects } else { &redo_keys };

                if iter > MAX_ITERS {
                    error!("ran out of key space. items: {:?}", keys);
                    panic!("counldn't find unique hashes");
                }

                let size = std::cmp::max(255, (gamma * keys.len() as f64) as u64);
                let a = BitVector::new(size as usize);
                let collide = BitVector::new(size as usize);

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

                let mut redo_keys_tmp = Vec::new();
                for v in keys.iter() {
                    let idx = hash_with_seed(seed, v) % size;

                    if collide.contains(idx as usize) {
                        redo_keys_tmp.push(v.clone());
                        a.remove(idx as usize);
                    }
                }

                bitvecs.push(a);
                redo_keys_tmp
            };

            if redo_keys.len() == 0 {
                break;
            }
            iter += 1;
        }

        let ranks = Self::compute_ranks(&bitvecs);
        let r = Mphf {
            bitvecs: bitvecs,
            ranks: ranks,
            phantom: PhantomData,
        };
        let sz = r.heap_size_of_children();
        info!(
            "\nItems: {}, Mphf Size: {}, Bits/Item: {}",
            n,
            sz,
            (sz * 8) as f32 / n as f32
        );
        r
    }

    fn compute_ranks(bvs: &Vec<BitVector>) -> Vec<Vec<u64>> {
        let mut ranks = Vec::new();
        let mut pop = 0 as u64;

        for bv in bvs {
            let mut rank: Vec<u64> = Vec::new();
            for i in 0..bv.num_words() {
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
                return Some(self.get_rank(hash, iter));
            }
        }

        None
    }
}

impl<T: Hash + Clone + Debug + Sync + Send> Mphf<T> {
    /// Same as `new`, but parallelizes work on the rayon default Rayon threadpool.
    /// Configure the number of threads on that threadpool to control CPU usage.
    pub fn new_parallel(gamma: f64, objects: &Vec<T>,
                        starting_seed: Option<u64>) -> Mphf<T> {
        let n = objects.len();
        let mut bitvecs = Vec::new();
        let mut iter = 0;
        let mut redo_keys = Vec::new();

        assert!(gamma > 1.01);

        loop {
            // this scope structure is needed to allow keys to be
            // the 'objects' reference on the first loop, and a reference
            // to 'redo_keys' on subsequent iterations.
            redo_keys = {
                let keys = if iter == 0 { objects } else { &redo_keys };

                if iter > MAX_ITERS {
                    println!("ran out of key space. items: {:?}", keys);
                    panic!("counldn't find unique hashes");
                }

                let size = std::cmp::max(255, (gamma * keys.len() as f64) as u64);
                let a = BitVector::new(size as usize);
                let collide = BitVector::new(size as usize);

                let seed = match starting_seed {
                    None => iter,
                    Some(seed) => iter + seed,
                };

                (&keys).par_chunks(1 << 16).for_each(|chnk| {
                    for v in chnk.iter() {
                        let idx = hash_with_seed(seed, v) % size;
                        if collide.contains(idx as usize) {
                            continue;
                        }

                        let a_was_set = !a.insert(idx as usize);
                        if a_was_set {
                            collide.insert(idx as usize);
                        }
                    }
                });

                let redo_keys_tmp: Vec<T> = (&keys)
                    .par_chunks(1 << 16)
                    .flat_map(|chnk| {
                        let mut redo_keys_chunk = Vec::new();

                        for v in chnk.iter() {
                            let idx = hash_with_seed(seed, v) % size;

                            if collide.contains(idx as usize) {
                                redo_keys_chunk.push(v.clone());
                                a.remove(idx as usize);
                            }
                        }

                        redo_keys_chunk
                    })
                    .collect();

                bitvecs.push(a);
                redo_keys_tmp
            };

            if redo_keys.len() == 0 {
                break;
            }
            iter += 1;
        }

        let ranks = Self::compute_ranks(&bitvecs);
        let r = Mphf {
            bitvecs: bitvecs,
            ranks: ranks,
            phantom: PhantomData,
        };
        let sz = r.heap_size_of_children();
        info!(
            "Items: {}, Mphf Size: {}, Bits/Item: {}",
            n,
            sz,
            (sz * 8) as f32 / n as f32
        );
        r
    }
}

#[cfg(feature = "fast-constructors")]
struct Queue<'a, I: 'a, N, T>
where
    &'a I: IntoIterator<Item = N>, N: IntoIterator<Item = T>
{
    keys_object: &'a I,
    queue: <&'a I as IntoIterator>::IntoIter,

    num_keys: usize,
    last_key_index: usize,

    job_id: u8,

    phantom_t: PhantomData<T>,
    phantom_n: PhantomData<N>,
}

#[cfg(feature = "fast-constructors")]
impl<'a, I: 'a, N, T> Queue<'a, I, N, T>
where
    &'a I: IntoIterator<Item = N>, N: IntoIterator<Item = T>,
    <N as IntoIterator>::IntoIter: FastIterator, N: ExactSizeIterator {

    fn new(object: &'a I, num_keys: usize) -> Queue<'a, I, N, T>
    {
        Queue{
            keys_object: object,
            queue: object.into_iter(),

            num_keys: num_keys,
            last_key_index: 0,

            job_id: 0,

            phantom_t: PhantomData,
            phantom_n: PhantomData,
        }
    }

    fn next(&mut self, done_keys_count: &Arc<AtomicUsize>) -> Option<(N, u8, usize, usize)> {
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
        let num_keys = node.len();

        self.last_key_index += num_keys;

        return Some( (node, self.job_id,
                      node_keys_start,
                      num_keys)
        );
    }
}

#[cfg(feature = "fast-constructors")]
impl<'a, T: 'a + Hash + Clone + Debug + Send + Sync> Mphf<T> {

    pub fn new_parallel_with_keys<I, N>(gamma: f64, objects: &'a I,
                                        max_iters: Option<u64>,
                                        n: usize, num_threads: usize)
                                        -> Mphf<T>
    where
        &'a I: IntoIterator<Item = N>, N: IntoIterator<Item = T>,
        <N as IntoIterator>::IntoIter: FastIterator,
        <&'a I as IntoIterator>::IntoIter: Send, N: Send + ExactSizeIterator, I: Sync
    {

        // TODO CONSTANT, might have to change
        // Allowing atmost 381Mb for buffer
        const MAX_BUFFER_SIZE: usize = 50000000;
        const ONE_PERCENT_KEYS: f32 = 0.01;
        let min_buffer_keys_threshold: usize = (ONE_PERCENT_KEYS * n as f32) as usize;

        let mut iter: u64 = 0;
        let mut bitvecs = Vec::<BitVector>::new();
        let done_keys = Arc::new(BitVector::new(std::cmp::max(255, n)));

        assert!(gamma > 1.01);

        let find_collisions = | seed: &u64, key: &T, size: &u64,
                                collide: &Arc<BitVector>,
                                a: &Arc<BitVector> | {
            let idx = hash_with_seed(*seed, key) % size;

            if ! collide.contains(idx as usize) {
                let a_was_set = !a.insert(idx as usize);
                if a_was_set {
                    collide.insert(idx as usize);
                }
            }
        };

        let remove_collisions = | seed: &u64, key: &T, size: &u64,
                                  keys_index: usize,
                                  collide: &Arc<BitVector>,
                                  a: &Arc<BitVector>,
                                  done_keys: &BitVector,
                                  buffer_keys: &Arc<AtomicBool>,
                                  buffered_keys_vec: &Arc<Mutex<Vec<T>>> | {
            let idx = hash_with_seed(*seed, key) % size;

            if collide.contains(idx as usize) {
                a.remove(idx as usize);
                if buffer_keys.load(Ordering::SeqCst) {
                    buffered_keys_vec.lock().unwrap().push(key.clone());
                }
            }
            else {
                done_keys.insert(keys_index as usize);
            }
        };

        let buffered_keys_vec = Arc::new(Mutex::new(Vec::new()));
        let buffer_keys = Arc::new(AtomicBool::new(false));
        loop {
            if max_iters.is_some() && iter > max_iters.unwrap() {
                error!("ran out of key space. items: {:?}", done_keys.len());
                panic!("counldn't find unique hashes");
            }

            let keys_remaining = if iter == 0 { n } else { n - done_keys.len() };
            if keys_remaining == 0 { break; }
            if keys_remaining < MAX_BUFFER_SIZE
                && keys_remaining < min_buffer_keys_threshold {
                    buffer_keys.store(true, Ordering::SeqCst);
            }

            let size = std::cmp::max(255, (gamma * keys_remaining as f64) as u64);
            let a = Arc::new(BitVector::new(size as usize));
            let collide = Arc::new(BitVector::new(size as usize));

            let work_queue = Arc::new(Mutex::new(Queue::new(objects, n)));
            let done_keys_count = Arc::new(AtomicUsize::new(0));

            crossbeam::scope(|scope| {

                for _ in 0 .. num_threads {
                    let buffer_keys = buffer_keys.clone();
                    let buffered_keys_vec = buffered_keys_vec.clone();
                    let done_keys_count = done_keys_count.clone();
                    let work_queue = work_queue.clone();
                    let done_keys = done_keys.clone();
                    let collide = collide.clone();
                    let a = a.clone();

                    scope.spawn(move || {
                        loop {

                            let (node, job_id, offset, num_keys) =
                                match work_queue.lock().unwrap().next(&done_keys_count) {
                                    None => break,
                                    Some(val) => val,
                                };

                            let mut into_node = node.into_iter();
                            for index in 0..num_keys {
                                let key_index = offset + index;
                                if !done_keys.contains(key_index) {
                                    let key = into_node.next().unwrap();

                                    if job_id == 0 {
                                        find_collisions(&iter, &key, &size,
                                                        &collide, &a);
                                    }
                                    else {
                                        remove_collisions(&iter, &key, &size,
                                                          key_index,
                                                          &collide, &a,
                                                          &done_keys, &buffer_keys,
                                                          &buffered_keys_vec);
                                    }
                                } //end-if
                                else{
                                    into_node.skip_next();
                                }
                            }

                            done_keys_count.fetch_add(num_keys, Ordering::SeqCst);
                        } //end-loop
                    }); //end-scope
                } //end-threads-for
            }); //end-crossbeam

            let unwrapped_a = Arc::try_unwrap(a).unwrap();
            bitvecs.push(unwrapped_a);

            iter += 1;
            if buffer_keys.load(Ordering::SeqCst) {
                break;
            }
        } //end-loop

        let buffered_keys_vec = buffered_keys_vec.lock().unwrap();
        if buffered_keys_vec.len() > 1 {
            let buffered_mphf = Mphf::new_parallel(1.7, &buffered_keys_vec, Some(iter));
            for bitvec in buffered_mphf.bitvecs {
                bitvecs.push(bitvec);
            }
        }

        let ranks = Self::compute_ranks(&bitvecs);
        let r = Mphf {
            bitvecs: bitvecs,
            ranks: ranks,
            phantom: PhantomData,
        };
        let sz = r.heap_size_of_children();
        info!(
            "\nItems: {}, Mphf Size: {}, Bits/Item: {}",
            n,
            sz,
            (sz * 8) as f32 / n as f32
        );
        r
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

    /// Check that a Minimal perfect hash function (MPHF) is generated for the set xs
    fn check_mphf<T>(xs: HashSet<T>) -> bool
    where
        T: Sync + Hash + PartialEq + Eq + Clone + Debug + Send,
    {
        let mut xsv: Vec<T> = Vec::new();
        xsv.extend(xs.into_iter());

        // test single-shot data input
        check_mphf_serial(&xsv) && check_mphf_parallel(&xsv)
    }

    /// Check that a Minimal perfect hash function (MPHF) is generated for the set xs
    fn check_mphf_serial<T>(xsv: &Vec<T>) -> bool
    where
        T: Hash + PartialEq + Eq + Clone + Debug,
    {

        // Generate the MPHF
        let phf = Mphf::new(1.7, &xsv);

        // Hash all the elements of xs
        let mut hashes = Vec::new();

        for v in xsv {
            hashes.push(phf.hash(&v));
        }

        hashes.sort();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..xsv.len() as u64).collect();
        hashes == gt
    }

    /// Check that a Minimal perfect hash function (MPHF) is generated for the set xs
    fn check_mphf_parallel<T>(xsv: &Vec<T>) -> bool
    where
        T: Sync + Hash + PartialEq + Eq + Clone + Debug + Send,
    {
        // Generate the MPHF
        let phf = Mphf::new_parallel(1.7, &xsv, None);

        // Hash all the elements of xs
        let mut hashes = Vec::new();

        for v in xsv {
            hashes.push(phf.hash(&v));
        }

        hashes.sort();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..xsv.len() as u64).collect();
        hashes == gt
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

    use heapsize::HeapSizeOf;
    #[test]
    fn test_heap_size_vec() {
        let mut vs = Vec::new();
        for _ in 0..100 {
            let vn = vec![123usize; 100];
            vs.push(vn);
        }
        println!("heap_size: {}", vs.heap_size_of_children());
        assert!(vs.heap_size_of_children() > 80000);
    }

    #[test]
    fn test_heap_size_bv() {
        let bv = BitVector::new(100000);
        println!("heap_size: {}", bv.heap_size_of_children());
        assert!(bv.heap_size_of_children() > 100000 / 64);
    }
}
