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

#[cfg(feature = "heapsize")]
extern crate heapsize;
extern crate rayon;
extern crate crossbeam_utils;
extern crate wyhash;

#[macro_use]
extern crate log;

#[cfg(feature = "serde")]
#[macro_use]
extern crate serde;

#[cfg(feature = "heapsize")]
use heapsize::HeapSizeOf;
use rayon::prelude::*;

pub mod hashmap;
mod bitvector;
use bitvector::*;

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering, AtomicBool};

#[inline]
fn fold(v: u64) -> u32 {
    ((v & 0xFFFFFFFF) as u32) ^ ((v >> 32) as u32)
}

#[inline]
#[allow(dead_code)]
fn hash_with_seed_slow<T: Hash>(iter: u64, v: &T) -> u64 {
    let mut state = fnv::FnvHasher::with_key(iter);
    v.hash(&mut state);
    state.finish()
}

#[inline]
fn hash_with_seed<T: Hash>(iter: u64, v: &T) -> u64 {
  let mut state = wyhash::WyHash::with_seed(1<<iter + iter);
  v.hash(&mut state);
  state.finish()
}

#[inline]
fn hash_with_seed32<T: Hash>(iter: u64, v: &T) -> u32 {
    fold(hash_with_seed(iter, v))
}


#[inline]
fn fastmod(hash: u32, n: u32) -> u64 {
    ((hash as u64) * (n as u64) >> 32)
}

#[inline]
fn hashmod<T: Hash>(iter: u64, v: &T, n: usize) -> u64 {
    // when n < 2^32, use the fast alternative to modulo described here: 
    // https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    if n < 1<<32 { 
        let h = hash_with_seed32(iter, v);
        fastmod(h, n as u32) as u64
    } else {
        let h = hash_with_seed(iter, v);
        h % (n as u64)
    }
}

/// A minimal perfect hash function over a set of objects of type `T`.
#[derive(Clone,Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mphf<T> {
    bitvecs: Box<[BitVector]>,
    ranks: Box<[Box<[u64]>]>,
    phantom: PhantomData<T>,
}

#[cfg(feature = "heapsize")]
impl<T> HeapSizeOf for Mphf<T> {
    fn heap_size_of_children(&self) -> usize {
        self.bitvecs.heap_size_of_children() + self.ranks.heap_size_of_children()
    }
}

const MAX_ITERS: u64 = 100;

impl<'a, T: 'a + Hash + Clone + Debug> Mphf<T> {


    /// Constructs an MPHF from a (possibly lazy) iterator over iterators.
    /// This allows construction of very large MPHFs without holding all the keys
    /// in memory simultaneously.
    /// `objects` is an `IntoInterator` yielding a stream of `IntoIterator`s that must not contain any duplicate items.
    /// `objects` must be able to be iterated over multiple times and yield the same stream of items each time.
    /// `gamma` controls the tradeoff between the construction-time and run-time speed,
    /// and the size of the datastructure representing the hash function. See the paper for details.
    /// `max_iters` - None to never stop trying to find a perfect hash (safe if no duplicates).
    /// NOTE: the inner iterator `N::IntoIter` should override `nth` if there's an efficient way to skip 
    /// over items when iterating.  This is important because later iterations of the MPHF construction algorithm
    /// skip most of the items.
    pub fn from_chunked_iterator<I, N>(gamma: f64, objects: &'a I, n: usize) -> Mphf<T>
    where
        &'a I: IntoIterator<Item = N>, N: IntoIterator<Item = T> + Send,
        <N as IntoIterator>::IntoIter: ExactSizeIterator,
        <&'a I as IntoIterator>::IntoIter: Send, I: Sync
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

            let mut a = BitVector::new(size as usize);
            let mut collide = BitVector::new(size as usize);

            let seed = iter;
            let mut offset = 0;

            for object in objects {
                let mut object_iter = object.into_iter();

                // Note: we will use Iterator::nth() to advance the iterator if
                // we've skipped over some items.
                let mut object_pos = 0;
                let len = object_iter.len();

                for object_index in 0..len {
                    let index = offset + object_index;

                    if !done_keys.contains(index) {
                        let key = match object_iter.nth(object_index - object_pos) {
                            None => panic!("ERROR: max number of items overflowed"),
                            Some(key) => key,
                        };

                        object_pos = object_index + 1;

                        let idx = hashmod(seed, &key, size as usize);

                        if collide.contains(idx as usize) {
                            continue;
                        }
                        let a_was_set = !a.insert_sync(idx as usize);
                        if a_was_set {
                            collide.insert_sync(idx as usize);
                        }
                    }
                }// end-window for

                offset += len;
            }// end-objects for

            let mut offset = 0;
            for object in objects {
                let mut object_iter = object.into_iter();

                // Note: we will use Iterator::nth() to advance the iterator if
                // we've skipped over some items.
                let mut object_pos = 0;
                let len = object_iter.len();

                for object_index in 0..len {
                    let index = offset + object_index;

                    if !done_keys.contains(index) {
                        // This will fast-forward the iterator over unneeded items.
                        let key = match object_iter.nth(object_index - object_pos) {
                            None => panic!("ERROR: max number of items overflowed"),
                            Some(key) => key,
                        };

                        object_pos = object_index + 1;

                        let idx = hashmod(seed, &key, size as usize);

                        if collide.contains(idx as usize) {
                            a.remove(idx as usize);
                        } else {
                            done_keys.insert(index as usize);
                        }
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
            bitvecs: bitvecs.into_boxed_slice(),
            ranks: ranks,
            phantom: PhantomData,
        };

        r.log_heap_size(n);
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
                let mut a = BitVector::new(size as usize);
                let mut collide = BitVector::new(size as usize);

                let seed = iter;

                for v in keys.iter() {
                    let idx = hashmod(seed, v, size as usize);

                    if collide.contains(idx as usize) {
                        continue;
                    }

                    let a_was_set = !a.insert_sync(idx as usize);
                    if a_was_set {
                        collide.insert_sync(idx as usize);
                    }
                }

                let mut redo_keys_tmp = Vec::new();
                for v in keys.iter() {
                    let idx = hashmod(seed, v, size as usize);

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
            bitvecs: bitvecs.into_boxed_slice(),
            ranks: ranks,
            phantom: PhantomData,
        };

        r.log_heap_size(n);
        r
    }

    
    fn log_heap_size(&self, _items: usize) {
        #[cfg(feature = "heapsize")]
        {
            let sz = self.heap_size_of_children();
            info!(
                "\nItems: {}, Mphf Size: {}, Bits/Item: {}",
                _items,
                sz,
                (sz * 8) as f32 / _items as f32
            );
        }
    }

    fn compute_ranks(bvs: &Vec<BitVector>) -> Box<[Box<[u64]>]> {
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

            ranks.push(rank.into_boxed_slice())
        }

        ranks.into_boxed_slice()
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
        for i in 0..self.bitvecs.len() {
            let bv = &self.bitvecs[i];
            let hash = hashmod(i as u64, item, bv.capacity());

            if bv.contains(hash as usize) {
                return self.get_rank(hash, i);
            }
        }

        unreachable!("must find a hash value");
    }

    /// Compute the hash value of `item`. If `item` was not present
    /// in the set of objects used to construct the hash function, the return
    /// value will an arbitrary value Some(x), or None.
    pub fn try_hash(&self, item: &T) -> Option<u64> {
        for i in 0..self.bitvecs.len() {
            let bv = &(self.bitvecs)[i];
            let hash = hashmod(i as u64, item, bv.capacity());

            if bv.contains(hash as usize) {
                return Some(self.get_rank(hash, i));
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
                        let idx = hashmod(seed, v, size as usize);
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
                            let idx = hashmod(seed, v, size as usize);

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
            bitvecs: bitvecs.into_boxed_slice(),
            ranks: ranks,
            phantom: PhantomData,
        };

        r.log_heap_size(n);
        r
    }
}

struct Queue<'a, I: 'a, T>
where
    &'a I: IntoIterator,
    <&'a I as IntoIterator>::Item: IntoIterator<Item = T>
{
    keys_object: &'a I,
    queue: <&'a I as IntoIterator>::IntoIter,

    num_keys: usize,
    last_key_index: usize,

    job_id: u8,

    phantom_t: PhantomData<T>,
}

impl<'a, I: 'a, N1, N2, T> Queue<'a, I, T>
where
    &'a I: IntoIterator<Item=N1>,
    N2: Iterator<Item = T> + ExactSizeIterator,
    N1: IntoIterator<Item = T, IntoIter = N2> + Clone {

    fn new(object: &'a I, num_keys: usize) -> Queue<'a, I, T>
    {
        Queue{
            keys_object: object,
            queue: object.into_iter(),

            num_keys: num_keys,
            last_key_index: 0,

            job_id: 0,

            phantom_t: PhantomData,
        }
    }

    fn next(&mut self, done_keys_count: &Arc<AtomicUsize>) -> Option<(N2, u8, usize, usize)> {
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
        
        let num_keys = node.clone().into_iter().len();

        self.last_key_index += num_keys;

        return Some((node.into_iter(), self.job_id,
                      node_keys_start,
                      num_keys)
        );
    }
}

impl<'a, T: 'a + Hash + Clone + Debug + Send + Sync> Mphf<T> {


    /// Same as to `from_chunked_iterator` but parallelizes work over `num_threads` threads.
    pub fn from_chunked_iterator_parallel<I, N>(gamma: f64, objects: &'a I,
                                        max_iters: Option<u64>,
                                        n: usize, num_threads: usize)
                                        -> Mphf<T>
    where
        &'a I: IntoIterator<Item = N>, 
        N: IntoIterator<Item = T> + Send + Clone,
        <N as IntoIterator>::IntoIter: ExactSizeIterator,
        <&'a I as IntoIterator>::IntoIter: Send, I: Sync
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
            let idx = hashmod(*seed, key, *size as usize);

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
            let idx = hashmod(*seed, key, *size as usize);

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

            crossbeam_utils::thread::scope(|scope| {

                for _ in 0 .. num_threads {
                    let buffer_keys = buffer_keys.clone();
                    let buffered_keys_vec = buffered_keys_vec.clone();
                    let done_keys_count = done_keys_count.clone();
                    let work_queue = work_queue.clone();
                    let done_keys = done_keys.clone();
                    let collide = collide.clone();
                    let a = a.clone();

                    scope.spawn(move |_| {
                        loop {

                            let (node, job_id, offset, num_keys) =
                                match work_queue.lock().unwrap().next(&done_keys_count) {
                                    None => break,
                                    Some(val) => val,
                                };

                            let mut into_node = node.into_iter();
                            let mut node_pos = 0;

                            for index in 0..num_keys {
                                let key_index = offset + index;
                                if !done_keys.contains(key_index) {

                                    let key = into_node.nth(index - node_pos).unwrap();
                                    node_pos = index + 1;

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
                            }

                            done_keys_count.fetch_add(num_keys, Ordering::SeqCst);
                        } //end-loop
                    }); //end-scope
                } //end-threads-for
            }).unwrap(); //end-crossbeam

            let unwrapped_a = Arc::try_unwrap(a).unwrap();
            bitvecs.push(unwrapped_a);

            iter += 1;
            if buffer_keys.load(Ordering::SeqCst) {
                break;
            }
        } //end-loop

        let buffered_keys_vec = buffered_keys_vec.lock().unwrap();
        if buffered_keys_vec.len() > 1 {
            let mut buffered_mphf = Mphf::new_parallel(1.7, &buffered_keys_vec, Some(iter));

            for i in 0..buffered_mphf.bitvecs.len() {
                let buff_vec = std::mem::replace(&mut buffered_mphf.bitvecs[i], BitVector::new(0));
                bitvecs.push(buff_vec);
            }   
        }

        let ranks = Self::compute_ranks(&bitvecs);
        let r = Mphf {
            bitvecs: bitvecs.into_boxed_slice(),
            ranks: ranks,
            phantom: PhantomData,
        };

        r.log_heap_size(n);
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

    fn check_chunked_mphf<T>(values: Vec<Vec<T>>, total: usize) -> bool 
    where T: Sync + Hash + PartialEq + Eq + Clone + Debug + Send 
    {
        let phf = Mphf::from_chunked_iterator(1.7, &values, total);

        // Hash all the elements of xs
        let mut hashes = Vec::new();

        for v in values.iter().flat_map(|x| x) {
            hashes.push(phf.hash(&v));
        }

        hashes.sort();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..total as u64).collect();
        hashes == gt
    }


    fn check_chunked_mphf_parallel<T>(values: Vec<Vec<T>>, total: usize) -> bool 
    where T: Sync + Hash + PartialEq + Eq + Clone + Debug + Send 
    {
        let phf = Mphf::from_chunked_iterator_parallel(1.7, &values, None, total, 2);

        // Hash all the elements of xs
        let mut hashes = Vec::new();

        for v in values.iter().flat_map(|x| x) {
            hashes.push(phf.hash(&v));
        }

        hashes.sort();

        // Hashes must equal 0 .. n
        let gt: Vec<u64> = (0..total as u64).collect();
        hashes == gt
    }

    quickcheck! {
        fn check_int_slices(v: HashSet<u64>, lens: Vec<usize>) -> bool {

            let mut lens = lens;

            let items: Vec<u64> = v.iter().cloned().collect();
            if lens.len() == 0 || lens.iter().all(|x| *x == 0) {
                lens.clear();
                lens.push(items.len())
            }

            let mut slices: Vec<Vec<u64>> = Vec::new();

            let mut total = 0;
            for slc_len in lens {
                let end = std::cmp::min(items.len(), total + slc_len);
                let slc = Vec::from(&items[total..end]);
                slices.push(slc);
                total = end;

                if total == items.len() {
                    break;
                }
            }
            
            check_chunked_mphf(slices.clone(), total) && check_chunked_mphf_parallel(slices, total)
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

    #[cfg(feature = "heapsize")]
    mod heap_size {
        use heapsize::HeapSizeOf;
        use super::*;

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
}
