// Copyright (c) 2018 10x Genomics, Inc. All rights reserved.
//
// Note this code was copied from https://github.com/zhaihj/bitvector (MIT licensed),
// and modified to add rank/select operations, and to use atomic primitives to allow
// multi-threaded access. The original copyright license text is here:
//
// The MIT License (MIT)
//
// Copyright (c) 2016 Hongjie Zhai

//! ### BitVector Module
//!
//! BitVector uses one bit to represent a bool state.
//! BitVector is useful for the programs that need fast set operation (intersection, union,
//! difference), because that all these operations can be done with simple bitand, bitor, bitxor.
//!
//! ### Implementation Details
//!
//! BitVector is realized with a `Vec<u64>`. Each bit of an u64 represent if a elements exists.
//! BitVector always increases from the end to begin, it meats that if you add element `0` to an
//! empty bitvector, then the `Vec<u64>` will change from `0x00` to `0x01`.
//!
//! Of course, if the real length of set can not be divided by 64,
//! it will have a `capacity() % 64` bit memory waste.
//!

use std::fmt;
#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "serde")]
use serde::{self, Deserialize, Serialize};

#[cfg(feature = "parallel")]
type Word = AtomicU64;

#[cfg(not(feature = "parallel"))]
type Word = u64;

/// Bitvector
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BitVector {
    bits: u64,
    #[cfg(feature = "parallel")]
    #[cfg_attr(
        feature = "serde",
        serde(serialize_with = "ser_atomic_vec", deserialize_with = "de_atomic_vec")
    )]
    vector: Box<[AtomicU64]>,

    #[cfg(not(feature = "parallel"))]
    vector: Box<[u64]>,
}

// Custom serializer
#[cfg(all(feature = "serde", feature = "parallel"))]
fn ser_atomic_vec<S>(v: &[AtomicU64], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(v.len()))?;
    for x in v {
        seq.serialize_element(&x.load(Ordering::SeqCst))?;
    }
    seq.end()
}

// Custom deserializer
#[cfg(all(feature = "serde", feature = "parallel"))]
pub fn de_atomic_vec<'de, D>(deserializer: D) -> Result<Box<[AtomicU64]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct AtomicU64SeqVisitor;

    impl<'de> serde::de::Visitor<'de> for AtomicU64SeqVisitor {
        type Value = Box<[AtomicU64]>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a 64bit unsigned integer")
        }

        fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
        where
            S: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::<AtomicU64>::with_capacity(access.size_hint().unwrap_or(0));

            while let Some(x) = access.next_element()? {
                vec.push(AtomicU64::new(x));
            }
            Ok(vec.into_boxed_slice())
        }
    }
    let x = AtomicU64SeqVisitor;
    deserializer.deserialize_seq(x)
}

impl core::clone::Clone for BitVector {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits,
            #[cfg(feature = "parallel")]
            vector: self
                .vector
                .iter()
                .map(|x| AtomicU64::new(x.load(Ordering::SeqCst)))
                .collect(),
            #[cfg(not(feature = "parallel"))]
            vector: self.vector.clone(),
        }
    }
}

impl fmt::Display for BitVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        write!(
            f,
            "{}",
            self.iter()
                .fold(String::new(), |x0, x| x0 + &format!("{}, ", x))
        )?;
        write!(f, "]")?;
        Ok(())
    }
}

impl PartialEq for BitVector {
    fn eq(&self, other: &BitVector) -> bool {
        self.eq_left(other, self.bits)
    }
}

impl BitVector {
    /// Build a new empty bitvector
    pub fn new(bits: u64) -> Self {
        let n = u64s(bits);
        let mut v: Vec<Word> = Vec::with_capacity(n as usize);
        for _ in 0..n {
            v.push(Word::default());
        }

        BitVector {
            bits,
            vector: v.into_boxed_slice(),
        }
    }

    /// new bitvector contains all elements
    ///
    /// If `bits % 64 > 0`, the last u64 is guaranteed not to
    /// have any extra 1 bits.
    #[allow(dead_code)]
    pub fn ones(bits: u64) -> Self {
        let (word, offset) = word_offset(bits);
        let mut bvec: Vec<Word> = Vec::with_capacity((word + 1) as usize);
        for _ in 0..word {
            bvec.push(u64::max_value().into());
        }

        let last_val = u64::max_value() >> (64 - offset);
        bvec.push(last_val.into());
        BitVector {
            bits,
            vector: bvec.into_boxed_slice(),
        }
    }

    /// return if this set is empty
    ///
    /// if set does not contain any elements, return true;
    /// else return false.
    ///
    /// This method is averagely faster than `self.len() > 0`.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        #[cfg(feature = "parallel")]
        return self.vector.iter().all(|x| x.load(Ordering::Relaxed) == 0);

        #[cfg(not(feature = "parallel"))]
        return self.vector.iter().all(|x| *x == 0);
    }

    /// the number of elements in set
    pub fn len(&self) -> u64 {
        self.vector.iter().fold(0u64, |x0, x| {
            #[cfg(feature = "parallel")]
            return x0 + x.load(Ordering::Relaxed).count_ones() as u64;

            #[cfg(not(feature = "parallel"))]
            return x0 + x.count_ones() as u64;
        })
    }

    /*
    /// Clear all elements from a bitvector
    pub fn clear(&mut self) {
        for p in &mut self.vector {
            *p = 0;
        }
    }
    */

    /// If `bit` belongs to set, return `true`, else return `false`.
    ///
    /// Insert, remove and contains do not do bound check.
    #[inline]
    pub fn contains(&self, bit: u64) -> bool {
        let (word, mask) = word_mask(bit);
        (self.get_word(word) & mask) != 0
    }

    /// compare if the following is true:
    ///
    /// self \cap {0, 1, ... , bit - 1} == other \cap {0, 1, ... ,bit - 1}
    pub fn eq_left(&self, other: &BitVector, bit: u64) -> bool {
        if bit == 0 {
            return true;
        }
        let (word, offset) = word_offset(bit - 1);
        // We can also use slice comparison, which only take 1 line.
        // However, it has been reported that the `Eq` implementation of slice
        // is extremly slow.
        //
        // self.vector.as_slice()[0 .. word] == other.vector.as_slice[0 .. word]
        //
        self.vector
            .iter()
            .zip(other.vector.iter())
            .take(word as usize)
            .all(|(s1, s2)| {
                #[cfg(feature = "parallel")]
                return s1.load(Ordering::Relaxed) == s2.load(Ordering::Relaxed);

                #[cfg(not(feature = "parallel"))]
                return s1 == s2;
            })
            && (self.get_word(word as usize) << (63 - offset))
                == (other.get_word(word as usize) << (63 - offset))
    }

    /// insert a new element to set
    ///
    /// If value is inserted, return true,
    /// if value already exists in set, return false.
    ///
    /// Insert, remove and contains do not do bound check.
    #[inline]
    #[cfg(feature = "parallel")]
    pub fn insert(&self, bit: u64) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &self.vector[word];

        let prev = data.fetch_or(mask, Ordering::Relaxed);
        prev & mask == 0
    }

    #[inline]
    #[cfg(not(feature = "parallel"))]
    pub fn insert(&mut self, bit: u64) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &mut self.vector[word];

        let prev = *data;
        *data = *data | mask;
        prev & mask == 0
    }

    /// insert a new element synchronously.
    /// requires &mut self, but doesn't use
    /// atomic instructions so may be faster
    /// than `insert()`.
    ///
    /// If value is inserted, return true,
    /// if value already exists in set, return false.
    ///
    /// Insert, remove and contains do not do bound check.
    #[inline]
    pub fn insert_sync(&mut self, bit: u64) -> bool {
        let (word, mask) = word_mask(bit);
        #[cfg(feature = "parallel")]
        let data = self.vector[word].get_mut();
        #[cfg(not(feature = "parallel"))]
        let data = &mut self.vector[word];

        let old_data = *data;
        *data |= mask;
        old_data & mask == 0
    }

    /// remove an element from set
    ///
    /// If value is removed, return true,
    /// if value doesn't exist in set, return false.
    ///
    /// Insert, remove and contains do not do bound check.
    #[cfg(feature = "parallel")]
    pub fn remove(&self, bit: u64) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &self.vector[word];

        let prev = data.fetch_and(!mask, Ordering::Relaxed);
        prev & mask != 0
    }

    #[cfg(not(feature = "parallel"))]
    pub fn remove(&mut self, bit: u64) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &mut self.vector[word];

        let prev = *data;
        *data = *data & !mask;
        prev & mask != 0
    }

    /// import elements from another bitvector
    ///
    /// If any new value is inserted, return true,
    /// else return false.
    #[allow(dead_code)]
    #[cfg(feature = "parallel")]
    pub fn insert_all(&self, all: &BitVector) -> bool {
        assert!(self.vector.len() == all.vector.len());
        let mut changed = false;

        for (i, j) in self.vector.iter().zip(all.vector.iter()) {
            let prev = i.fetch_or(j.load(Ordering::Relaxed), Ordering::Relaxed);

            if prev != i.load(Ordering::Relaxed) {
                changed = true;
            }
        }

        changed
    }

    #[allow(dead_code)]
    #[cfg(not(feature = "parallel"))]
    pub fn insert_all(&mut self, all: &BitVector) -> bool {
        assert!(self.vector.len() == all.vector.len());
        let mut changed = false;

        for (i, j) in self.vector.iter_mut().zip(all.vector.iter()) {
            let prev = *i;
            *i |= *j;

            if prev != *i {
                changed = true;
            }
        }

        changed
    }

    /// the max number of elements can be inserted into set
    pub fn capacity(&self) -> u64 {
        self.bits
    }

    #[inline]
    pub fn get_word(&self, word: usize) -> u64 {
        #[cfg(feature = "parallel")]
        return self.vector[word].load(Ordering::Relaxed) as u64;

        #[cfg(not(feature = "parallel"))]
        return self.vector[word] as u64;
    }

    pub fn num_words(&self) -> usize {
        self.vector.len()
    }

    /// Return a iterator of the set element in the bitvector,
    pub fn iter(&self) -> BitVectorIter<'_> {
        BitVectorIter {
            iter: self.vector.iter(),
            current: 0,
            idx: 0,
            size: self.bits,
        }
    }
}

/// Iterator for BitVector
pub struct BitVectorIter<'a> {
    iter: ::std::slice::Iter<'a, Word>,
    current: u64,
    idx: u64,
    size: u64,
}

impl<'a> Iterator for BitVectorIter<'a> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.idx >= self.size {
            return None;
        }
        while self.current == 0 {
            self.current = if let Some(_i) = self.iter.next() {
                #[cfg(feature = "parallel")]
                let i = _i.load(Ordering::Relaxed);
                #[cfg(not(feature = "parallel"))]
                let i = *_i;
                if i == 0 {
                    self.idx += 64;
                    continue;
                } else {
                    self.idx = u64s(self.idx) * 64;
                    i
                }
            } else {
                return None;
            }
        }
        let offset = self.current.trailing_zeros() as u64;
        self.current >>= offset;
        self.current >>= 1; // shift otherwise overflows for 0b1000_0000_…_0000
        self.idx += offset + 1;
        Some(self.idx - 1)
    }
}

fn u64s(elements: u64) -> u64 {
    (elements + 63) / 64
}

fn word_offset(index: u64) -> (u64, u64) {
    (index / 64, index % 64)
}

#[inline]
fn word_mask(index: u64) -> (usize, u64) {
    let word = (index / 64) as usize;
    let mask = 1 << (index % 64);
    (word, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn union_two_vecs() {
        #[allow(unused_mut)]
        let mut vec1 = BitVector::new(65);
        #[allow(unused_mut)]
        let mut vec2 = BitVector::new(65);
        assert!(vec1.insert(3));
        assert!(!vec1.insert(3));
        assert!(vec2.insert(5));
        assert!(vec2.insert(64));
        assert!(vec1.insert_all(&vec2));
        assert!(!vec1.insert_all(&vec2));
        assert!(vec1.contains(3));
        assert!(!vec1.contains(4));
        assert!(vec1.contains(5));
        assert!(!vec1.contains(63));
        assert!(vec1.contains(64));
    }

    #[test]
    fn bitvec_iter_works() {
        #[allow(unused_mut)]
        let mut bitvec = BitVector::new(100);
        bitvec.insert(1);
        bitvec.insert(10);
        bitvec.insert(19);
        bitvec.insert(62);
        bitvec.insert(63);
        bitvec.insert(64);
        bitvec.insert(65);
        bitvec.insert(66);
        bitvec.insert(99);
        assert_eq!(
            bitvec.iter().collect::<Vec<_>>(),
            [1, 10, 19, 62, 63, 64, 65, 66, 99]
        );
    }

    #[test]
    fn bitvec_iter_works_2() {
        #[allow(unused_mut)]
        let mut bitvec = BitVector::new(319);
        bitvec.insert(0);
        bitvec.insert(127);
        bitvec.insert(191);
        bitvec.insert(255);
        bitvec.insert(319);
        assert_eq!(bitvec.iter().collect::<Vec<_>>(), [0, 127, 191, 255, 319]);
    }

    #[test]
    fn eq_left() {
        #[allow(unused_mut)]
        let mut bitvec = BitVector::new(50);
        for i in &[0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(*i);
        }
        #[allow(unused_mut)]
        let mut bitvec2 = BitVector::new(50);
        for i in &[0, 1, 3, 5, 7, 11, 13, 17, 19, 23] {
            bitvec2.insert(*i);
        }

        assert!(bitvec.eq_left(&bitvec2, 1));
        assert!(bitvec.eq_left(&bitvec2, 2));
        assert!(bitvec.eq_left(&bitvec2, 3));
        assert!(bitvec.eq_left(&bitvec2, 4));
        assert!(bitvec.eq_left(&bitvec2, 5));
        assert!(bitvec.eq_left(&bitvec2, 6));
        assert!(bitvec.eq_left(&bitvec2, 7));
        assert!(!bitvec.eq_left(&bitvec2, 8));
        assert!(!bitvec.eq_left(&bitvec2, 9));
        assert!(!bitvec.eq_left(&bitvec2, 50));
    }

    #[test]
    fn eq() {
        #[allow(unused_mut)]
        let mut bitvec = BitVector::new(50);
        for i in &[0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(*i);
        }
        #[allow(unused_mut)]
        let mut bitvec2 = BitVector::new(50);
        for i in &[0, 1, 3, 5, 7, 11, 13, 17, 19, 23] {
            bitvec2.insert(*i);
        }
        #[allow(unused_mut)]
        let mut bitvec3 = BitVector::new(50);
        for i in &[0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec3.insert(*i);
        }

        assert!(bitvec != bitvec2);
        assert!(bitvec == bitvec3);
        assert!(bitvec2 != bitvec3);
    }

    #[test]
    fn remove() {
        #[allow(unused_mut)]
        let mut bitvec = BitVector::new(50);
        for i in &[0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(*i);
        }
        assert!(bitvec.contains(3));
        bitvec.remove(3);
        assert!(!bitvec.contains(3));
        assert_eq!(
            bitvec.iter().collect::<Vec<_>>(),
            vec![0, 1, 5, 11, 12, 19, 23]
        );
    }

    #[test]
    fn is_empty() {
        assert!(!BitVector::ones(60).is_empty());
        assert!(!BitVector::ones(65).is_empty());
        #[allow(unused_mut)]
        let mut bvec = BitVector::new(60);

        assert!(bvec.is_empty());

        bvec.insert(5);
        assert!(!bvec.is_empty());
        bvec.remove(5);
        assert!(bvec.is_empty());
        #[allow(unused_mut)]
        let mut bvec = BitVector::ones(65);
        for i in 0..65 {
            bvec.remove(i);
        }
        assert!(bvec.is_empty());
    }

    #[test]
    fn test_ones() {
        let bvec = BitVector::ones(60);
        for i in 0..60 {
            assert!(bvec.contains(i));
        }
        assert_eq!(bvec.iter().collect::<Vec<_>>(), (0..60).collect::<Vec<_>>());
    }

    #[test]
    fn len() {
        assert_eq!(BitVector::ones(60).len(), 60);
        assert_eq!(BitVector::ones(65).len(), 65);
        assert_eq!(BitVector::new(65).len(), 0);
        #[allow(unused_mut)]
        let mut bvec = BitVector::new(60);
        bvec.insert(5);
        assert_eq!(bvec.len(), 1);
        bvec.insert(6);
        assert_eq!(bvec.len(), 2);
        bvec.remove(5);
        assert_eq!(bvec.len(), 1);
    }
}

#[cfg(all(feature = "unstable", test))]
mod bench {
    extern crate test;
    use self::test::Bencher;
    use super::*;
    use std::collections::{BTreeSet, HashSet};
    #[bench]
    fn bench_bitset_operator(b: &mut Bencher) {
        b.iter(|| {
            #[allow(unused_mut)]
            let mut vec1 = BitVector::new(65);
            #[allow(unused_mut)]
            let mut vec2 = BitVector::new(65);
            for i in vec![0, 1, 2, 10, 15, 18, 25, 31, 40, 42, 60, 64] {
                vec1.insert(i);
            }
            for i in vec![3, 5, 7, 12, 13, 15, 21, 25, 30, 29, 42, 50, 61, 62, 63, 64] {
                vec2.insert(i);
            }
            vec1.intersection(&vec2);
            vec1.union(&vec2);
            vec1.difference(&vec2);
        });
    }

    #[bench]
    fn bench_bitset_operator_inplace(b: &mut Bencher) {
        b.iter(|| {
            #[allow(unused_mut)]
            let mut vec1 = BitVector::new(65);
            #[allow(unused_mut)]
            let mut vec2 = BitVector::new(65);
            for i in vec![0, 1, 2, 10, 15, 18, 25, 31, 40, 42, 60, 64] {
                vec1.insert(i);
            }
            for i in vec![3, 5, 7, 12, 13, 15, 21, 25, 30, 29, 42, 50, 61, 62, 63, 64] {
                vec2.insert(i);
            }
            vec1.intersection_inplace(&vec2);
            vec1.union_inplace(&vec2);
            vec1.difference_inplace(&vec2);
        });
    }

    #[bench]
    fn bench_hashset_operator(b: &mut Bencher) {
        b.iter(|| {
            #[allow(unused_mut)]
            let mut vec1 = HashSet::with_capacity(65);
            #[allow(unused_mut)]
            let mut vec2 = HashSet::with_capacity(65);
            for i in vec![0, 1, 2, 10, 15, 18, 25, 31, 40, 42, 60, 64] {
                vec1.insert(i);
            }
            for i in vec![3, 5, 7, 12, 13, 15, 21, 25, 30, 29, 42, 50, 61, 62, 63, 64] {
                vec2.insert(i);
            }

            vec1.intersection(&vec2).cloned().collect::<HashSet<_>>();
            vec1.union(&vec2).cloned().collect::<HashSet<_>>();
            vec1.difference(&vec2).cloned().collect::<HashSet<_>>();
        });
    }

    #[bench]
    fn bench_btreeset_operator(b: &mut Bencher) {
        b.iter(|| {
            #[allow(unused_mut)]
            let mut vec1 = BTreeSet::new();
            #[allow(unused_mut)]
            let mut vec2 = BTreeSet::new();
            for i in vec![0, 1, 2, 10, 15, 18, 25, 31, 40, 42, 60, 64] {
                vec1.insert(i);
            }
            for i in vec![3, 5, 7, 12, 13, 15, 21, 25, 30, 29, 42, 50, 61, 62, 63, 64] {
                vec2.insert(i);
            }

            vec1.intersection(&vec2).cloned().collect::<HashSet<_>>();
            vec1.union(&vec2).cloned().collect::<HashSet<_>>();
            vec1.difference(&vec2).cloned().collect::<HashSet<_>>();
        });
    }
}
