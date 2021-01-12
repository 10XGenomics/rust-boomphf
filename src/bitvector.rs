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
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "serde")]
use serde::{self, Deserialize, Serialize};

/// Bitvector
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BitVector {
    bits: usize,
    #[cfg_attr(
        feature = "serde",
        serde(serialize_with = "ser_atomic_vec", deserialize_with = "de_atomic_vec")
    )]
    vector: Box<[AtomicUsize]>,
}

// Custom serializer
#[cfg(feature = "serde")]
fn ser_atomic_vec<S>(v: &Box<[AtomicUsize]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(v.len()))?;
    for ref x in v.iter() {
        seq.serialize_element(&x.load(Ordering::SeqCst))?;
    }
    seq.end()
}

// Custom deserializer
#[cfg(feature = "serde")]
pub fn de_atomic_vec<'de, D>(deserializer: D) -> Result<Box<[AtomicUsize]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct AtomicUsizeSeqVisitor;

    impl<'de> serde::de::Visitor<'de> for AtomicUsizeSeqVisitor {
        type Value = Box<[AtomicUsize]>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a 64bit unsigned integer")
        }

        fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
        where
            S: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::<AtomicUsize>::with_capacity(access.size_hint().unwrap_or(0));

            while let Some(x) = access.next_element()? {
                vec.push(AtomicUsize::new(x));
            }
            Ok(vec.into_boxed_slice())
        }
    }
    let x = AtomicUsizeSeqVisitor;
    deserializer.deserialize_seq(x)
}

impl core::clone::Clone for BitVector {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits,
            vector: self
                .vector
                .iter()
                .map(|x| AtomicUsize::new(x.load(Ordering::SeqCst)))
                .collect(),
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

fn to_au(v: usize) -> AtomicUsize {
    AtomicUsize::new(v)
}

impl BitVector {
    /// Build a new empty bitvector
    pub fn new(bits: usize) -> Self {
        let n = u64s(bits);
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(to_au(0));
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
    pub fn ones(bits: usize) -> Self {
        let (word, offset) = word_offset(bits);
        let mut bvec = Vec::with_capacity(word + 1);
        for _ in 0..word {
            bvec.push(to_au(usize::max_value()));
        }

        bvec.push(to_au(usize::max_value() >> (64 - offset)));
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
        self.vector.iter().all(|x| x.load(Ordering::Relaxed) == 0)
    }

    /// the number of elements in set
    pub fn len(&self) -> usize {
        self.vector.iter().fold(0usize, |x0, x| {
            x0 + x.load(Ordering::Relaxed).count_ones() as usize
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
    pub fn contains(&self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        (self.get_word(word) as usize & mask) != 0
    }

    /// compare if the following is true:
    ///
    /// self \cap {0, 1, ... , bit - 1} == other \cap {0, 1, ... ,bit - 1}
    pub fn eq_left(&self, other: &BitVector, bit: usize) -> bool {
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
            .take(word)
            .all(|(s1, s2)| s1.load(Ordering::Relaxed) == s2.load(Ordering::Relaxed))
            && (self.get_word(word) << (63 - offset)) == (other.get_word(word) << (63 - offset))
    }

    /// insert a new element to set
    ///
    /// If value is inserted, return true,
    /// if value already exists in set, return false.
    ///
    /// Insert, remove and contains do not do bound check.
    #[inline]
    pub fn insert(&self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &self.vector[word];

        let prev = data.fetch_or(mask, Ordering::Relaxed);
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
    pub fn insert_sync(&mut self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        let data = self.vector[word].get_mut();

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
    pub fn remove(&self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &self.vector[word];

        let prev = data.fetch_and(!mask, Ordering::Relaxed);
        prev & mask != 0
    }

    /// import elements from another bitvector
    ///
    /// If any new value is inserted, return true,
    /// else return false.
    #[allow(dead_code)]
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

    /// the max number of elements can be inserted into set
    pub fn capacity(&self) -> usize {
        self.bits
    }

    #[inline]
    pub fn get_word(&self, word: usize) -> u64 {
        self.vector[word].load(Ordering::Relaxed) as u64
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
    iter: ::std::slice::Iter<'a, AtomicUsize>,
    current: usize,
    idx: usize,
    size: usize,
}

impl<'a> Iterator for BitVectorIter<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.idx >= self.size {
            return None;
        }
        while self.current == 0 {
            self.current = if let Some(_i) = self.iter.next() {
                let i = _i.load(Ordering::Relaxed);
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
        let offset = self.current.trailing_zeros() as usize;
        self.current >>= offset;
        self.current >>= 1; // shift otherwise overflows for 0b1000_0000_…_0000
        self.idx += offset + 1;
        Some(self.idx - 1)
    }
}

fn u64s(elements: usize) -> usize {
    (elements + 63) / 64
}

fn word_offset(index: usize) -> (usize, usize) {
    (index / 64, index % 64)
}

#[inline]
fn word_mask(index: usize) -> (usize, usize) {
    let word = index / 64;
    let mask = 1 << (index % 64);
    (word, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn union_two_vecs() {
        let vec1 = BitVector::new(65);
        let vec2 = BitVector::new(65);
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
        let bitvec = BitVector::new(100);
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
        let bitvec = BitVector::new(319);
        bitvec.insert(0);
        bitvec.insert(127);
        bitvec.insert(191);
        bitvec.insert(255);
        bitvec.insert(319);
        assert_eq!(bitvec.iter().collect::<Vec<_>>(), [0, 127, 191, 255, 319]);
    }

    #[test]
    fn eq_left() {
        let bitvec = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(i);
        }
        let bitvec2 = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 7, 11, 13, 17, 19, 23] {
            bitvec2.insert(i);
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
        let bitvec = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(i);
        }
        let bitvec2 = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 7, 11, 13, 17, 19, 23] {
            bitvec2.insert(i);
        }
        let bitvec3 = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec3.insert(i);
        }

        assert!(bitvec != bitvec2);
        assert!(bitvec == bitvec3);
        assert!(bitvec2 != bitvec3);
    }

    #[test]
    fn remove() {
        let bitvec = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(i);
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
        let bvec = BitVector::new(60);

        assert!(bvec.is_empty());

        bvec.insert(5);
        assert!(!bvec.is_empty());
        bvec.remove(5);
        assert!(bvec.is_empty());
        let bvec = BitVector::ones(65);
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
        let bvec = BitVector::new(60);
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
            let vec1 = BitVector::new(65);
            let vec2 = BitVector::new(65);
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
            let mut vec1 = BitVector::new(65);
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
            let mut vec1 = HashSet::with_capacity(65);
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
            let mut vec1 = BTreeSet::new();
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
