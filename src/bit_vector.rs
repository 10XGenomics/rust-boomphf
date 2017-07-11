//! ### BitVector Module
//!
//! BitVector uses one bit to represent a bool state.
//! BitVector is useful for the programs that need fast set operation (intersection, union,
//! difference), because that all these operations can be done with simple bitand, bitor, bitxor.
//!
//! Usually, the length of a BitVector should not be changed after constructed, for example:
//!
//! ```
//! extern crate bitvector;
//! use bitvector::*;
//!
//! fn main(){
//!   // a bitvector contains 30 elements
//!   let mut bitvec = BitVector::new(30);
//!   // add 10 elements
//!   for i in 0 .. 10 { bitvec.insert(i); }
//!   // you can use Iterator to iter over all the elements
//!   assert_eq!(bitvec.iter().collect::<Vec<_>>(), vec![0,1,2,3,4,5,6,7,8,9]);
//!
//!   let mut bitvec2 = BitVector::new(30);
//!   for i in 5 .. 15 { bitvec2.insert(i); }
//!
//!   // set union
//!   assert_eq!(bitvec.union(&bitvec2).iter().collect::<Vec<_>>(),
//!              vec![0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]);
//!
//!   // set intersection
//!   assert_eq!(bitvec.intersection(&bitvec2).iter().collect::<Vec<_>>(),
//!              vec![5,6,7,8,9]);
//!
//!   // set difference
//!   assert_eq!(bitvec.difference(&bitvec2).iter().collect::<Vec<_>>(),
//!              vec![0,1,2,3,4]);
//!
//!   // you can also use `&`(intersection) `|`(union) and `^`(difference)
//!   // to do the set operations
//!   assert_eq!((&bitvec ^ &bitvec2).iter().collect::<Vec<_>>(),
//!              vec![0,1,2,3,4]);
//! }
//! ```
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

#![cfg_attr(feature = "unstable", feature(test))]
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Bitvector
#[derive(Debug)]
pub struct BitVector {
    bits: usize,
    vector: Vec<AtomicUsize>,
}

impl fmt::Display for BitVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "["));
        try!(write!(f,
                    "{}",
                    self.iter().fold(String::new(), |x0, x| x0 + &format!("{}, ", x))));
        write!(f, "]")
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
        let mut v = Vec::new();
        for _ in 0 .. n {
            v.push(to_au(0));
        }

        BitVector {
            bits: bits,
            vector: v,
        }
    }

    /// new bitvector contains all elements
    ///
    /// If `bits % 64 > 0`, the last u64 is guaranteed not to
    /// have any extra 1 bits.
    pub fn ones(bits: usize) -> Self {
        let (word, offset) = word_offset(bits);
        let mut bvec = Vec::new();
        for _ in 0 .. word {
            bvec.push(to_au(usize::max_value()));
        }

        bvec.push(to_au(usize::max_value() >> (64 - offset)));
        BitVector {
            bits: bits,
            vector: bvec,
        }
    }

    /// return if this set is empty
    ///
    /// if set does not contain any elements, return true;
    /// else return false.
    ///
    /// This method is averagely faster than `self.len() > 0`.
    pub fn is_empty(&self) -> bool {
        self.vector.iter().all(|x| x.load(Ordering::Relaxed) == 0)
    }

    /// the number of elements in set
    pub fn len(&self) -> usize {
        self.vector.iter().fold(0usize, |x0, x| x0 + x.load(Ordering::Relaxed).count_ones() as usize)
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
    pub fn contains(&self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        (self.get_word(word) as usize & mask) != 0
    }

    /// compare if the following is true:
    ///
    /// self \cap {0, 1, ... , bit - 1} == other \cap {0, 1, ... ,bit - 1}
    ///
    /// for example:
    ///
    /// ```
    /// use bitvector::*;
    ///
    /// let mut A = BitVector::new(11);
    /// let mut B = BitVector::new(11);
    /// for i in vec![0, 1, 3 ,5 ,7, 10] { A.insert(i); }
    /// for i in vec![0, 1, 3, 4, 5, 7, 10] { B.insert(i); }
    ///
    ///
    /// assert!(A.eq_left(&B, 1));  // [0             ]  = [0              ]
    /// assert!(A.eq_left(&B, 2));  // [0, 1          ]  = [0, 1           ]
    /// assert!(A.eq_left(&B, 3));  // [0, 1          ]  = [0, 1           ]
    /// assert!(A.eq_left(&B, 4));  // [0, 1,   3     ]  = [0, 1,   3      ]
    /// assert!(!A.eq_left(&B, 5)); // [0, 1,   3     ] != [0, 1,   3, 4   ]
    /// assert!(!A.eq_left(&B, 6)); // [0, 1,   3,   5] != [0, 1,   3, 4, 5]
    /// ```
    ///
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
        self.vector.iter().zip(other.vector.iter()).take(word).all(|(s1, s2)| s1.load(Ordering::Relaxed) == s2.load(Ordering::Relaxed)) &&
        (self.get_word(word) << (63 - offset)) == (other.get_word(word) << (63 - offset))
    }

    /// insert a new element to set
    ///
    /// If value is inserted, return true,
    /// if value already exists in set, return false.
    ///
    /// Insert, remove and contains do not do bound check.
    pub fn insert(&mut self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &mut self.vector[word];

        let prev = data.fetch_or(mask, Ordering::Relaxed);
        prev & mask == 0
    }

    /// remove an element from set
    ///
    /// If value is removed, return true,
    /// if value doesn't exist in set, return false.
    ///
    /// Insert, remove and contains do not do bound check.
    pub fn remove(&mut self, bit: usize) -> bool {
        let (word, mask) = word_mask(bit);
        let data = &mut self.vector[word];

        let prev = data.fetch_and(!mask, Ordering::Relaxed);
        prev & mask != 0
    }


    /// import elements from another bitvector
    ///
    /// If any new value is inserted, return true,
    /// else return false.
    pub fn insert_all(&mut self, all: &BitVector) -> bool {
        assert!(self.vector.len() == all.vector.len());
        let mut changed = false;
        for (i, j) in self.vector.iter_mut().zip(&all.vector) {

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

    pub fn get_word(&self, word: usize) -> u64 {
        self.vector[word].load(Ordering::Relaxed) as u64
    }

    pub fn num_words(&self) -> usize {
        self.vector.len()
    }


    /// Return a iterator of element based on current bitvector,
    /// for example:
    ///
    /// ```
    /// extern crate bitvector;
    /// use bitvector::*;
    ///
    /// fn main() {
    ///     let mut bitvec = BitVector::new(5);
    ///     bitvec.insert(2);
    ///     bitvec.insert(3);
    ///     // The bitvector becomes: 0x00 0x00 0x00 0x0C
    ///     assert_eq!(bitvec.iter().collect::<Vec<_>>(), vec![2,3]);
    ///     // collected vector will contains the real element not the bit.
    /// }
    /// ```
    pub fn iter<'a>(&'a self) -> BitVectorIter<'a> {
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
        return Some(self.idx - 1);
    }
}



fn u64s(elements: usize) -> usize {
    (elements + 63) / 64
}

fn word_offset(index: usize) -> (usize, usize) {
    (index / 64, index % 64)
}

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
        let mut vec1 = BitVector::new(65);
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
        assert_eq!(bitvec.iter().collect::<Vec<_>>(),
                   [1, 10, 19, 62, 63, 64, 65, 66, 99]);
    }


    #[test]
    fn bitvec_iter_works_2() {
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
        let mut bitvec = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(i);
        }
        let mut bitvec2 = BitVector::new(50);
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
        let mut bitvec = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(i);
        }
        let mut bitvec2 = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 7, 11, 13, 17, 19, 23] {
            bitvec2.insert(i);
        }
        let mut bitvec3 = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec3.insert(i);
        }

        assert!(bitvec != bitvec2);
        assert!(bitvec == bitvec3);
        assert!(bitvec2 != bitvec3);
    }

    #[test]
    fn remove() {
        let mut bitvec = BitVector::new(50);
        for i in vec![0, 1, 3, 5, 11, 12, 19, 23] {
            bitvec.insert(i);
        }
        assert!(bitvec.contains(3));
        bitvec.remove(3);
        assert!(!bitvec.contains(3));
        assert_eq!(bitvec.iter().collect::<Vec<_>>(),
                   vec![0, 1, 5, 11, 12, 19, 23]);
    }

    #[test]
    fn is_empty() {
        assert!(!BitVector::ones(60).is_empty());
        assert!(!BitVector::ones(65).is_empty());
        let mut bvec = BitVector::new(60);

        assert!(bvec.is_empty());

        bvec.insert(5);
        assert!(!bvec.is_empty());
        bvec.remove(5);
        assert!(bvec.is_empty());
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
    use std::collections::{HashSet, BTreeSet};
    use self::test::Bencher;
    use super::*;
    #[bench]
    fn bench_bitset_operator(b: &mut Bencher) {

        b.iter(|| {
            let mut vec1 = BitVector::new(65);
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