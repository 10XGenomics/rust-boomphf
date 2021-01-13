//! HashMap data structures, using MPHFs to encode the position of each key in a dense array.

#[cfg(feature = "serde")]
use serde::{self, Deserialize, Serialize};

use crate::Mphf;
use std::borrow::Borrow;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::ExactSizeIterator;

/// A HashMap data structure where the mapping between keys and values is encoded in a Mphf. This lets us store the keys and values in dense
/// arrays, with ~3 bits/item overhead in the Mphf.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BoomHashMap<K: Hash, D> {
    mphf: Mphf<K>,
    pub(crate) keys: Vec<K>,
    pub(crate) values: Vec<D>,
}

impl<K, D> BoomHashMap<K, D>
where
    K: Hash + Debug + PartialEq,
    D: Debug,
{
    fn create_map(mut keys: Vec<K>, mut values: Vec<D>, mphf: Mphf<K>) -> BoomHashMap<K, D> {
        // reorder the keys and values according to the Mphf
        for i in 0..keys.len() {
            loop {
                let kmer_slot = mphf.hash(&keys[i]) as usize;
                if i == kmer_slot {
                    break;
                }
                keys.swap(i, kmer_slot);
                values.swap(i, kmer_slot);
            }
        }
        BoomHashMap { mphf, keys, values }
    }

    /// Create a new hash map from the parallel array `keys` and `values`
    pub fn new(keys: Vec<K>, data: Vec<D>) -> BoomHashMap<K, D> {
        let mphf = Mphf::new(1.7, &keys);
        Self::create_map(keys, data, mphf)
    }

    /// Get the value associated with `key`, if available, otherwise return None
    pub fn get<Q: ?Sized>(&self, kmer: &Q) -> Option<&D>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let maybe_pos = self.mphf.try_hash(kmer);
        match maybe_pos {
            Some(pos) => {
                let hashed_kmer = &self.keys[pos as usize];
                if kmer == hashed_kmer.borrow() {
                    Some(&self.values[pos as usize])
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Get the position in the Mphf of a key, if the key exists.
    pub fn get_key_id<Q: ?Sized>(&self, kmer: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let maybe_pos = self.mphf.try_hash(&kmer);
        match maybe_pos {
            Some(pos) => {
                let hashed_kmer = &self.keys[pos as usize];
                if kmer == hashed_kmer.borrow() {
                    Some(pos as usize)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    /// Total number of key/value pairs
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn get_key(&self, id: usize) -> Option<&K> {
        let max_key_id = self.len();
        if id > max_key_id {
            None
        } else {
            Some(&self.keys[id])
        }
    }

    pub fn iter(&self) -> BoomIterator<K, D> {
        BoomIterator {
            hash: self,
            index: 0,
        }
    }
}

impl<K, D> BoomHashMap<K, D>
where
    K: Hash + Debug + PartialEq + Send + Sync,
    D: Debug,
{
    /// Create a new hash map from the parallel array `keys` and `values`, using a parallelized method to construct the Mphf.
    pub fn new_parallel(keys: Vec<K>, data: Vec<D>) -> BoomHashMap<K, D> {
        let mphf = Mphf::new_parallel(1.7, &keys, None);
        Self::create_map(keys, data, mphf)
    }
}

/// Iterate over key-value pairs in a BoomHashMap
pub struct BoomIterator<'a, K: Hash + 'a, D: 'a> {
    hash: &'a BoomHashMap<K, D>,
    index: usize,
}

impl<'a, K: Hash, D> Iterator for BoomIterator<'a, K, D> {
    type Item = (&'a K, &'a D);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.hash.keys.len() {
            return None;
        }

        let elements = Some((&self.hash.keys[self.index], &self.hash.values[self.index]));
        self.index += 1;

        elements
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.hash.keys.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, K: Hash, D1> ExactSizeIterator for BoomIterator<'a, K, D1> {}

impl<'a, K: Hash, D> IntoIterator for &'a BoomHashMap<K, D> {
    type Item = (&'a K, &'a D);
    type IntoIter = BoomIterator<'a, K, D>;

    fn into_iter(self) -> BoomIterator<'a, K, D> {
        BoomIterator {
            hash: self,
            index: 0,
        }
    }
}

/// A HashMap data structure where the mapping between keys and 2 values is encoded in a Mphf. You should usually use `BoomHashMap` with a tuple/struct value type.
/// If the layout overhead of the struct / tuple must be avoided, this variant of is an alternative.
/// This lets us store the keys and values in dense
/// arrays, with ~3 bits/item overhead in the Mphf.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BoomHashMap2<K: Hash, D1, D2> {
    mphf: Mphf<K>,
    keys: Vec<K>,
    values: Vec<D1>,
    aux_values: Vec<D2>,
}

pub struct Boom2Iterator<'a, K: Hash + 'a, D1: 'a, D2: 'a> {
    hash: &'a BoomHashMap2<K, D1, D2>,
    index: usize,
}

impl<'a, K: Hash, D1, D2> Iterator for Boom2Iterator<'a, K, D1, D2> {
    type Item = (&'a K, &'a D1, &'a D2);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.hash.keys.len() {
            return None;
        }

        let elements = Some((
            &self.hash.keys[self.index],
            &self.hash.values[self.index],
            &self.hash.aux_values[self.index],
        ));
        self.index += 1;
        elements
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.hash.keys.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a, K: Hash, D1, D2> ExactSizeIterator for Boom2Iterator<'a, K, D1, D2> {}

impl<'a, K: Hash, D1, D2> IntoIterator for &'a BoomHashMap2<K, D1, D2> {
    type Item = (&'a K, &'a D1, &'a D2);
    type IntoIter = Boom2Iterator<'a, K, D1, D2>;

    fn into_iter(self) -> Boom2Iterator<'a, K, D1, D2> {
        Boom2Iterator {
            hash: self,
            index: 0,
        }
    }
}

impl<K, D1, D2> BoomHashMap2<K, D1, D2>
where
    K: Hash + Debug + PartialEq,
    D1: Debug,
    D2: Debug,
{
    fn create_map(
        mut keys: Vec<K>,
        mut values: Vec<D1>,
        mut aux_values: Vec<D2>,
        mphf: Mphf<K>,
    ) -> BoomHashMap2<K, D1, D2> {
        // reorder the keys and values according to the Mphf
        for i in 0..keys.len() {
            loop {
                let kmer_slot = mphf.hash(&keys[i]) as usize;
                if i == kmer_slot {
                    break;
                }
                keys.swap(i, kmer_slot);
                values.swap(i, kmer_slot);
                aux_values.swap(i, kmer_slot);
            }
        }

        BoomHashMap2 {
            mphf,
            keys,
            values,
            aux_values,
        }
    }

    /// Create a new hash map from the parallel arrays `keys` and `values`, and `aux_values`
    pub fn new(keys: Vec<K>, values: Vec<D1>, aux_values: Vec<D2>) -> BoomHashMap2<K, D1, D2> {
        let mphf = Mphf::new(1.7, &keys);
        Self::create_map(keys, values, aux_values, mphf)
    }

    pub fn get<Q: ?Sized>(&self, kmer: &Q) -> Option<(&D1, &D2)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let maybe_pos = self.mphf.try_hash(kmer);
        match maybe_pos {
            Some(pos) => {
                let hashed_kmer = &self.keys[pos as usize];
                if kmer == hashed_kmer.borrow() {
                    Some((&self.values[pos as usize], &self.aux_values[pos as usize]))
                } else {
                    None
                }
            }
            None => None,
        }
    }

    pub fn get_key_id<Q: ?Sized>(&self, kmer: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let maybe_pos = self.mphf.try_hash(&kmer);
        match maybe_pos {
            Some(pos) => {
                let hashed_kmer = &self.keys[pos as usize];
                if kmer == hashed_kmer.borrow() {
                    Some(pos as usize)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    // Return iterator over key-values pairs
    pub fn iter(&self) -> Boom2Iterator<K, D1, D2> {
        Boom2Iterator {
            hash: self,
            index: 0,
        }
    }

    pub fn get_key(&self, id: usize) -> Option<&K> {
        let max_key_id = self.len();
        if id > max_key_id {
            None
        } else {
            Some(&self.keys[id])
        }
    }
}

impl<K, D1, D2> BoomHashMap2<K, D1, D2>
where
    K: Hash + Debug + PartialEq + Send + Sync,
    D1: Debug,
    D2: Debug,
{
    /// Create a new hash map from the parallel arrays `keys` and `values`, and `aux_values`, using a parallel algorithm to construct the Mphf.
    pub fn new_parallel(keys: Vec<K>, data: Vec<D1>, aux_data: Vec<D2>) -> BoomHashMap2<K, D1, D2> {
        let mphf = Mphf::new_parallel(1.7, &keys, None);
        Self::create_map(keys, data, aux_data, mphf)
    }
}

/// A HashMap data structure where the mapping between keys and values is encoded in a Mphf. *Keys are not stored* - this can greatly improve the memory consumption,
/// but can only be used if you can guarantee that you will only query for keys that were in the original set.  Querying for a new key will return a random value, silently.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NoKeyBoomHashMap<K, D1> {
    pub mphf: Mphf<K>,
    pub values: Vec<D1>,
}

impl<K, D1> NoKeyBoomHashMap<K, D1>
where
    K: Hash + Debug + PartialEq + Send + Sync,
    D1: Debug,
{
    pub fn new_parallel(mut keys: Vec<K>, mut values: Vec<D1>) -> NoKeyBoomHashMap<K, D1> {
        let mphf = Mphf::new_parallel(1.7, &keys, None);
        for i in 0..keys.len() {
            loop {
                let kmer_slot = mphf.hash(&keys[i]) as usize;
                if i == kmer_slot {
                    break;
                }
                keys.swap(i, kmer_slot);
                values.swap(i, kmer_slot);
            }
        }

        NoKeyBoomHashMap { mphf, values }
    }

    pub fn new_with_mphf(mphf: Mphf<K>, values: Vec<D1>) -> NoKeyBoomHashMap<K, D1> {
        NoKeyBoomHashMap { mphf, values }
    }

    /// Get the value associated with `key`, if available, otherwise return None
    pub fn get<Q: ?Sized>(&self, kmer: &Q) -> Option<&D1>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let maybe_pos = self.mphf.try_hash(kmer);
        match maybe_pos {
            Some(pos) => Some(&self.values[pos as usize]),
            _ => None,
        }
    }
}

/// A HashMap data structure where the mapping between keys and values is encoded in a Mphf. *Keys are not stored* - this can greatly improve the memory consumption,
/// but can only be used if you can guarantee that you will only query for keys that were in the original set.  Querying for a new key will return a random value, silently.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NoKeyBoomHashMap2<K, D1, D2> {
    pub mphf: Mphf<K>,
    pub values: Vec<D1>,
    pub aux_values: Vec<D2>,
}

impl<K, D1, D2> NoKeyBoomHashMap2<K, D1, D2>
where
    K: Hash + Debug + PartialEq + Send + Sync,
    D1: Debug,
    D2: Debug,
{
    pub fn new_parallel(
        mut keys: Vec<K>,
        mut values: Vec<D1>,
        mut aux_values: Vec<D2>,
    ) -> NoKeyBoomHashMap2<K, D1, D2> {
        let mphf = Mphf::new_parallel(1.7, &keys, None);
        for i in 0..keys.len() {
            loop {
                let kmer_slot = mphf.hash(&keys[i]) as usize;
                if i == kmer_slot {
                    break;
                }
                keys.swap(i, kmer_slot);
                values.swap(i, kmer_slot);
                aux_values.swap(i, kmer_slot);
            }
        }
        NoKeyBoomHashMap2 {
            mphf,
            values,
            aux_values,
        }
    }

    pub fn new_with_mphf(
        mphf: Mphf<K>,
        values: Vec<D1>,
        aux_values: Vec<D2>,
    ) -> NoKeyBoomHashMap2<K, D1, D2> {
        NoKeyBoomHashMap2 {
            mphf,
            values,
            aux_values,
        }
    }

    /// Get the value associated with `key`, if available, otherwise return None
    pub fn get<Q: ?Sized>(&self, kmer: &Q) -> Option<(&D1, &D2)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let maybe_pos = self.mphf.try_hash(kmer);
        match maybe_pos {
            Some(pos) => Some((&self.values[pos as usize], &self.aux_values[pos as usize])),
            _ => None,
        }
    }
}
