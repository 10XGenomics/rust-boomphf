use std::hash::Hash;

use crate::hashmap::BoomHashMap;
use rayon::iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

impl<'data, K, V> IntoParallelIterator for &'data BoomHashMap<K, V>
where
    K: Hash + Sync + 'data,
    V: Sync + 'data,
{
    type Item = (&'data K, &'data V);
    type Iter = Iter<'data, K, V>;

    fn into_par_iter(self) -> Self::Iter {
        Iter {
            keys: &self.keys,
            values: &self.values,
        }
    }
}

/// Parallel iterator over immutable items in a slice
#[derive(Debug)]
pub struct Iter<'data, K, V> {
    keys: &'data [K],
    values: &'data [V],
}

impl<'data, K, V> ParallelIterator for Iter<'data, K, V>
where
    K: Sync + 'data,
    V: Sync + 'data,
{
    type Item = (&'data K, &'data V);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.len())
    }
}

impl<'data, K, V> IndexedParallelIterator for Iter<'data, K, V>
where
    K: Sync + 'data,
    V: Sync + 'data,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        self.keys.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        callback.callback(IterProducer {
            keys: self.keys,
            values: self.values,
        })
    }
}

struct IterProducer<'data, K, V> {
    keys: &'data [K],
    values: &'data [V],
}

impl<'data, K, V> Producer for IterProducer<'data, K, V>
where
    K: Sync + 'data,
    V: Sync + 'data,
{
    type Item = (&'data K, &'data V);
    type IntoIter = KeyValIter<'data, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        KeyValIter {
            keys: self.keys,
            values: self.values,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let (left_keys, right_keys) = self.keys.split_at(index);
        let (left_vals, right_vals) = self.values.split_at(index);
        (
            IterProducer {
                keys: left_keys,
                values: left_vals,
            },
            IterProducer {
                keys: right_keys,
                values: right_vals,
            },
        )
    }
}

struct KeyValIter<'data, K, V> {
    keys: &'data [K],
    values: &'data [V],
}

impl<'data, K, V> Iterator for KeyValIter<'data, K, V> {
    type Item = (&'data K, &'data V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.keys.is_empty() {
            return None;
        }
        let item = (&self.keys[0], &self.values[0]);
        self.keys = &self.keys[1..];
        self.values = &self.values[1..];
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.keys.len(), Some(self.keys.len()))
    }
}

impl<'data, K, V> ExactSizeIterator for KeyValIter<'data, K, V> {}

impl<'data, K, V> DoubleEndedIterator for KeyValIter<'data, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.keys.is_empty() {
            return None;
        }
        let len = self.keys.len();
        let item = (&self.keys[len - 1], &self.values[len - 1]);
        self.keys = &self.keys[..len - 1];
        self.values = &self.values[..len - 1];
        Some(item)
    }
}
