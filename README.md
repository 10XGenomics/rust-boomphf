# Fast and Scalable Minimal Perfect Hash Functions in Rust

A Rust impl of [**Fast and scalable minimal perfect hashing for massive key sets**](https://arxiv.org/abs/1702.03154).

The library generates a minimal perfect hash functions (MPHF) for a collection of hashable objects. This algorithm generates MPHFs that consume ~3-6 bits/item.  The memory consumption during construction is a small multiple (< 2x) of the size of the dataset and final size of the MPHF. 
Note, minimal perfect hash functions only return a usable hash value for objects in the set used to create the MPHF. Hashing a new object will return an arbitrary hash value. If your use case may result in hashing new values, you will need an auxiliary scheme to detect this condition.

See [Docs](https://10xgenomics.github.io/rust-boomphf/)

Example usage:

 ```rust
 use boomphf::*;

 // sample set of obejcts
 let possible_objects = vec![1, 10, 1000, 23, 457, 856, 845, 124, 912];
 let n = possible_objects.len();

 // generate a minimal perfect hash function of these items
 let phf = Mphf::new(1.7, possible_objects.clone(), None);

 // Get hash value of all objects
 let mut hashes = Vec::new();
 for v in possible_objects {
     hashes.push(phf.hash(&v));
 }
 hashes.sort();

 // Expected hash output is set of all integers from 0..n
 let expected_hashes: Vec<u64> = (0 .. n as u64).collect();
 assert!(hashes == expected_hashes)
 ```

Note: this crate carries it's own bit-vector implementation to support rank-select queries and multi-threaded read-write access.
