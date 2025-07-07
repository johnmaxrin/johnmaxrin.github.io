+++
title = "Intro to Polyhedral Analysis in MLIR - II"
description = "MLIR, Polyhedral Analysis"
date = 2025-07-25
draft = false
template = "post.html"
+++

# Understanding Memory Access Relations
In [Part 1](/posts/painmlir/) of this series, we introduced the fundamentals of the polyhedral model and how it fits into the MLIR framework. In this post, we dive into one of the most important components of that model: the memory access relation.

A memory access relation is a mathematical mapping from loop iterations to the memory locations accessed during those iterations.

You can think of it like this:

    AccessRel: [loop indices] → [array subscripts]

In the context of the polyhedral model:
- Domain: The iteration space of the loop
- Range: The memory space being accessed (array indices)

This relation is fundamental for detecting data dependences, validating transformations, and enabling optimizations.

# Example in MLIR
Let’s take a simple loop written in MLIR:
```c++
affine.for %i = 0 to 7 {
  %a = affine.load %A[%i + 1] : memref<?xf32>
  %sum = arith.addf %one, %a : f32
  affine.store %sum, %A[%i] : memref<?xf32>
}
```

Here’s what’s happening:
- We're reading from `A[%i + 1]`
- We’re writing to `A[%i]`
There are two memory access relations here: one for the load, and one for the store.

## Loop Constraints
These are the constraints on the loop iteration variable %i, extracted from the domain:
```bash
Loop Constraints
Domain: 0, Range: 1, Symbols: 0, Locals: 0
( ) -> ( Id<0x12ee37dd0> ) : [ ] 2 constraints
(Value  const)
  1   0 >= 0       ; i >= 0
 -1   6 >= 0       ; i <= 6  (since loop is i = 0 to 7 exclusive)
```
This defines the iteration domain: `0 ≤ i < 7`.

## Memory Access Relations
Let’s look at the access relations for both load and store.
Load: `A[%i + 1]`
```bash
memory access rel - load
Domain: 0, Range: 1, Symbols: 0, Locals: 1
( ) -> ( Id<0x12ee37dd0> ) : [ ] 3 constraints
(Value  Local   const)
  1   -1   1   = 0    ; m = i + 1
  1    0   0 >= 0     ; i >= 0
 -1    0   6 >= 0     ; i <= 6
```

`1 -1 1 = 0` means:
Memory index `m` = loop index `i + 1`
(Here, `m` is represented using a local variable)

Store: `A[%i]`
```bash
memory access rel - store
Domain: 0, Range: 1, Symbols: 0, Locals: 1
( ) -> ( Id<0x12ee37dd0> ) : [ ] 3 constraints
(Value  Local   const)
  1  -1   0  = 0     ; m = i
  1   0   0 >= 0     ; i >= 0
 -1   0   6 >= 0     ; i <= 6
```

This line:
`1 -1 0 = 0` means:
Memory index `m` = loop index `i`

MLIR provides utilities to extract these relations automatically from your code. 

```c++
mlir::affine::MemRefAccess src(loadOp);  
mlir::affine::MemRefAccess dst(storeOp);  

mlir::presburger::PresburgerSpace space = 
    mlir::presburger::PresburgerSpace::getRelationSpace();

mlir::presburger::IntegerRelation srcRel(space), dstRel(space);

// Extract memory access relations into IntegerRelation
if (failed(src.getAccessRelation(srcRel)) ||
    failed(dst.getAccessRelation(dstRel))) {
  return mlir::affine::DependenceResult::Failure;
}

affine::FlatAffineValueConstraints srcDomain(srcRel.getDomainSet());
affine::FlatAffineValueConstraints dstDomain(dstRel.getDomainSet());

srcDomain.dump();
dstDomain.dump();
```

With this setup, you can analyze, print, or even manipulate memory access relations directly from within an MLIR pass or analysis pipeline.

# Coming Next

In the next part of this series, we’ll explore how to use these access relations to perform dependence analysis that enables transformations like loop interchange, tiling, or fusion.

Stay tuned!