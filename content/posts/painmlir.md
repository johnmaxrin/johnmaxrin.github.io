+++
title = "Intro to Polyhedral Analysis in MLIR - I"
description = "MLIR, Polyhedral Analysis"
date = 2025-07-24
draft = false
template = "post.html"
+++


[**Polyhedral analysis**](http://polyhedral.info/) is a mathematical framework used to analyze and transform loop nests in programs—particularly those with static control parts (where loop bounds and memory accesses are affine expressions). This model represents iteration domains, loop bounds, and memory access patterns as systems of linear inequalities. These sets can then be used for powerful optimizations such as loop fusion, tiling, interchange, and parallelization.

# What It Means in Practice

Suppose you have a matrix multiplication function written in MLIR using the `affine.for` dialect with dynamic sizes for matrices:

```mlir
func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %M = memref.dim %A, %c0 : memref<?x?xf32>
  %N = memref.dim %B, %c1 : memref<?x?xf32>
  %K = memref.dim %A, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %M step 1 {
    affine.for %j = 0 to %N {
      affine.for %k = 0 to %K {
        %a = affine.load %A[%i, %k] : memref<?x?xf32>
        %b = affine.load %B[%k, %j] : memref<?x?xf32>
        %c = affine.load %C[%i, %j] : memref<?x?xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<?x?xf32>
      }
    }
  }
  return
}
```

This code uses affine.for loops with dynamic bounds computed at runtime (%M, %N, %K) via memref.dim. The loop nests iterate over all valid (i, j, k) indices for matrix multiplication, and use affine loads and stores to access memory in a way that can be modeled with the polyhedral model.

# Extracting the Polyhedral Model

MLIR provides the infrastructure to extract this iteration domain into a formal representation called a FlatAffineValueConstraints. This object stores the loop's constraints in a matrix form, representing:

- Domain dimensions (induction variables)
- Symbols (parameters/constants)
- A set of linear inequalities

You can extract this model using the getIndexSet function:

```mlir
affine::FlatAffineValueConstraints constraints; 
SmallVector<Operation *, 4> ops;

module->walk<mlir::WalkOrder::PreOrder>([&](mlir::affine::AffineForOp forOp) {
  ops.push_back(forOp);  
});

affine::getIndexSet(ops, &constraints);
constraints.dump();

```

This generates a model like:
```yaml
Domain: 0, Range: 3, Symbols: 3, Locals: 0
( ) -> ( i j k ) : [ M N K ]
6 constraints
(Value  Value   Value   Value   Value   Value   const)
  1  0  0  0  0  0  0 >= 0
 -1  0  0  1  0  0 -1 >= 0
  0  1  0  0  0  0  0 >= 0
  0 -1  0  0  1  0 -1 >= 0
  0  0  1  0  0  0  0 >= 0
  0  0 -1  0  0  1 -1 >= 0
```

## Interpreting the Output

This output captures the bounds of the loop nest. Here's how to interpret it:

- There are 3 loop variables (i, j, k) corresponding to %arg3, %arg4, and %arg5.
- There are 3 symbols (M, N, K) corresponding to the upper bounds 200, 300, and 400 (these can be parametric).
- The constraints describe the bounds:

```yaml
i >= 0
i <= M - 1
j >= 0
j <= N - 1
k >= 0
k <= K - 1
```
This forms a polyhedron in 3D space, bounded by the affine inequalities.

# Why This Matters

Once this model is constructed, the compiler can:
- Detect loop-carried dependencies
- Apply loop transformations legally
- Optimize data movement
- Parallelize independent loop iterations

All of this is grounded in formal, provable correctness through the polyhedral representation.

# Final Thoughts

MLIR brings polyhedral analysis closer to everyday compiler IR workflows. With affine dialect, and analysis utilities like FlatAffineValueConstraints, MLIR makes it possible to perform sophisticated optimizations automatically. This is especially powerful in domains like HPC, deep learning, and scientific computing.

If you're writing compilers or building optimizers, MLIR's polyhedral framework is not just a theory—it's a tool you can use today.

# References
- [ 2022 EuroLLVM Dev Mtg “Precise Polyhedral Analyses For MLIR using the FPL Presburger Library” ](https://www.youtube.com/watch?v=Xg4RfgPIT-Y)
- [ Polyhedral Analyses using MLIR's Affine dialect - Vinayaka Bandishti ](https://www.youtube.com/watch?v=fcBkr4mk-8A)
- [ Polyhedral Analyses using MLIR's Affine Dialect (contd.) ](https://www.youtube.com/watch?v=jBgU_EeUKYw)
- [ Compiler Design Module 128 : Introduction to the Polyhedral Framework ](https://www.youtube.com/watch?v=BKcFoP6B4Rw)





