+++
title = "Towards a Universal Compiler"
description = "My Learnings"
date = 2025-08-25
draft = false
template = "post.html"
+++

I always wanted to create a **universal compiler**, Something that could take **source code + a system ISA specification** and directly produce the **binary** that runs on hardware. Imagine describing your architecture, feeding it into the compiler, and immediately being able to run programs on it.  

In this blog, I’ll share some of my learnings on this journey, starting from one of the earliest serious attempts at a retargetable compiler: **Fraser’s lcc (1991)**.  

---
## Fraser’s Vision (1991)  
The idea of a “universal compiler” isn’t new. Back in 1991, **Christopher Fraser and David Hanson** built **lcc**, a C compiler that was:  
- **Fast** (both in compilation speed and generated code quality),  
- **Retargettable** (easy to support new CPUs),  
- **Small** (compact codebase, far smaller than GCC),  
- **High quality** (conformed to ANSI C, passed validation suites).  

Their secret was **separating the frontend and backend**:  

- The **frontend** was target-agnostic and did a handful of simple but effective optimizations:  
  - Constant folding  
  - Local common subexpression elimination  
  - Simulated register declarations  
  - Dense jump tables for switch statements  

- The **backend** was driven by a **rule-based DSL**.  
  Rules defined how DAG (Directed Acyclic Graph) nodes in the intermediate representation mapped to target instructions.  

Example of what a rule looked like (simplified):

```asm
add(r, s) → ADD r, s cost=1
```


These rules captured:  
- The **pattern of DAG operators** (e.g., `add`, `mul`, `mem[addr]`)  
- The **target instruction form**  
- **Costs/constraints**  

A **rule compiler** (written in Icon!) translated these rules into a **hard-coded instruction selector** (a tree-pattern matcher). The backend became a **monolithic C program**: fast, compact, and specific to the ISA. The same mechanism also supported **peephole optimizations**: rules that rewrote naïve sequences into more efficient ones.  

The frontend and backend communicated via a **tight interface**:  
- ~17 functions (`emit_prologue`, `emit_data`, …)  
- A 36-operator DAG language  

This **clean split** made retargeting remarkably easy. To port lcc, you only needed:  
1. A **configuration file** (data sizes, alignments, calling conventions),  
2. A few **backend support functions**,  
3. A **rule set** for code generation.  

Fraser reported that each new target (VAX, MIPS, SPARC, 68020) required only **377–522 lines** of code. That’s a crazy low effort compared to retargeting GCC.  

👉 You can read the original paper here: [A Retargetable Compiler for ANSI C (1991)](https://dl.acm.org/doi/pdf/10.1145/122616.122621).  

---

## Drawbacks of Fraser’s Approach  

While lcc was ahead of its time, it also had **serious limitations**. Many of these explain why modern infrastructures like LLVM and MLIR took different paths.  

1. **Limited Optimization**  
   - No global optimizations, no SSA, no dataflow analyses.  
   - Generated code was “good enough” but lagged behind vendor compilers and later GCC/LLVM.  

2. **Tiny IR (36-operator DAG)**  
   - Too low-level, closer to canonicalized trees than a flexible IR.  
   - Couldn’t represent advanced optimizations like loop transformations or vectorization.  

3. **Local Instruction Selection Only**  
   - Rules mapped one DAG at a time → no global instruction scheduling.  
   - Couldn’t handle complex idioms (e.g., `memcpy` → `rep movsb` on x86, delay slots on SPARC).  

4. **Shallow Retargetability**  
   - Worked well for clean RISC ISAs (MIPS, SPARC, VAX).  
   - Much harder for “irregular” ISAs like x86, DSPs, or GPUs.  

5. **Opaque Generated Backends**  
   - The rule compiler produced large monolithic C code.  
   - Fast, but hard to debug or customize compared to data-driven matchers.  

6. **Language Lock-In**  
   - Frontend was tightly coupled to ANSI C.  
   - Couldn’t serve as a general-purpose compiler infrastructure.  

7. **Didn’t Scale with Hardware Trends**  
   - Built before superscalar, SIMD, multicore, and GPUs became dominant.  
   - The tiny DAG IR couldn’t express parallelism or vector-level semantics.  

---

## Why It Still Matters  

Despite its limitations, Fraser’s approach was **visionary**.  
- The **declarative rule-based backend** foreshadowed LLVM’s **TableGen**, MLIR’s **ODS**, Cranelift’s **ISLE**, and GNU’s **CGEN**.  
- The **tight front–backend interface** showed that **retargeting can be simple** if the abstraction is chosen well.  
- The **focus on speed + smallness** influenced generations of lightweight compilers used in embedded systems and education.  

In short: Fraser’s lcc was a **proto-universal compiler**. It wasn’t powerful enough to scale into LLVM, but it proved the idea: *source code + machine spec → working compiler backend*.  

---

## My Takeaway  

As I think about building a universal compiler today, Fraser’s work reminds me:  
- Keep the **interfaces clean**.  
- Automate as much as possible (rule compilers, declarative ISAs).  
- Balance **simplicity vs optimization** — you can always add more sophisticated passes later.  

LLVM, MLIR, and even modern DSLs for ISAs (like [SLEIGH in Ghidra](https://ghidra.re/ghidra_docs/languages/html/sleigh.html) or [Sail for formal specs](https://github.com/riscv/sail-riscv)) are essentially “Fraser’s dream, at scale”.  

The journey towards a universal compiler isn’t finished, but lcc shows we’ve been chasing this dream for decades.  

---
