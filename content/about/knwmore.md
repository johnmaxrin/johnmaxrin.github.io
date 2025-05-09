+++
title = "Research"
template = "post.html"
date = 2025-05-05
+++

### Distributed Heterogeneous Computing
I work on making it easier to build compilers for supercomputers. Today, if you are developing a domain-specific language (DSL) or a compiler, supporting distributed and heterogeneous systems usually means writing a lot of low-level code for synchronization, communication, and device-specific details.  

To address this, I designed a new **intermediate representation (IR)** called **DHIR**.  

- **DHIR is built on MLIR** and acts as a **middleware IR** between high-level dialects (like SCF, Affine etc) and low-level hardware IRs (such as LLVM IR).  
- It provides a structured way to represent **distributed heterogeneous execution**, so compiler developers don’t need to manually insert synchronization or communication.  
- The infrastructure automatically takes care of:  
  - Respecting **dependencies**  
  - Inserting **synchronization**  
  - Handling **data movement and communication primitives**  

This means DSL and compiler developers can focus on the **semantics of their language**, while DHIR handles the **system-level concerns** of mapping to distributed, heterogeneous machines.  

👉 You can find the code here: [DHIR Repository](https://github.com/johnmaxrin/avial)

## StarPlat MILR
Working on it ... 

## Debug Dialect
Working on it ... 