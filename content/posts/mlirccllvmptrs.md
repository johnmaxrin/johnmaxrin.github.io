+++
title = "Starplat, Graph, MLIR, LLVM & Pointers"
description = "MLIR, LLVM & Pointers"
date = 2025-03-28
draft = false
template = "post.html"
+++

I was working on lowering our DSL’s IR directly to LLVM IR! Yeah, a lot of high-level information had to be stripped away during lowering to LLVM, but I was convinced the adventure was worth it. When it came to representing our Graph Type in LLVM, the only option I could think of was using a struct and that’s where the real story begins.

# Starplat and Graph Type
Starplat is a DSL built for graph analytics, and having a type named Graph is an absolute must. Our Graph type isn’t just a simple container it does a ton of things: storing graphs in different representations, retrieving the total number of nodes and edges, and much more.

By default, we represent the graph in Compressed Sparse Row (CSR) format. We also have a set of associated functions, like attachNodeProperty. Think of Single Source Shortest Path, You need a "DISTANCE" value attached to every node. That’s exactly what attachNodeProperty does: it assigns node properties dynamically, making the graph more than just an abstract structure.
```cpp
g.attachNodeProperty(dist=INF);
```
Now, the real challenge? Representing all of this in LLVM. <br>
<strong>The Plan </strong> <br>
First, we need to understand how to deal with arrays in LLVM. We'll start small,  Creating a simple structure with a single int32 member. If we can successfully load and store values into this member, we’ll take the next step: replacing it with an array. Then, we scale up, Adding multiple members to the struct, manipulating values inside it, and ensuring everything works as expected.

Sounds simple enough, right? Well, let’s find out. 
## Struct with Single int32
In C, We can define a simple struct like this:
```c
struct 
{
    int a; 
};
```

In LLVM, this translates to:

```asm
%Graph = type {i32}
```
Now that we have the mapping, the next step is to <strong> emit a graph struct using MLIR.</strong>

```cpp
auto int32ty = builder.getI32Type();
auto structType = LLVM::LLVMStructType::getIdentified(&context, "Graph");
structType.setBody({LLVM::LLVMPointerType::get(&context), builder.getI32Type()}, false);
auto graphObj = buidler.create<LLVM::AllocaOp>(loc, ptrType, structType, one);
```
This will generate LLVM IR similar to the following.
```asm
%1 = llvm.allocate %0 x !llvm.struct<"Graph", (i32)> : (i32) -> !llvm.ptr
```




# Struct and Pointer
