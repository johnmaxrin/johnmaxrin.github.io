+++
title = "Distributed Heterogeneous IR"
description = "MLIR, LLVM, HPC"
date = 2025-11-03
draft = false
template = "post.html"
+++

# The Challenge
Modern computing is increasingly heterogeneous. Your laptop has a CPU and maybe a GPU. Data centers have CPUs, GPUs, TPUs, and specialized accelerators. High-performance computing clusters combine all of these across multiple nodes connected by high-speed networks.

Writing programs that efficiently utilize this heterogeneous hardware is incredibly challenging. You need to:
- Partition your computation across different devices
- Express parallelism within each device (threads, CUDA blocks, MPI ranks)
- Move data between devices efficiently
- Coordinate execution and handle dependencies

Traditionally, this means writing tangled code mixing MPI, OpenMP, CUDA, and custom communication logic. It's error-prone, hard to maintain, and difficult to port to new hardware configurations.

This is what DHIR is solving. DHIR (Distributed Heterogeneous Intermediate Representation) is an MLIR dialect for expressing distributed, heterogeneous computations as tasks that can execute across CPUs, GPUs, and distributed systems.

# Core Concepts
## Tasks
Tasks are the fundamental unit of computation in DHIR, analogous to functions in C/C++. Each task:

- Encapsulates a specific computation
- Has explicit inputs and outputs
- Can be scheduled for execution on different devices 
- Contains parallelism information

## Schedule
Just like the main function. 

## `system_config.json`
Helps in defining the distributed system that you have. As of now we have 
```
module attributes {avial.target_devices = [#dlti.target_device_spec<"type" = "node", "arch" = "x86_64", "cost" = 1.000000e+00 : f32, "node_id" = "node0", "gpu_count" = 0 : i32>, #dlti.target_device_spec<"type" = "node", "arch" = "x86_64", "cost" = 1.000000e+00 : f32, "node_id" = "node1", "gpu_count" = 0 : i32>, #dlti.target_device_spec<"type" = "node", "arch" = "x86_64", "cost" = 5.000000e-01 : f32, "node_id" = "node2", "gpu_count" = 1 : i32>]}
```


