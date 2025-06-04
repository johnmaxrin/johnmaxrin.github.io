+++
title = "MLIR GPU-Dialect “Hello World” Tutorial"
description = "MLIR, LLVM & Pointers"
date = 2025-06-04
draft = false
template = "post.html"
+++

This tutorial shows how to write, compile, and run a simple “Hello World” program using MLIR’s GPU dialect targeting an NVIDIA GPU (CUDA backend). We assume MLIR is built with NVPTX support and a CUDA toolkit is installed. We focus on GPU-dialect specifics: writing the kernel IR, host integration, and launch configuration, and the MLIR compilation pipeline to generate a GPU executable.

## 1. Write the GPU Kernel in MLIR

The GPU dialect uses gpu.launch to invoke kernels on the GPU with a specified grid (blocks) and block (threads) configuration. Inside a gpu.launch region you write device code with GPU ops (like gpu.thread_id or gpu.printf) and terminate with gpu.terminator. For example, to print from the GPU, you can use a gpu.printf inside a 1×1×1 launch:

```asm
module {
  func.func @main() -> i32 {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c42_i32 = arith.constant 33 : i32

    gpu.launch 
      blocks(%arg0, %arg1, %arg2) in (%arg6 = %0, %arg7 = %0, %arg8 = %0) 
      threads(%arg3, %arg4, %arg5) in (%arg9 = %0, %arg10 = %0, %arg11 = %0) {
        gpu.printf "Hi There: "
        gpu.terminator
    }

    return %c42_i32 : i32
  }
}
```

Let’s walk through this MLIR code:

- `func.func @main()` defines the entry point of the program.

- `arith.constant` is used to define integer constants. In this case, `%c1_i32` is a constant value 1, and `%c42_i32` is the return value 42.

- `arith.index_cast` converts the i32 constant to an index type, which is typically used for loop bounds or launch parameters in MLIR.

- The `gpu.launch` operation launches a GPU kernel. It specifies the grid and block dimensions:

    - `blocks(...) in (...)`: Describes how many blocks will be launched.

    - `threads(...) in (...)`: Describes how many threads per block.

    Here, both blocks and threads are launched with 1 in each dimension (x, y, z), so it’s a minimal configuration.

- Inside the GPU kernel body, we simply use `gpu.printf` to print a string "**Hi There:** ".

- `gpu.terminator` marks the end of the GPU kernel region.

- The function returns the integer value `33`.

## 3 .The Transformation Pipeline
To transform a high-level GPU MLIR program into a low-level NVVM binary representation, we can use the following mlir-opt pipeline:
```sh
mlir-opt sample.mlir \
  --pass-pipeline="builtin.module(
    gpu-kernel-outlining,
    gpu.module(convert-gpu-to-nvvm),
    nvvm-attach-target{chip=sm_61},
    gpu.module(convert-gpu-to-nvvm),
    gpu-to-llvm,
    gpu-module-to-binary
  )" \
  -o example-nvvm.mlir

```

| Pass                              | Description                                                                               |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| `gpu-kernel-outlining`            | Extracts GPU kernel bodies into separate `gpu.module`s                                    |
| `gpu.module(convert-gpu-to-nvvm)` | Converts high-level GPU ops (e.g. `gpu.launch_func`) into NVVM dialect                    |
| `nvvm-attach-target{chip=sm_61}`  | Specifies the NVIDIA GPU target architecture (sm\_61 = Pascal, e.g., GTX 1080)            |
| `gpu-to-llvm`                     | Converts the host-side GPU ops into the LLVM dialect                                      |
| `gpu-module-to-binary`            | Compiles the NVVM kernel into a binary blob and embeds it in the module as a `gpu.binary` |

After the above transformations we get :
```asm
module attributes {gpu.container_module} {
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(42 : i32) : i32
    gpu.launch_func @main_kernel::@main_kernel blocks in (%0, %0, %0) threads in (%0, %0, %0) : i64 
    llvm.return %1 : i32
  }

  gpu.binary @main_kernel [#gpu.object<#nvvm.target<chip = "sm_61">, properties = {
    ISAToBinaryTimeInMs = 44 : i64,
    LLVMIRToISATimeInMs = 1 : i64
  }, "<binary blob>"] 
}
```

## 4. MLIR to LLVM IR
We use the mlir-translate tool to convert our MLIR NVVM dialect file into LLVM IR:
```sh
mlir-translate example-nvvm.mlir --mlir-to-llvmir  -o example.ll
```
This command generates the LLVM IR representation of our GPU kernel and host interaction code, saved in example.ll.

Here’s an excerpt of the generated LLVM IR:

```asm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

%0 = type {}

@main_kernel_binary = internal constant [2832 x i8] c" ... "  ; Embedded binary kernel blob

@main_kernel_module = internal global ptr null

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 123, ptr @main_kernel_load, ptr null }
]

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 123, ptr @main_kernel_unload, ptr null }
]

@main_kernel_main_kernel_name = private unnamed_addr constant [12 x i8] c"main_kernel\00"

define i32 @main() {
entry:
  %1 = alloca %0, align 8
  %2 = alloca ptr, i64 0, align 8
  %3 = load ptr, ptr @main_kernel_module, align 8
  %4 = call ptr @mgpuModuleGetFunction(ptr %3, ptr @main_kernel_main_kernel_name)
  %5 = call ptr @mgpuStreamCreate()
  call void @mgpuLaunchKernel(ptr %4, i64 1, i64 1, i64 1, i64 1, i64 1, i64 1, i32 0, ptr %5, ptr %2, ptr null, i64 0)
  call void @mgpuStreamSynchronize(ptr %5)
  call void @mgpuStreamDestroy(ptr %5)
  ret i32 42
}

define internal void @main_kernel_load() section ".text.startup" {
entry:
  %0 = call ptr @mgpuModuleLoad(ptr @main_kernel_binary, i64 2832)
  store ptr %0, ptr @main_kernel_module, align 8
  ret void
}

declare ptr @mgpuModuleLoad(ptr, i64)

define internal void @main_kernel_unload() section ".text.startup" {
entry:
  %0 = load ptr, ptr @main_kernel_module, align 8
  call void @mgpuModuleUnload(ptr %0)
  ret void
}

declare void @mgpuModuleUnload(ptr)

declare ptr @mgpuModuleGetFunction(ptr, ptr)

declare ptr @mgpuStreamCreate()

declare void @mgpuLaunchKernel(ptr, i64, i64, i64, i64, i64, i64, i32, ptr, ptr, ptr, i64)

declare void @mgpuStreamSynchronize(ptr)

declare void @mgpuStreamDestroy(ptr)

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```
- The kernel binary is embedded as a large internal constant `(@main_kernel_binary)`, representing the compiled GPU code.

- The `@main_kernel_module` global pointer stores the loaded module handle.

- Global constructors and destructors (`@llvm.global_ctors` and `@llvm.global_dtors`) ensure the kernel module is loaded and unloaded properly at program startup and shutdown.

- The `@main` function performs the following:

    - Loads the kernel module handle.

    - Retrieves the kernel function by name (main_kernel).

    - Creates a GPU stream.

    - Launches the kernel with specified grid and block dimensions.

    - Synchronizes and destroys the stream.

    - Returns an arbitrary integer (42).

- The loader and unloader functions handle module lifecycle via calls to external runtime functions (mgpuModuleLoad, mgpuModuleUnload, etc.).

This LLVM IR forms the basis for generating the final PTX assembly in the next compilation step.

## 5. Define the mgpuruntime Interface for MLIR GPU Execution
MLIR’s GPU dialect allows the lowering of GPU kernels, but to actually execute those kernels, you need a runtime implementation that can load modules, launch kernels, and manage CUDA streams. The following C++ code implements a minimal runtime using CUDA’s **Driver API**.

```cpp
#include <cuda.h>
#include <iostream>
#include <iomanip>

extern "C" void* mgpuModuleLoad(void* data, long long size) {
    CUmodule mod;
    CUdevice device;
    CUcontext ctx;


    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuInit failed with error code " << res << "\n";
        return nullptr;
    }

    // Get CUDA device
    res = cuDeviceGet(&device, 0);  // Use device 0
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuDeviceGet failed with error code " << res << "\n";
        return nullptr;
    }

    // Create context
    res = cuCtxCreate(&ctx, 0, device);
    if (res != CUDA_SUCCESS) {
        std::cerr << "cuCtxCreate failed with error code " << res << "\n";
        return nullptr;
    }

    //CUresult initRes = cuInit(0);  
    res = cuModuleLoadData(&mod, data);

    std::cout<<"RES: "<<res<<"\n";

    if (res != CUDA_SUCCESS) std::cerr << "cuModuleLoadData failed\n";
    return mod;
}

extern "C" void mgpuModuleUnload(void* mod) {
    cuModuleUnload((CUmodule)mod);
}

extern "C" void* mgpuModuleGetFunction(void* mod, const char* name) {
    CUfunction func;


    CUresult res = cuModuleGetFunction(&func, (CUmodule)mod, name);
    std::cout<<"Res "<<res <<"\n";
    
    if (res != CUDA_SUCCESS) std::cerr << "cuModuleGetFunction failed\n";
    return func;
}

extern "C" void* mgpuStreamCreate() {
    CUstream stream;
    cuStreamCreate(&stream, 0);
    return stream;
}

extern "C" void mgpuLaunchKernel(void* func,
                                 long long gx, long long gy, long long gz,
                                 long long bx, long long by, long long bz,
                                 int shared, void* stream,
                                 void* params, void*, long long) {
    void** param_array = (void**)params;
    cuLaunchKernel((CUfunction)func,
                   gx, gy, gz,
                   bx, by, bz,
                   shared, (CUstream)stream, param_array, nullptr);
}

extern "C" void mgpuStreamSynchronize(void* stream) {
    cuStreamSynchronize((CUstream)stream);
}

extern "C" void mgpuStreamDestroy(void* stream) {
    cuStreamDestroy((CUstream)stream);
}
```

This runtime layer allows the MLIR-compiled GPU code to be executed without needing external CUDA host code. You can compile and execute GPU kernels generated from MLIR entirely within a runtime like this, which makes it highly customizable and useful for research, prototyping, or building new compiler backends.

## 6. Compiling LLVM IR to Executable
After generating the LLVM IR (example.ll), we compile it together with the GPU runtime source (mgpuruntime.cc) using clang++. This produces a host executable that will load and launch our GPU kernel.
```sh
clang++  mgpuruntime.cc example.ll -O3 -lcuda -o run_kernel
```

## 7. Running the Executable
Finally, we run the compiled program:
```sh
./run_kernel
```
The output should be:
```
Hi There:
```