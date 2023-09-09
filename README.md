## What is TinyVSU
TinyVSU is a tiny vector speed up library, which use AVX SIMD instruction set to speed up vector computation on CPU and use CUDA to speed up on GPU.

## Build from source
```
// clone third_party module
>>> git submodule update --init --recursive
>>> cmake -S . -B build
>>> cmake --build build
```