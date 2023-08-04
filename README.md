## What is TinyVSU
TinyVSU is a tiny vector speed up library, which use AVX SIMD instruction set to speed up vector computation on CPU and use CUDA to speed up on GPU.

## Run tests
Make sure you have installed cmake!<br>
Run the following command to run tests.
```
>>> cmake -S . -B build
>>> cmake --build build
>>> cd build/test
>>> ./test
```