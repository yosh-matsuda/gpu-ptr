# GPU pointer wrapper for CUDA/HIP

Cross-platform GPU smart pointer with C++20 range support.

## Features

*   No more raw pointers, even for GPU programming!
*   Smart pointers for array and value with lifetime management.
*   Support for unified memory between host and device.
*   3 cross-platform support for CUDA and HIP (AMD / NVIDIA GPUs).
*   C++20 range adaption on both CPU and GPU.

## Requirements

*   CUDA >= 12 or HIP >= 6.0
*   C++20 host compiler
    *   LLVM >= 16
    *   GCC >= 12

## Examples

GPU array with trivial data type:

```cpp
#include <gpu_ptr.hpp>
using namespace gpu_ptr;

// Create integer array on GPU
auto arr = array<int>(5, 1);

// Call kernel function with array wrapper
template <typename T>
__global__ void func(array<T> arr)
{
    // Access elements with range-based for loop on GPU
    for (auto& e : arr)
    {
        e *= 2;
    }

    // Capture shared memory
    // NOTE: This has no effect on the CPU side
    extern __shared__ T sh[];
    auto shared_arr = array<int>(sh, 5);
}
func<<<1, 1, sizeof(int) * 5>>>(arr);

// Extract GPU array to range object on the CPU side
auto vec = arr.to<std::vector>();
// -> {2, 2, 2, 2, 2}
```

Support unified memory array:  
(non-trivial data type is also acceptable)

```cpp
#include <gpu_ptr.hpp>
using namespace gpu_ptr;

// Non-trivial data type
struct Data
{
    unified_array<int> ai = {1, 2, 3, 4, 5};
    unified_array<double> ad = {1.0, 2.0, 3.0, 4.0, 5.0};
};

// Create unified memory array
auto data_array = unified_array<Data>(5);

// Accessable on both CPU and GPU
for (auto& d : data_array)
{
    d.ai[0] += 10;
    d.ad[0] += 10.0;
}

// Prefetch data to GPU
data_array.prefetch_to_gpu();

// Call kernel function with array wrapper
__global__ void func(unified_array<Data> data_array)
{
    // Access elements with range-based for loop on GPU
    for (auto& d : data_array)
    {
        d.ai[0] -= 10;
        d.ad[0] -= 10.0;
    }
}
func<<<1, 1>>>(data_array);

// Prefetch data to CPU
data_array.prefetch_to_cpu();
```

Conversion between CPU and GPU arrays:

```cpp
#include <gpu_ptr.hpp>
using namespace gpu_ptr;

// Create zero-initialized integer GPU array
auto arr1 = array<int>(5);

// Create default-initialized integer GPU array (uninitialized)
auto arr1 = array<int>(default_init);

// Convert range object to GPU array
auto src2 = std::vector<double>(5, 1.0);
auto arr2 = array(src2);
// -> array<double>

// Initialize GPU array with initializer list
auto arr3 = array{1, 2, 3, 4, 5};
// -> array<int>

// Conversion from nested arrays
auto src4 = std::vector<std::vector<int>>(5, std::vector<int>(5, 1));
auto arr4 = unified_array(src4);
// -> unified_array<unified_array<int>>

// Convert GPU array to range object
auto vec = arr4.to<std::vector>();
// -> std::vector<std::vector<int>>
```

Value pointer wrapper for scalar value:

```cpp
#include <gpu_ptr.hpp>
using namespace gpu_ptr;

// Create integer value on GPU
auto val = value<int>{5};

// Copy value from GPU to CPU
auto v = *val;
// -> 5

// Create value on unified memory
auto uval = unified_value<int>{5};

// Read value from unified memory
auto uv = *uval;
// -> 5
```

## How to use

To integrate gpu_ptr into your CMake project, simply add the following:

```cmake
add_subdirectory(<PATH_TO_CLONE_DIR>/gpu_ptr ${CMAKE_CURRENT_BINARY_DIR}/gpu_ptr)
target_link_libraries(${PROJECT_NAME} PRIVATE gpu_ptr::gpu_ptr)
```

If you have installed gpu-ptr via CMake, `find_package` command is enabled:

```cmake
find_package(gpu_ptr CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE gpu_ptr::gpu_ptr)
```

### GPU platform selection

#### CUDA

By default, CUDA is used as the backend. The target architectures must be set to `CMAKE_CUDA_ARCHITECTURES`.

For example in `CMakeLists.txt`:

```cmake
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA supported")
endif()

# for V100, A100, and H100 GPUs
set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
enable_language(CUDA)
...
find_package(gpu_ptr CONFIG REQUIRED)   # or add_subdirectory
target_link_libraries(${PROJECT_NAME} PRIVATE gpu_ptr::gpu_ptr)
```

or CMake configure command:

```bash
$ cmake -DCMAKE_CUDA_ARCHITECTURES=70,80,90 ...
```

#### HIP

To use HIP instead, set `ENABLE_HIP` to `ON` and `CMAKE_HIP_PLATFORM` to `nvidia` or `amd`.

For AMD platform example in `CMakeLists.txt`:

```cmake
check_language(HIP)
if(NOT CMAKE_HIP_COMPILER)
  message(FATAL_ERROR "No HIP supported")
endif()

set(ENABLE_HIP ON)
set(CMAKE_HIP_PLATFORM amd) # or nvidia for NVIDIA GPU
set(CMAKE_HIP_ARCHITECTURES gfx942) # for MI300X GPU
enable_language(HIP)
...
find_package(gpu_ptr CONFIG REQUIRED)   # or add_subdirectory
target_link_libraries(${PROJECT_NAME} PRIVATE gpu_ptr::gpu_ptr)
```

or CMake configure command:

```bash
$ cmake -DENABLE_HIP=ON -DCMAKE_HIP_PLATFORM=amd -DCMAKE_HIP_ARCHITECTURES=gfx942 ...
```
