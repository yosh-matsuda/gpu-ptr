# gpu-ptr: Make GPU programming more modern C++ friendly

gpu-ptr is a header-only C++20 library that brings RAII and Range-based abstractions to GPU memory management and data layouts, enabling code safety and performance optimizations with zero overhead. By abstracting away raw pointers and memory layouts, gpu-ptr allows developers to focus on algorithm logic rather than resource bookkeeping.

Maximum GPU performance with Modern C++ syntax.

[![Tests](https://github.com/yosh-matsuda/gpu-ptr/actions/workflows/tests.yml/badge.svg)](https://github.com/yosh-matsuda/gpu-ptr/actions/workflows/tests.yml)

## Features

*   Smart pointer-like wrappers:
    *   Full RAII (Resource Acquisition Is Initialization) support for GPU memory management, ensuring automatic cleanup.
*   Performance-Oriented Memory Layouts:
    *   AoS to SoA Conversion: Converting Array-of-Structures (AoS) to Structure-of-Arrays (SoA) to ensure coalesced memory access for maximum GPU throughput. AoS stores data as contiguous structures, while SoA separates each field into its own array for better memory access patterns.
    *   Jagged Array Wrappers: Manage multi-dimensional data with varying row lengths using a single, efficient 1-D memory allocation and optimized for cache locality and performance.
*   C++20 Integration:
    *   Compatible with modern standards, including ranges and iterator concepts even for GPU kernel code.
    *   Range adapters for grid-stride access patterns (e.g., block-thread, grid-thread, grid-block, etc.).
*   Dual backend:
    *   Support for NVIDIA CUDA and AMD HIP.
*   Header-only library and no external dependencies.

### Requirements

Tested with the following environments:

*   CUDA 12.6.3 or later / HIP 6.2.4 or later
    *   🆖 CUDA 12.9.X: compiler segmentation faults
*   GCC 13 or later, Clang 16 or later

## Quick Start

### Installation

As a header-only library, you can simply copy the `include` directory to your project. If you are using CMake, you can add the following lines to your `CMakeLists.txt`:

```cmake
add_subdirectory(path/to/gpu-ptr)
target_link_libraries(your_target PRIVATE gpu_ptr::gpu_ptr)
```

Alternatively, you can use CMake's `FetchContent` module instead of manually downloading the library:

```cmake
include(FetchContent)
FetchContent_Declare(
    gpu_ptr
    GIT_REPOSITORY https://github.com/yosh-matsuda/gpu-ptr.git
    GIT_TAG v0.3.0
)
FetchContent_MakeAvailable(gpu_ptr)
target_link_libraries(your_target PRIVATE gpu_ptr::gpu_ptr)
```

### Example: Device memory management with smart pointers

gpu-ptr provides several smart pointer-like classes to manage GPU memory, including `array` and `managed_array` for arrays with range concepts, and `value` and `managed_value` for single value pointers.  
These classes automatically handle memory allocation and deallocation on the GPU. The `managed_` variants use unified memory, allowing seamless access from both host and device.

```cpp
#include <cooperative_groups.h>
#include <gpu_ptr.hpp>
#include <iostream>

using namespace gpu_ptr;

// Example kernel: initialize all elements
template <std::ranges::input_range T>
__global__ void kernel(T array)
{
    for (auto& v : array | views::grid_thread_stride)
        v += 1;
}

void example()
{
    // Allocate managed (or unmanaged) memory for 1024 integers
    auto array = managed_array<int>(1024);

    // Launch kernel to set values
    kernel<<<1, 128>>>(array);

    // Wrapper for cudaDeviceSynchronize/hipDeviceSynchronize
    api::gpuDeviceSynchronize();

    // Print results
    for (const auto& v: array) std::cout << v << " ";
}
```

gpu-ptr also provides safe initialization for memory allocation. For memory accessible only from the GPU (`array` and `value`), it checks at compile time that the type is safe for `memcpy` and initializes the allocated memory using the specified method. For Unified memory accessible from both the GPU and CPU (`managed_array` and `managed_value`), it constructs each element by calling its constructor.

### Example: Conversion from host to device memory and vice versa

Arrays and values classes can be easily converted from and to C++ containers (e.g., `std::vector`, `std::array`). The data is copied from host to device during construction.

```cpp
#include <gpu_ptr.hpp>
#include <vector>

using namespace gpu_ptr;

void example()
{
    // Create vector on host
    auto vec = std::vector<int>(100);
    for (auto i = 0; auto& v: vec) v = i++;

    // Convert from host vector to device array
    auto array = managed_array(vec);

    // Call kernel to perform operations on GPU
    // ...

    // Convert from device array to host vector
    vec = array.to<std::vector>();
}
```

### Example: Grid-stride range adapters

The kernel code can utilize C++20 range views for grid-stride access patterns (so-called grid-stride loop). gpu-ptr provides several [Range Adapter Closure Object](https://en.cppreference.com/w/cpp/named_req/RangeAdaptorClosureObject.html) such as `views::block_thread_stride`, `views::grid_thread_stride`, and `views::grid_block_stride` to facilitate this without any overhead. The following example demonstrates how to achieve memory coalescing when initializing nested arrays using grid-stride access.

```cpp
#include <gpu_ptr.hpp>
#include <iostream>
#include <vector>

using namespace gpu_ptr;

// Example kernel: initialize nested array
template <std::ranges::input_range T>
requires std::ranges::input_range<std::ranges::range_value_t<T>>
__global__ void kernel_example(T array)
{
    for (auto& a : array | views::grid_block_stride)
    {
        for (auto& v : a | views::block_thread_stride)
        {
            v *= 2;
        }
    }

    /* The above code is equivalent to the following:
    using namespace cooperative_groups;
    const auto grid = this_grid();
    const auto block = this_thread_block();
    for (auto i = grid.block_rank(); i < array.size(); i += grid.num_blocks())
    {
        for (auto j = block.thread_rank(); j < array[i].size(); j += block.size())
        {
            array[i][j] *= 2;
        }
    }
    */
}

void example()
{
    // Create nested vector on host
    auto vec_vec = std::vector(256, std::vector<int>(1024, 1));

    // Convert from nested host vector to nested device array
    auto nested_array = managed_array(vec_vec);

    // Launch kernel to initialize nested array
    kernel_example<<<32, 256>>>(nested_array);
    api::gpuDeviceSynchronize();

    // Print results
    for (const auto& inner_array : nested_array)
    {
        for (const auto& v : inner_array) std::cout << v << " ";
        std::cout << std::endl;
    }
}
```

### Example: AoS and SoA

gpu-ptr supports both Array of structures (AoS) and Structure of arrays (SoA) for memory layout optimization via `array` and `structure_of_arrays` classes, respectively. The memory layout comparison between `array` (AoS) and `structure_of_arrays` (SoA) is as follows:

![array of structure vs. structure_of_arrays](https://github.com/user-attachments/assets/219085eb-80c7-44e5-9e3b-6607bd8174bf)

 In either case, gpu-ptr provides a structure retrieval interface via array indices. Thus, `structure_of_arrays<tuple-derived>` can be used as a drop-in replacement for `array<tuple-derived>` with optimizing memory layout without altering your algorithm's implementation.

```cpp
#include <gpu_ptr.hpp>
#include <tuple>
#include <vector>

using namespace gpu_ptr;

// std::tuple or its derived struct can be used as structure type
// The below example shows a tuple-derived struct with three members and their accessors
template <typename... Ts>
requires (sizeof...(Ts) == 3)
struct CustomTuple : public std::tuple<Ts...>
{
    using std::tuple<Ts...>::tuple;
    __host__ __device__ auto& get_a() { return std::get<0>(*this); }
    __host__ __device__ auto& get_b() { return std::get<1>(*this); }
    __host__ __device__ auto& get_c() { return std::get<2>(*this); }
};
using Struct = CustomTuple<int, float, double>;

// Example kernel: process both AoS and SoA
template <std::ranges::input_range T>
__global__ void kernel_example(T array)
{
    for (auto&& v : array | views::grid_thread_stride)
    {
        // Access structure members for both AoS and SoA
        v.get_a() *= 2;
        v.get_b() *= 2.0f;
        v.get_c() *= 2.0;
    }
}

void example()
{
    // Create vector of structures
    auto vec = std::vector<Struct>(100, {1, 2.0f, 3.0});

    // Array of structures (AoS): single array for entire structure
    auto aos = managed_array<Struct>(vec);
    kernel_example<<<1, 32>>>(aos);

    // Structure of arrays (SoA): multiple arrays for each member internally
    auto soa = managed_structure_of_arrays<Struct>(vec);
    kernel_example<<<1, 32>>>(soa);
}
```

> [!TIP]
> Which is better, AoS or SoA? It depends on the access pattern to the structure members within a warp, the size of the structure, and the number of available registers. If all threads in a warp access the same member of the structure, SoA is generally better for maximizing coalesced memory access. However, if each thread accesses entire structures or different members, AoS may be more efficient due to better cache locality. Benchmarking both layouts with representative workloads is recommended to determine the optimal choice for your specific use case. Whichever, **gpu-ptr makes it easy to switch between AoS and SoA without changing the access interface**.

### Example: Jagged array

gpu-ptr provides `jagged_array` class to manage multi-dimensional array with varying row lengths, using a **single memory allocation to maximize coalescing access**.
This behaves like a wrapper for `managed_array` or `managed_jagged_array` with multi-dimensional indexing. The `jagged_array` is constructed from a 1-D array with sizes or multi-dimensional container (e.g., `std::vector<std::vector<T>>`).

The logical and physical data layout of `jagged_array` is as follows:

![data layout of jagged array](https://github.com/user-attachments/assets/7773537d-7259-4d2c-a695-8572906a6057)

```cpp
#include <gpu_ptr.hpp>
#include <iostream>
#include <vector>

using namespace gpu_ptr;

// Example kernel: modify all elements
template <std::ranges::input_range T>
__global__ void kernel(T array)
{
    for (auto& v : array | views::grid_thread_stride)
        v *= 2;
}

auto vec = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
auto vec_vec = std::vector<std::vector<int>>{{0}, {1, 2}, {3, 4, 5}, {6, 7, 8, 9}};

void example()
{
    // Create jagged array from nested std::vector
    auto jarray = jagged_array<managed_array<int>>(vec_vec);
    // Equivalent to the above:
    // auto jarray = jagged_array<managed_array<int>>({1, 2, 3, 4}, vec);

    // Launch kernel to re-set values
    kernel<<<1, 32>>>(jarray);
    api::gpuDeviceSynchronize();

    // Access each row and each element
    for (std::size_t i = 0; i < jarray.num_rows(); ++i)
    {
        for (std::size_t j = 0; j < jarray.size(i); ++j)
        {
            std::cout << jarray[{i, j}] << " ";
        }
        std::cout << std::endl;
    }
}
```

### Zero-overhead

gpu-ptr is designed to have zero-overhead compared to traditional raw pointer usage. The following example shows equivalent kernels using raw pointers and `managed_array` as verified by PTX analysis. The generated PTX assembly code for both kernels is identical, confirming that there is no performance penalty when using gpu-ptr abstractions with range adapters.

```cpp
// Traditional raw pointer kernel
__global__ void func0(int* data, std::uint32_t size)
{
    const auto block = cooperative_groups::this_thread_block();
    for (std::uint32_t i = block.thread_rank(); i < size; i += block.size())
    {
        data[i] = 1;
    }
}

// Using managed_array
__global__ void func1(managed_array<int, std::uint32_t> arr)
{
    const auto block = cooperative_groups::this_thread_block();
    for (std::uint32_t i = block.thread_rank(); i < arr.size(); i += block.size())
    {
        arr[i] = 1;
    }
}

// Using managed_array with range adapters
__global__ void func3(managed_array<int, std::uint32_t> arr)
{
    for (auto& v : arr | views::block_thread_stride)
    {
        v = 1;
    }
}
```

<details>

<summary>The identical PTX assembly code (except parameters) generated for the above three kernels is as follows:</summary>

```text
// Function Definition: _Z5func3... (Mangled C++ name for a template function)
// It accepts a 'managed_array' struct by value, which is 24 bytes in size.
.visible .entry _Z5func3IN7gpu_ptr13managed_arrayIijEEEvT_(
    .param .align 8 .b8 _Z5func3IN7gpu_ptr13managed_arrayIijEEEvT__param_0[24]
)
{
    // Register Declarations
    .reg .pred  %p<3>;   // Predicate registers for logic
    .reg .b32  %r<17>;  // 32-bit integer registers
    .reg .b64  %rd<6>;  // 64-bit registers for pointers/addresses

    // --- Extracting values from the Struct Parameter ---
    // The struct is laid out in memory:
    // Offset 0: Array Size (32-bit)
    // Offset 8: Base Pointer (64-bit)
    ld.param.u64  %rd1, [_Z5func3IN7gpu_ptr13managed_arrayIijEEEvT__param_0+8]; // Load data pointer
    ld.param.u32  %r5, [_Z5func3IN7gpu_ptr13managed_arrayIijEEEvT__param_0];    // Load array size N

    // --- Calculate 1D Thread Index (%r16) ---
    // Formula: idx = (tid.z * ntid.y + tid.y) * ntid.x + tid.x
    mov.u32  %r2, %ntid.y;              // %r2 = blockDim.y
    mov.u32  %r9, %tid.z;               // %r9 = threadIdx.z
    mov.u32  %r10, %tid.y;              // %r10 = threadIdx.y
    mad.lo.s32  %r11, %r2, %r9, %r10;   // %r11 = (blockDim.y * threadIdx.z) + threadIdx.y

    mov.u32  %r3, %ntid.x;              // %r3 = blockDim.x
    mov.u32  %r12, %tid.x;              // %r12 = threadIdx.x
    mad.lo.s32  %r16, %r11, %r3, %r12;  // %r16 = (%r11 * blockDim.x) + threadIdx.x

    // --- Initial Boundary Check ---
    setp.ge.u32  %p1, %r16, %r5;        // if (idx >= N)
    @%p1 bra  $L__BB2_3;                // Exit if initial index is out of bounds

    // --- Calculate Stride (Total Threads in Block) ---
    // stride = blockDim.x * blockDim.y * blockDim.z
    mul.lo.s32  %r13, %r3, %r2;         // %r13 = blockDim.x * blockDim.y
    mov.u32  %r14, %ntid.z;             // %r14 = blockDim.z
    mul.lo.s32  %r6, %r13, %r14;        // %r6 = total threads (stride)

    // Convert generic pointer to global memory pointer
    cvta.to.global.u64  %rd3, %rd1;     // %rd3 = global(data)

// --- Main Loop: Grid-Stride Writing ---
$L__BB2_2:
    // Memory address calculation: addr = base + (idx * 4)
    mul.wide.u32  %rd4, %r16, 4;        // Calculate 64-bit byte offset
    add.s64  %rd5, %rd3, %rd4;          // Add offset to base pointer

    // Store integer 1 into memory
    mov.u32  %r15, 1;                   // Value = 1
    st.global.u32  [%rd5], %r15;        // data[idx] = 1;

    // Increment index by stride
    add.s32  %r16, %r16, %r6;           // idx += stride;

    // Loop Condition
    setp.lt.u32  %p2, %r16, %r5;        // if (idx < N)
    @%p2 bra  $L__BB2_2;                // Repeat if within bounds

$L__BB2_3:
    ret;                                // Kernel end
}
```

</details>

### Tips

To reduce the number of registers used by the kernel, consider setting `size_type` to `std::uint32_t` instead of `default_size_type (= std::size_t)` when declaring GPU pointer types. For example, use `managed_array<T, std::uint32_t>` when the number of elements is less than 2<sup>32</sup>. To change `default_size_type` to `std::uint32_t`, define the `GPU_USE_32BIT_SIZE_TYPE_DEFAULT` macro before including `gpu_ptr.hpp`.

### Backends selection

Define `ENABLE_HIP` macro to use HIP backend. By default, CUDA backend is used. You can define this in your CMakeLists.txt or compiler flags.

## Reference

### `array` / `managed_array`

```cpp
template <typename T, typename size_type = size_type_default>
requires std::is_trivially_copyable_v<T>
class array;

template <typename T, typename size_type = size_type_default>
class managed_array;
```

The `array` and `managed_array` classes provide smart pointer-like wrappers for managing arrays on the GPU. They support C++20 ranges and iterator concepts, allowing seamless integration with modern C++ code and exporting to/from range-based containers.  The managed variant uses unified memory for easy access from both host and device. The non-managed variant allocates memory on the device using `cudaMalloc/hipMalloc` and `cudaMemcpy/hipMemcpy` for data transfer, which requires the type `T` to be trivially copyable for safety.

#### Constructors

```cpp
// (1) default constructor
array();
managed_array();

// (2) copy/move constructors
__host__ __device__ array(const array& other);
__host__ __device__ array(array&& other) noexcept;
__host__ __device__ managed_array(const managed_array& other);
__host__ __device__ managed_array(managed_array&& other) noexcept;

// (3) construct with size
__host__ explicit array(std::size_t n);
__host__ array(std::size_t n, const T& init_value);
__host__ array(std::size_t n, default_init_tag default_init);
__host__ explicit managed_array(std::size_t n);
__host__ managed_array(std::size_t n, const T& init_value);
__host__ managed_array(std::size_t n, default_init_tag default_init);

// (4) construct from range (e.g., std::vector, std::array)
template <std::ranges::input_range Range>
__host__ explicit array(const Range& range);
template <std::ranges::input_range Range>
__host__ explicit managed_array(const Range& range);
__host__ array(std::initializer_list<T> list);
__host__ managed_array(std::initializer_list<T> list);

// (5) construct from raw pointer (device pointer)
__host__ array(T* device_ptr, std::size_t n);
__device__ array(T* device_ptr, size_type n);
__host__ managed_array(T* device_ptr, std::size_t n);
__device__ managed_array(T* device_ptr, size_type n);
```

Where:

1.  Default constructor creates an empty array with null pointer.
2.  Copy and move constructors for copying pointer and size.
3.  Constructors with size allocate memory on the GPU. Optionally, an initial value or [default initialization](https://en.cppreference.com/w/cpp/language/default_initialization.html).
4.  Constructors from ranges copy data from host containers to device memory. Copying from `array` and `managed_array` types is not allowed to avoid unintended device-to-device copies. Use `to<>()` method for explicit device-to-device copy instead.
5.  Constructors from raw device pointers wrap existing device memory.

For nested ranges, nested `managed_array` is deduced: `std::vector<std::vector<T>> -> managed_array<managed_array<T>>`.

#### Exporters

```cpp
// (1) Copy data to host container
template <typename Container>
__host__ Container to() const;
template <template <typename...> typename Container>
__host__ Container<T> to() const;
template <template <typename...> typename Container>
__host__ Container<Container<...>> to() const; // nested ranges deduction only for managed_array

// (2) Copy data to gpu-ptr array
template <typename U>
__host__ array<U> to<array<U>>() const;
template <typename U>
__host__ managed_array<U> to<managed_array<U>>() const;

// (3) Static cast to host container
template <typename Container>
__host__ explicit operator Container() const;
```

Where:

1.  `to<Constainer>()` copies data from device to host container (e.g., `std::vector<T>`, `std::list<T>`). Range value type can be deduced automatically (e.g., `to<std::vector>()`). For nested ranges, nested containers are deduced only for `managed_array`, (e.g., `managed_array<managed_array<U>>::to<std::vector> -> std::vector<std::vector<U>>`).
2.  `to<array<U>>()` and `to<managed_array<U>>()` copy data from device array to another gpu-ptr array type with element type `U`.
3.  Explicit conversion operator to host container, equivalent to `to<Container>()`, but conversion to gpu-array types are not supported.

#### Range interface

Member types:

```cpp
array::size_type;
array::value_type;
array::reference;
array::const_reference;
array::iterator;
array::const_iterator;
array::pointer;
array::const_pointer;

managed_array::size_type;
managed_array::value_type;
managed_array::reference;
managed_array::const_reference;
managed_array::iterator;
managed_array::const_iterator;
managed_array::pointer;
managed_array::const_pointer;
```

Member functions:

```cpp
__host__ __device__ reference operator[](size_type index) noexcept;
__host__ __device__ const_reference operator[](size_type index) const noexcept;
__host__ __device__ iterator begin() noexcept;
__host__ __device__ const_iterator begin() const noexcept;
__host__ __device__ iterator end() noexcept;
__host__ __device__ const_iterator end() const noexcept;
__host__ __device__ const_iterator cbegin() const noexcept;
__host__ __device__ const_iterator cend() const noexcept;
__host__ __device__ std::reverse_iterator<iterator> rbegin() noexcept;
__host__ __device__ std::reverse_iterator<const_iterator> rbegin() const noexcept;
__host__ __device__ std::reverse_iterator<iterator> rend() noexcept;
__host__ __device__ std::reverse_iterator<const_iterator> rend() const noexcept;
__host__ __device__ reference front() noexcept;
__host__ __device__ const_reference front() const noexcept;
__host__ __device__ reference back() noexcept;
__host__ __device__ const_reference back() const noexcept;
__host__ __device__ pointer data() noexcept;
__host__ __device__ const_pointer data() const noexcept;
__host__ __device__ size_type size() const noexcept;
__host__ __device__ bool empty() const noexcept;
```

Concepts:

```cpp
std::ranges::range<array<T>>;
std::ranges::borrowed_range<array<T>>;  // only for device code
std::ranges::view<array<T>>;
std::ranges::output_range<array<T>, T>;
std::ranges::input_range<array<T>>;
std::ranges::forward_range<array<T>>;
std::ranges::bidirectional_range<array<T>>;
std::ranges::random_access_range<array<T>>;
std::ranges::sized_range<array<T>>;
std::ranges::contiguous_range<array<T>>;
std::ranges::common_range<array<T>>;
std::ranges::viewable_range<array<T>>;

std::ranges::range<managed_array<T>>;
std::ranges::borrowed_range<managed_array<T>>;  // only for device code
std::ranges::view<managed_array<T>>;
std::ranges::output_range<managed_array<T>, T>;
std::ranges::input_range<managed_array<T>>;
std::ranges::forward_range<managed_array<T>>;
std::ranges::bidirectional_range<managed_array<T>>;
std::ranges::random_access_range<managed_array<T>>;
std::ranges::sized_range<managed_array<T>>;
std::ranges::contiguous_range<managed_array<T>>;
std::ranges::common_range<managed_array<T>>;
std::ranges::viewable_range<managed_array<T>>;
```

#### Smart pointer interface

```cpp
// (1) Reset pointer and size
__host__ __device__ void reset();
__host__ void reset(T* device_ptr, std::size_t n);
__device__ void reset(T* device_ptr, size_type n);

// (2) Boolean conversion
__host__ __device__ explicit operator bool() const noexcept;

// (3) Use count
__host__ std::uint32_t use_count() const noexcept;
```

Where:

1.  If host code calls `reset(...)`, the current device memory is freed and set new device pointer and size. If device code calls `reset(...)`, it only sets the internal pointer and size without freeing memory.
2.  Bool conversion operator to check if the internal pointer is not null.
3.  Returns the current use count of the internal pointer. Note that this is only valid in host code.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer and size on the device side.

#### Memory management

Note: Memory management functions are only available for `managed_array` since they use unified memory.

```cpp
// (1) Prefetch
__host__ void prefetch(size_type start_idx, size_type len, int device_id = current_device_id, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(int device_id = current_device_id, api::gpuStream_t stream = 0, bool recursive = true) const;

// (2) Prefetch to host memory
__host__ void prefetch_to_cpu(size_type start_idx, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const;

// (3) Memory advice
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, int device_id = current_device_id, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id = current_device_id, bool recursive = true) const;

// (4) Memory advice to host memory
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const;
```

Where:

1.  Wrapper for `cudaMemPrefetchAsync/hipMemPrefetchAsync` to prefetch unified memory to specified device. The former overload prefetches a memory range, while the latter overload prefetches the entire memory. If `recursive` is true and the value type of the array has `prefetch(...)` function, prefetch is called recursively for nested or member arrays.
2.  Host memory prefetching overloads with similar behavior to (1).
3.  Wrapper for `cudaMemAdvise/hipMemAdvise` to set memory advice for unified memory. The former overload sets advice for a memory range, while the latter overload sets advice for the entire memory. If `recursive` is true and the value type of the array has `mem_advise(...)` function, mem_advise is called recursively for nested or member arrays.
4.  Host memory advice overloads with similar behavior to (3).

### `value` / `managed_value`

```cpp
template <typename T>
requires std::is_trivially_copyable_v<T>
class value;

template <typename T>
class managed_value;
```

The `value` and `managed_value` classes provide smart pointer-like wrappers for managing single values on the GPU. They allow seamless integration and exporting to/from host values. The managed variant uses unified memory for easy access from both host and device. The non-managed variant allocates memory on the device using `cudaMalloc/hipMalloc` and `cudaMemcpy/hipMemcpy` for data transfer, which requires the type `T` to be trivially copyable for safety.

#### Constructors

```cpp
// (1) default constructor
value();
managed_value();

// (2) copy/move constructors
__host__ __device__ value(const value& other);
__host__ __device__ value(value&& other) noexcept;
__host__ __device__ managed_value(const managed_value& other);
__host__ __device__ managed_value(managed_value&& other) noexcept;

// (3) construct with initial value
__host__ explicit value(const T& init_value);
__host__ explicit managed_value(const T& init_value);
__host__ explicit value(default_init_tag default_init);
__host__ explicit managed_value(default_init_tag default_init);

// (4) Construct the element in-place by arguments
template <typename... Args>
__host__ explicit value(Args&&... args);
template <typename... Args>
__host__ explicit managed_value(Args&&... args);

// (5) construct from raw pointer (device pointer)
__host__ __device__ array(T* device_ptr);
__host__ __device__ managed_array(T* device_ptr);
```

Where:

1.  Default constructor creates an empty value with null pointer.
2.  Copy and move constructors for copying pointer.
3.  Constructors with initial value or [default initialization](https://en.cppreference.com/w/cpp/language/default_initialization.html).
4.  Constructors that forward arguments to construct the element in-place. The arguments are perfectly forwarded to the constructor of `T`.
5.  Constructors from raw device pointers wrap existing device memory.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer and size on the device side.

#### Smart pointer interface

Member types:

```cpp
value::element_type;
managed_value::element_type;
```

Member functions:

```cpp
// (1) Operators for `value`
__device__ T& operator*() const noexcept;
__host__ T operator*() const;
__device__ T* operator->() const noexcept;
__host__ proxy_object operator->() const;

// (2) Operators for `managed_value`
__host__ __device__ T& operator*() const noexcept;
__host__ __device__ T* operator->() const noexcept;

// (3) Get raw pointer
__host__ __device__ T* get() const noexcept;

// (4) Reset pointer
__host__ __device__ void reset(T* device_ptr = nullptr);

// (5) Boolean conversion
__host__ __device__ explicit operator bool() const noexcept;

// (6) Use count
__host__ std::uint32_t use_count() const noexcept;
```

Where:

1.  Dereference and member access operators for `value`. Note that dereference operator is only available in device code, while member access operator returns a proxy object in host code to access copy of the value.
2.  Dereference and member access operators for `managed_value`, available in both host and device code.
3.  Get the raw device pointer.
4.  If host code calls `reset(...)`, the current device memory is freed and set new device pointer. If device code calls `reset(...)`, it only sets the internal pointer without freeing memory.
5.  Bool conversion operator to check if the internal pointer is not null.
6.  Returns the current use count of the internal pointer. Note that this is only valid in host code.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer on the device side.

#### Memory management

Note: Memory management functions are only available for `managed_value` since they use unified memory.

```cpp
// (1) Prefetch
__host__ void prefetch(int device_id = current_device_id, api::gpuStream_t stream = 0, bool recursive = true) const;

// (2) Prefetch to host memory
__host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const;

// (3) Memory advice
__host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id = current_device_id, bool recursive = true) const;

// (4) Memory advice to host memory
__host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const;
```

Where:

1.  Wrapper for `cudaMemPrefetchAsync/hipMemPrefetchAsync` to prefetch unified memory to specified device. If `recursive` is true and the value type has `prefetch(...)` function, prefetch is called recursively for member arrays.
2.  Host memory prefetching overload with similar behavior to (1).
3.  Wrapper for `cudaMemAdvise/hipMemAdvise` to set memory advice for unified memory. If `recursive` is true and the value type has `mem_advise(...)` function, mem_advise is called recursively for member arrays.
4.  Host memory advice overload with similar behavior to (3).

### `structure_of_arrays` / `managed_structure_of_arrays`

```cpp
template <typename... Ts>
using structure_of_arrays<Ts...> = structure_of_arrays<std::tuple<Ts...>>;
template <template <typename...> typename Tuple = std::tuple, typename... Ts, typename SizeType = size_type_default>
class structure_of_arrays<Tuple<Ts...>, SizeType>;


template <typename... Ts>
using managed_structure_of_arrays<Ts...> = managed_structure_of_arrays<std::tuple<Ts...>>;
template <template <typename...> typename Tuple = std::tuple, typename... Ts, typename SizeType = size_type_default>
class managed_structure_of_arrays<Tuple<Ts...>, SizeType>;
```

The `structure_of_arrays` and `managed_structure_of_arrays` classes provide smart pointer-like wrappers for managing Structure-of-Arrays (SoA) memory layout on the GPU. They allow for optimized memory access patterns by storing each member of a structure in separate contiguous arrays. The index access interface allows retrieval of the entire structure at a given index. This class is useful for maximizing coalesced memory access on GPUs. These classes support C++20 ranges and iterator concepts.

The value type of `structure_of_arrays` must be tuple-derived template class that inherits from `std::tuple<Ts...>` or is itself. The example definition of such tuple-derived class is as follows:

```cpp
template <typename... Ts>
requires (sizeof...(Ts) == 3)
struct CustomTuple : public std::tuple<Ts...>
{
    using std::tuple<Ts...>::tuple;
    __host__ __device__ auto& get_a() { return std::get<0>(*this); }
    __host__ __device__ auto& get_b() { return std::get<1>(*this); }
    __host__ __device__ auto& get_c() { return std::get<2>(*this); }
};
```

The template parameters `Ts...` correspond to the member types of the tuple-derived class. All parameters must be value types (i.e., not reference types), since the members are stored in separate arrays and returns by tuple of reference types of each element when accessed.

#### Constructors

```cpp
// (1) default constructor
structure_of_arrays();
managed_structure_of_arrays();

// (2) copy/move constructors
__host__ __device__ structure_of_arrays(const structure_of_arrays& other);
__host__ __device__ structure_of_arrays(structure_of_arrays&& other) noexcept;
__host__ __device__ managed_structure_of_arrays(const managed_structure_of_arrays& other);
__host__ __device__ managed_structure_of_arrays(managed_structure_of_arrays&& other) noexcept

// (3) construct with size
__host__ explicit structure_of_arrays(std::size_t n);
__host__ structure_of_arrays(std::size_t n, const Tuple<Ts...>& init_value);
__host__ structure_of_arrays(std::size_t n, default_init_tag default_init);
__host__ explicit managed_structure_of_arrays(std::size_t n);
__host__ managed_structure_of_arrays(std::size_t n, const Tuple<Ts...>& init_value);
__host__ managed_structure_of_arrays(std::size_t n, default_init_tag default_init);

// (4) construct from range of tuple-derived class
template <std::ranges::input_range Range>
__host__ explicit structure_of_arrays(const Range& range);
template <std::ranges::input_range Range>
__host__ explicit managed_structure_of_arrays(const Range& range);
__host__ structure_of_arrays(std::initializer_list<Tuple<Ts...>> list);
__host__ managed_structure_of_arrays(std::initializer_list<Tuple<Ts...>> list);

// (5) construct from multiple ranges
template <std::ranges::input_range... Ranges>
__host__ explicit structure_of_arrays(const Ranges& ranges...);
__host__ explicit structure_of_arrays(std::initializer_list<Ts>... lists);
template <std::ranges::input_range... Ranges>
__host__ explicit managed_structure_of_arrays(const Ranges& ranges...);
__host__ explicit managed_structure_of_arrays(std::initializer_list<Ts>... lists);
```

Where:

1.  Default constructor creates an empty structure_of_arrays with null pointers.
2.  Copy and move constructors for copying pointers and size.
3.  Constructors with size allocate memory on the GPU. Optionally, an initial value or [default initialization](https://en.cppreference.com/w/cpp/language/default_initialization.html).
4.  Constructors from ranges of tuple-derived class copy data from host containers to device memory. Copying from `structure_of_arrays` and `managed_structure_of_arrays` types is not allowed to avoid unintended device-to-device copies.
5.  Constructors from multiple ranges copy data from each host container to corresponding member arrays on the device.

#### Exporters

```cpp
// (1) Copy data to host container
template <typename Container>
__host__ Container to() const;
template <template <typename...> typename Container>
__host__ Container<Tuple<Ts...>> to() const;

// (2) Static cast to host container
template <typename Container>
__host__ explicit operator Container() const;
```

Where:

1.  `to<Constainer>()` copies data from device to host container (e.g., `std::vector<Tuple<Ts...>>`, `std::list<Tuple<Ts...>>`). Range value type can be deduced automatically (e.g., `to<std::vector>() -> std::vector<Tuple<Ts...>>`).
2.  Explicit conversion operator to host container, equivalent to `to<Container>()`.

#### Range interface

Member types:

```cpp
structure_of_arrays::size_type;
template <std::size_t N>
structure_of_arrays::element_type;

managed_structure_of_arrays::size_type;
template <std::size_t N>
managed_structure_of_arrays::element_type;
```

Member functions:

```cpp
using value = Tuple<Ts...>;
using reference = Tuple<Ts&...>;
using const_reference = Tuple<const Ts&...>;
using iterator = ...;
using const_iterator = ...;

__host__ __device__ reference operator[](size_type index) &;
__host__ __device__ const_reference operator[](size_type index) const&;
__host__ __device__ value operator[](size_type index) &&;
__host__ __device__ iterator begin() noexcept;
__host__ __device__ const_iterator begin() const noexcept;
__host__ __device__ iterator end() noexcept;
__host__ __device__ const_iterator end() const noexcept;
template <std::size_t N>
__host__ __device__ Ts[N]* data() noexcept;
template <std::size_t N>
__host__ __device__ const Ts[N]* data() const noexcept;
__host__ __device__ size_type size() const noexcept;
__host__ __device__ bool empty() const noexcept;
```

Concepts:

```cpp
using soa_type = structure_of_arrays<Tuple<Ts...>>;
using managed_soa_type = managed_structure_of_arrays<Tuple<Ts...>>;

std::ranges::range<soa_type>;
std::ranges::borrowed_range<soa_type>;  // only for device code
std::ranges::view<soa_type>;
std::ranges::output_range<soa_type, T>; // since C++23
std::ranges::input_range<soa_type>;
std::ranges::forward_range<soa_type>;
std::ranges::bidirectional_range<soa_type>;
std::ranges::random_access_range<soa_type>;
std::ranges::sized_range<soa_type>;
std::ranges::common_range<soa_type>;
std::ranges::viewable_range<soa_type>;

std::ranges::range<managed_soa_type>;
std::ranges::borrowed_range<managed_soa_type>;  // only for device code
std::ranges::view<managed_soa_type>;
std::ranges::output_range<managed_soa_type, T>; // since C++23
std::ranges::input_range<managed_soa_type>;
std::ranges::forward_range<managed_soa_type>;
std::ranges::bidirectional_range<managed_soa_type>;
std::ranges::random_access_range<managed_soa_type>;
std::ranges::sized_range<managed_soa_type>;
std::ranges::common_range<managed_soa_type>;
std::ranges::viewable_range<managed_soa_type>;
```

Note: When you define your own tuple-derived class, you may need to specialize `std::common_type` and `std::basic_common_reference` to satisfy some range concepts. For example:

```cpp
template <class... TTypes, class... UTypes>
requires requires { typename CustomTuple<std::common_type_t<TTypes, UTypes>...>; }
struct std::common_type<CustomTuple<TTypes...>, CustomTuple<UTypes...>>
{
    using type = CustomTuple<std::common_type_t<TTypes, UTypes>...>;
};

template <class... TTypes, class... UTypes, template <class> class TQual, template <class> class UQual>
requires requires { typename CustomTuple<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>; }
struct std::basic_common_reference<CustomTuple<TTypes...>, CustomTuple<UTypes...>, TQual, UQual>
{
    using type = CustomTuple<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>;
};
```

#### Smart pointer interface

```cpp
// (1) Reset pointer and size
__host__ void reset();
template <std::size_t N>
__device__ void reset(pointer<N> device_ptr);
template <std::size_t N>
__device__ void reset(const array<Ts[N]>& device_array);
template <std::size_t N>
__device__ void reset(const managed_array<Ts[N]>& device_array);

// (2) Boolean conversion
__host__ __device__ explicit operator bool() const noexcept;

// (3) Use count
__host__ std::uint32_t use_count() const noexcept;
```

Where:

1.  If host code calls `reset()`, the current device memory is freed and set new device pointers and size. If device code calls `reset<N>(...)`, it only sets the internal pointers of `N`-th array without freeing memory. The overloads with `array` and `managed_array` set the internal pointers and checking size consistency with `assert()` from the given device arrays.
2.  Bool conversion operator to check if the internal pointer is not null.
3.  Returns the current use count of the internal pointer. Note that this is only valid in host code.

Note: The device-side `reset` function does not affect to the memory management on the host side. It only changes the internal pointer on the device side.

#### Memory management

Note: Memory management functions are only available for `managed_structure_of_arrays` since they use unified memory.

```cpp
// (1) Prefetch
__host__ void prefetch(size_type start_idx, size_type len, int device_id = current_device_id, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch(int device_id = current_device_id, api::gpuStream_t stream = 0, bool recursive = true) const;

// (2) Prefetch to host memory
__host__ void prefetch_to_cpu(size_type start_idx, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const;
__host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const;

// (3) Memory advice
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, int device_id = current_device_id, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id = current_device_id, bool recursive = true) const;

// (4) Memory advice to host memory
__host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const;
__host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const;
```

Where:

1.  Wrapper for `cudaMemPrefetchAsync/hipMemPrefetchAsync` to prefetch unified memory to specified device. The former overload prefetches a memory range, while the latter overload prefetches the entire memory. If `recursive` is true and the value type of the array has `prefetch(...)` function, prefetch is called recursively for nested or member arrays.
2.  Host memory prefetching overloads with similar behavior to (1).
3.  Wrapper for `cudaMemAdvise/hipMemAdvise` to set memory advice for unified memory. The former overload sets advice for a memory range, while the latter overload sets advice for the entire memory. If `recursive` is true and the value type of the array has `mem_advise(...)` function, mem_advise is called recursively for nested or member arrays.
4.  Host memory advice overloads with similar behavior to (3).

### `jagged_array`

```cpp
template <typename T, typename SizeType = size_type_default>
class jagged_array : public managed_array<T, SizeType>;
template <template <typename...> typename Tuple = std::tuple, typename... Ts, typename SizeType = size_type_default>
class jagged_array : public managed_structure_of_arrays<Tuple<Ts...>, SizeType>;
```

The `jagged_array` class provides wrapper for managing multi-dimensional arrays with varying row lengths (jagged arrays) on the GPU. It inherits from the base array type, which can be either `managed_array<T>` or `structure_of_arrays<Tuple<Ts...>>`, to utilize their memory management and range interfaces. The jagged array has additional offsets to handle varying row sizes, allowing efficient access to elements using multi-dimensional indices.

Note that the only internal storage types currently supported are `managed_array` and `managed_structure_of_arrays`.

#### Constructors

```cpp
// (1) default constructor
jagged_array();

// (2) construct from sizes
template <std::ranges::input_range SizeRange>
__host__ explicit jagged_array(const SizeRange& sizes);
__host__ explicit jagged_array(std::initializer_list<size_type> sizes);

// (3) construct from sizes and base array (for managed_array)
template <std::ranges::input_range SizeRange>
__host__ jagged_array(const SizeRange& sizes, const managed_array<T, SizeType>& base_array);
__host__ jagged_array(std::initializer_list<size_type> sizes, const managed_array<T, SizeType>& base_array);

// (4) construct from sizes and base array (for managed_structure_of_arrays)
template <std::ranges::input_range SizeRange>
__host__ jagged_array(const SizeRange& sizes, const managed_structure_of_arrays<Tuple<Ts...>, SizeType>& base_array);
__host__ jagged_array(std::initializer_list<size_type> sizes, const managed_structure_of_arrays<Tuple<Ts...>, SizeType>& base_array);

// (5) construct from sizes and flat host container
template <std::ranges::input_range SizeRange, std::ranges::input_range Container>
__host__ jagged_array(const SizeRange& sizes, const Container& range);
__host__ jagged_array(std::initializer_list<size_type> sizes, const Container& range);

// (6) construct from nested host container
template <std::ranges::input_range NestedContainer>
__host__ jagged_array(const NestedContainer& nested_range);
__host__ jagged_array(std::initializer_list<std::initializer_list<T>> nested_list); // for managed_array
__host__ jagged_array(std::initializer_list<std::initializer_list<Tuple<Ts...>>> nested_list); // for managed_structure_of_arrays
```

Where:

1.  Default constructor creates an empty jagged_array with null pointers.
2.  Constructors from sizes allocate memory on the GPU for the jagged array based on the provided row sizes. The sizes can be provided as a range or an initializer list.
3.  Constructors from sizes and base array for `managed_array` type. The base array should contain the concatenated elements of all rows. The data is not copied; it is shared with the provided base array.
4.  Constructors from sizes and base array for `managed_structure_of_arrays` type. The base array should contain the concatenated elements of all rows in SoA layout. The data is not copied; it is shared with the provided base array.
5.  Constructors from sizes and flat host container copy data from the provided host container to the jagged array on the device. The host container should contain the concatenated elements of all rows.
6.  Constructors from nested host container copy data from the provided nested host container to the each row of jagged array on the device.

#### Exporters

Inherited from the base array type (`managed_array` or `managed_structure_of_arrays`).

#### Range interface

Inherited from the base array type (`managed_array` or `managed_structure_of_arrays`).

Additional member functions:

```cpp
// (1) Range interface for each row
__host__ __device__ std::ranges::subrange row(size_type i) noexcept;
__host__ __device__ std::ranges::subrange row(size_type i) const noexcept;
__host__ __device__ auto begin(size_type i) noexcept;
__host__ __device__ auto begin(size_type i) const noexcept;
__host__ __device__ auto end(size_type i) noexcept;
__host__ __device__ auto end(size_type i) const noexcept;
__host__ __device__ auto data(size_type i) noexcept;        // if base is managed_array
__host__ __device__ auto data(size_type i) const noexcept;  // if base is managed_array
__host__ __device__ size_type size(size_type i) const noexcept;
__host__ __device__ size_type num_rows() const noexcept;

// (2) Indexing operator with multi-dimensional indices
__host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) &;
__host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) const&;
__host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) &&;
__host__ __device__ decltype(auto) operator[](size_type i, size_type j) &;      // for C++23
__host__ __device__ decltype(auto) operator[](size_type i, size_type j) const&; // for C++23
__host__ __device__ decltype(auto) operator[](size_type i, size_type j) &&;     // for C++23
```

#### Smart pointer interface

Inherited from the base array type (`managed_array` or `managed_structure_of_arrays`).

#### Memory management

Inherited from the base array type (`managed_array` or `managed_structure_of_arrays`).

### Grid-stride range adapter

The following range adapter closure objects in the `views` namespace are provided for grid-stride loops in GPU kernels.

```cpp
views::block_thread_stride;
views::cluster_thread_stride;   // [*]
views::grid_thread_stride;
views::cluster_block_stride;    // [*]
views::grid_block_stride;       // [*]
views::grid_cluster_stride;     // [*]

// [*] Currently available only with CUDA backends.
```

They produce views that enable advancing the N-th element of the original range by a specified stride M. The pairs N and M correspond to the index of the thread/block/cluster within the block/cluster/grid and the number of threads/blocks/clusters, respectively.

#### `views::block_thread_stride`

The stride access based on the **thread index** and the number of threads in the **block**.

Example usage:

```cpp
for (auto& v : array | views::block_thread_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto block = cooperative_groups::this_thread_block();
for (auto i = static_cast<decltype(array.size())>(block.thread_rank()); i < array.size(); i += block.num_threads())
{
    // loop body
    array[i] = ...;
}
```

#### `views::cluster_thread_stride`

The stride access based on the **thread index** and the number of threads in the **cluster**.

Example usage:

```cpp
for (auto& v : array | views::cluster_thread_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto cluster = cooperative_groups::this_cluster();
for (auto i = static_cast<decltype(array.size())>(cluster.thread_rank()); i < array.size(); i += cluster.num_threads())
{
    // loop body
    array[i] = ...;
}
```

#### `views::grid_thread_stride`

The stride access based on the **thread index** and the number of threads in the **grid**.

Example usage:

```cpp
for (auto& v : array | views::grid_thread_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto grid = cooperative_groups::this_grid();
for (auto i = static_cast<decltype(array.size())>(grid.thread_rank()); i < array.size(); i += grid.num_threads())
{
    // loop body
    array[i] = ...;
}
```

#### `views::cluster_block_stride`

The stride access based on the **block index** and the number of blocks in the **cluster**.

Example usage:

```cpp
for (auto& v : array | views::cluster_block_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto cluster = cooperative_groups::this_cluster();
for (auto i = static_cast<decltype(array.size())>(cluster.block_rank()); i < array.size(); i += cluster.num_blocks())
{
    // loop body
    array[i] = ...;
}
```

#### `views::grid_block_stride`

The stride access based on the **block index** and the number of blocks in the **grid**.

Example usage:

```cpp
for (auto& v : array | views::grid_block_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto grid = cooperative_groups::this_grid();
for (auto i = static_cast<decltype(array.size())>(grid.block_rank()); i < array.size(); i += grid.num_blocks())
{
    // loop body
    array[i] = ...;
}
```

#### `views::grid_cluster_stride`

The stride access based on the **cluster index** and the number of clusters in the **grid**.

Example usage:

```cpp
for (auto& v : array | views::grid_cluster_stride)
{
    // loop body
    v = ...;
}
```

which is equivalent to:

```cpp
const auto grid = cooperative_groups::this_grid();
for (auto i = static_cast<decltype(array.size())>(grid.cluster_rank()); i < array.size(); i += grid.num_clusters())
{
    // loop body
    array[i] = ...;
}
```

### Utilities

#### CUDA/HIP API wrappers

The `gpu_ptr::api` namespace provides wrappers for commonly used CUDA and HIP API functions and types. The API functions are prefixed with `gpu` to avoid name conflicts instead of `cuda` or `hip`. See the definitions in the [gpu_runtime_api.hpp](include/gpu_runtime_api.hpp) file for details.

#### Macros

**Backend selection:**

Define `ENABLE_HIP` to use HIP backend. Otherwise, CUDA backend is used by default.

**Default size type selection:**

Define `GPU_USE_32BIT_SIZE_TYPE_DEFAULT` to use `std::uint32_t` as the default size type for array-like classes. Otherwise, `std::size_t` is used by default.

**API error checking:**

`GPU_CHECK_ERROR()` function macro to check CUDA/HIP API errors. If an error occurs, it throws a `std::runtime_error` with the error message. Example usage:

```cpp
GPU_CHECK_ERROR(gpu_ptr::api::gpuGetDevice(&device_id));
```

**Device and host compilation macros:**

gpu-ptr library defines `GPU_DEVICE_COMPILE`, `GPU_OVERLOAD_DEVICE`, and `GPU_OVERLOAD_HOST` macros depending on host or device code compilation. The `GPU_DEVICE_COMPILE` macro is defined when compiling device code. The `GPU_OVERLOAD_DEVICE` and `GPU_OVERLOAD_HOST` macros handle the differences in behavior between CUDA and HIP for [overloading based on host and device code](https://llvm.org/docs/CompileCudaWithLLVM.html#overloading-based-on-host-and-device-attributes). The nvcc does not allow overloading based on `__host__` and `__device__` attributes with the same function signature, while hipcc allows it.

Example usage:

```cpp
__host__ __device__ void func()
{
#ifdef GPU_DEVICE_COMPILE
    // Device code
#else
    // Host code
#endif
}

#ifdef GPU_OVERLOAD_HOST
__host__ void foo()
{
    // Host code
}
#endif
#ifdef GPU_OVERLOAD_DEVICE
__device__ int foo()
{
    // Device code
}
#endif
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
