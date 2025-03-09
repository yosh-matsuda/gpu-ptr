/*===================================================*
|  GPU pointer wrapper (gpu-ptr) version v0.0.1      |
|  https://github.com/yosh-matsuda/gpu-ptr           |
|                                                    |
|  Copyright (c) 2024 Yoshiki Matsuda @yosh-matsuda  |
|                                                    |
|  This software is released under the MIT License.  |
|  https://opensource.org/license/mit/               |
====================================================*/

#pragma once
#ifdef ENABLE_HIP
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#else
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#endif

#include <sstream>

// NOLINTBEGIN
namespace gpu_smart_ptr::detail
{
#ifdef ENABLE_HIP

    using gpuError_t = ::hipError_t;
#define gpuSuccess hipSuccess
    __host__ inline const char* gpuGetErrorName(gpuError_t gpu_error) { return ::hipGetErrorName(gpu_error); }
    __host__ inline const char* gpuGetErrorString(gpuError_t gpu_error) { return ::hipGetErrorString(gpu_error); }

    using gpuMemcpyKind = ::hipMemcpyKind;
    enum class gpuMemoryAdvise
    {
        SetReadMostly = hipMemAdviseSetReadMostly,
        UnsetReadMostly = hipMemAdviseUnsetReadMostly,
        SetPreferredLocation = hipMemAdviseSetPreferredLocation,
        UnsetPreferredLocation = hipMemAdviseUnsetPreferredLocation,
        SetAccessedBy = hipMemAdviseSetAccessedBy,
        UnsetAccessedBy = hipMemAdviseUnsetAccessedBy,
    };
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDefault hipMemcpyDefault

#define gpuMemAttachGlobal hipMemAttachGlobal
#define gpuMemAttachHost hipMemAttachHost
#define gpuMemAttachSingle hipMemAttachSingle

#define gpuCpuDeviceId hipCpuDeviceId
#define gpuInvalidDeviceId hipInvalidDeviceId

    using gpuStream_t = hipStream_t;

    __host__ inline decltype(auto) gpuMalloc(void** ptr, std::size_t size) { return ::hipMalloc(ptr, size); }
    __host__ inline decltype(auto) gpuMallocManaged(void** ptr, std::size_t size,
                                                    unsigned int flags = hipMemAttachGlobal)
    {
        return ::hipMallocManaged(ptr, size, flags);
    }
    __host__ inline decltype(auto) gpuFree(void* ptr) { return ::hipFree(ptr); }
    __host__ inline decltype(auto) gpuMemcpy(void* dst, const void* src, std::size_t size, gpuMemcpyKind kind)
    {
        return ::hipMemcpy(dst, src, size, kind);
    }
    __host__ inline decltype(auto) gpuMemPrefetchAsync(const void* dev_ptr, std::size_t count, int device,
                                                       gpuStream_t stream = 0)
    {
        return ::hipMemPrefetchAsync(dev_ptr, count, device, stream);
    }
    __host__ inline decltype(auto) gpuMemAdvise(const void* devPtr, size_t count, gpuMemoryAdvise advice, int device)
    {
        return ::hipMemAdvise(devPtr, count, static_cast<hipMemoryAdvise>(advice), device);
    }
    __host__ inline decltype(auto) gpuDeviceSynchronize() { return ::hipDeviceSynchronize(); }
    __host__ inline decltype(auto) gpuGetDeviceCount(int* count) { return ::hipGetDeviceCount(count); }
    __host__ inline decltype(auto) gpuGetDevice(int* device) { return ::hipGetDevice(device); }
    __host__ inline decltype(auto) gpuStreamCreate(gpuStream_t* stream) { return ::hipStreamCreate(stream); }
    __host__ inline decltype(auto) gpuStreamSynchronize(gpuStream_t stream) { return ::hipStreamSynchronize(stream); }
    template <typename T>
    __host__ decltype(auto) gpuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T f, int blockSize,
                                                                         size_t dynSharedMemPerBlk)
    {
        return ::hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize, dynSharedMemPerBlk);
    }
#ifdef __NVCC__
    template <typename T>
    __host__ inline decltype(auto) gpuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmem, T* f, int numBlocks,
                                                                            int blockSize)
    {
        return hipCUDAErrorTohipError(
            ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmem, f, numBlocks, blockSize));
    }
#endif
    __host__ inline decltype(auto) gpuSetDevice(int device) { return ::hipSetDevice(device); }
    __host__ inline decltype(auto) gpuGetLastError() { return ::hipGetLastError(); }

#else

    using gpuError_t = ::cudaError_t;
#define gpuSuccess cudaSuccess
    __host__ inline const char* gpuGetErrorName(gpuError_t gpu_error) { return ::cudaGetErrorName(gpu_error); }
    __host__ inline const char* gpuGetErrorString(gpuError_t gpu_error) { return ::cudaGetErrorString(gpu_error); }

    using gpuMemcpyKind = ::cudaMemcpyKind;
    enum class gpuMemoryAdvise
    {
        SetReadMostly = cudaMemAdviseSetReadMostly,
        UnsetReadMostly = cudaMemAdviseUnsetReadMostly,
        SetPreferredLocation = cudaMemAdviseSetPreferredLocation,
        UnsetPreferredLocation = cudaMemAdviseUnsetPreferredLocation,
        SetAccessedBy = cudaMemAdviseSetAccessedBy,
        UnsetAccessedBy = cudaMemAdviseUnsetAccessedBy,
    };
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDefault cudaMemcpyDefault

#define gpuMemAttachGlobal cudaMemAttachGlobal
#define gpuMemAttachHost cudaMemAttachHost
#define gpuMemAttachSingle cudaMemAttachSingle

#define gpuCpuDeviceId cudaCpuDeviceId
#define gpuInvalidDeviceId cudaInvalidDeviceId

    using gpuStream_t = cudaStream_t;

    __host__ inline decltype(auto) gpuMalloc(void** devPtr, std::size_t size) { return ::cudaMalloc(devPtr, size); }
    __host__ inline decltype(auto) gpuMallocManaged(void** devPtr, std::size_t size,
                                                    unsigned int flags = cudaMemAttachGlobal)
    {
        return ::cudaMallocManaged(devPtr, size, flags);
    }
    __host__ inline decltype(auto) gpuFree(void* devPtr) { return ::cudaFree(devPtr); }
    __host__ inline decltype(auto) gpuMemcpy(void* dst, const void* src, std::size_t size, gpuMemcpyKind kind)
    {
        return ::cudaMemcpy(dst, src, size, kind);
    }
    __host__ inline decltype(auto) gpuMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice,
                                                       gpuStream_t stream = 0)
    {
        return ::cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
    }
    __host__ inline decltype(auto) gpuMemAdvise(const void* devPtr, size_t count, gpuMemoryAdvise advice, int device)
    {
        return ::cudaMemAdvise(devPtr, count, static_cast<cudaMemoryAdvise>(advice), device);
    }
    __host__ inline decltype(auto) gpuDeviceSynchronize() { return ::cudaDeviceSynchronize(); }
    __host__ inline decltype(auto) gpuGetDeviceCount(int* count) { return ::cudaGetDeviceCount(count); }
    __host__ inline decltype(auto) gpuGetDevice(int* device) { return ::cudaGetDevice(device); }
    __host__ inline decltype(auto) gpuStreamCreate(gpuStream_t* stream) { return ::cudaStreamCreate(stream); }
    __host__ inline decltype(auto) gpuStreamSynchronize(gpuStream_t stream) { return ::cudaStreamSynchronize(stream); }
    template <typename T>
    __host__ decltype(auto) gpuOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T f, int blockSize,
                                                                         size_t dynSharedMemPerBlk)
    {
        return ::cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, f, blockSize, dynSharedMemPerBlk);
    }
    template <typename T>
    __host__ inline decltype(auto) gpuOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmem, T* f, int numBlocks,
                                                                            int blockSize)
    {
        return ::cudaOccupancyAvailableDynamicSMemPerBlock(dynamicSmem, f, numBlocks, blockSize);
    }
    __host__ inline decltype(auto) gpuSetDevice(int device) { return ::cudaSetDevice(device); }
    __host__ inline decltype(auto) gpuGetLastError() { return ::cudaGetLastError(); }
#endif

    __host__ inline void check_gpu_error(const gpuError_t e, [[maybe_unused]] const char* f,
                                         [[maybe_unused]] decltype(__LINE__) n)
    {
        if (e != gpuSuccess)
        {
            std::stringstream s;
#ifdef NDEBUG
            s << gpuGetErrorName(e) << " (" << static_cast<unsigned>(e) << "): " << gpuGetErrorString(e);
#else
            s << gpuGetErrorName(e) << " (" << static_cast<unsigned>(e) << ")@" << f << "#L" << n << ": "
              << gpuGetErrorString(e);
#endif
            throw std::runtime_error{s.str()};
        }
    }

}  // namespace gpu_smart_ptr::detail

#define CHECK_GPU_ERROR(expr) (gpu_smart_ptr::detail::check_gpu_error(expr, __FILE__, __LINE__))

// NOLINTEND
