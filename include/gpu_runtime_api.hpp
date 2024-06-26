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

// NOLINTBEGIN
namespace gpu_smart_ptr::detail
{

#ifdef ENABLE_HIP

    using gpuError_t = ::hipError_t;
#define gpuSuccess hipSuccess
    __host__ inline const char* gpuGetErrorName(gpuError_t gpu_error) { return ::hipGetErrorName(gpu_error); }
    __host__ inline const char* gpuGetErrorString(gpuError_t gpu_error) { return ::hipGetErrorString(gpu_error); }

    using gpuMemcpyKind = ::hipMemcpyKind;
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
    __host__ inline decltype(auto) gpuDeviceSynchronize() { return ::hipDeviceSynchronize(); }
    __host__ inline decltype(auto) gpuGetDeviceCount(int* count) { return ::hipGetDeviceCount(count); }
    __host__ inline decltype(auto) gpuGetDevice(int* device) { return ::hipGetDevice(device); }

#else

    using gpuError_t = ::cudaError_t;
#define gpuSuccess cudaSuccess
    __host__ inline const char* gpuGetErrorName(gpuError_t gpu_error) { return ::cudaGetErrorName(gpu_error); }
    __host__ inline const char* gpuGetErrorString(gpuError_t gpu_error) { return ::cudaGetErrorString(gpu_error); }

    using gpuMemcpyKind = cudaMemcpyKind;
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
    __host__ inline decltype(auto) gpuDeviceSynchronize() { return ::cudaDeviceSynchronize(); }
    __host__ inline decltype(auto) gpuGetDeviceCount(int* count) { return ::cudaGetDeviceCount(count); }
    __host__ inline decltype(auto) gpuGetDevice(int* device) { return ::cudaGetDevice(device); }
#endif
}  // namespace gpu_smart_ptr::detail

// NOLINTEND
