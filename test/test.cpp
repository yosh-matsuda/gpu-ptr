#include "gpu_ptr.hpp"

#include <gtest/gtest.h>
#include <array>
#include <list>
#include <vector>
// NOLINTBEGIN

#define CALL_KERNEL(cuda_func) \
    do {                       \
        (cuda_func);           \
    } while (0)
#define CALL_KERNEL_SYNC(cuda_func)                               \
    do {                                                          \
        (cuda_func);                                              \
        CHECK_GPU_ERROR(gpu_ptr::detail::gpuDeviceSynchronize()); \
    } while (0)

using int64cu_t = long long int;
using uint64cu_t = unsigned long long int;

template <typename T1, typename T2>
__device__ __forceinline__ T1 atomic_add(T1& target, T2 value)
{
    return ::atomicAdd(reinterpret_cast<T1*>(&target), static_cast<T1>(value));
}

template <typename T2>
__device__ __forceinline__ std::int64_t atomic_add(std::int64_t& target, T2 value)
{
    static_assert(sizeof(::int64cu_t) == sizeof(std::int64_t));
    return ::atomicAdd(reinterpret_cast<::uint64cu_t*>(&target), static_cast<::uint64cu_t>(value));
}

template <typename T2>
__device__ __forceinline__ std::uint64_t atomic_add(std::uint64_t& target, T2 value)
{
    static_assert(sizeof(::uint64cu_t) == sizeof(std::uint64_t));
    return ::atomicAdd(reinterpret_cast<::uint64cu_t*>(&target), static_cast<::uint64cu_t>(value));
}

using namespace gpu_ptr;

template <std::size_t Size, typename T>
__global__ void capture_shared_memory(array<T> a)
{
    extern __shared__ T sh[];
    __shared__ T sum;
    if (threadIdx.x == 0)
    {
        a.reset(sh, Size);
        sum = 0;
    }
    __syncthreads();

    for (unsigned idx = threadIdx.x; idx < Size; idx += blockDim.x)
    {
        sh[idx] = T(1);
    }
    __syncthreads();

    for (unsigned idx = threadIdx.x; idx < Size; idx += blockDim.x)
    {
        atomic_add(sum, sh[idx]);
    }
    __syncthreads();

    if (threadIdx.x == 0) assert(sum == Size);
}

template <typename ValuePtr>
__global__ void increment_value(ValuePtr v)
{
    if (threadIdx.x == 0) ++(*v);
}

template <typename ValueType>
__global__ void capture_shared_memory(value<ValueType> v)
{
    extern __shared__ ValueType sh[];
    if (threadIdx.x == 0)
    {
        auto temp = *v;
        v.reset(sh);
        *v = temp + 1;  // does not affect on CPU side
    }
}

TEST(gpu_ptr, array)
{
    using ValueType = int;
    static_assert(std::is_default_constructible_v<array<ValueType>>);
    static_assert(std::is_constructible_v<array<ValueType>, const array<ValueType>&>);
    static_assert(std::is_constructible_v<array<ValueType>, array<ValueType>&&>);
    static_assert(std::is_constructible_v<array<ValueType>, std::size_t>);
    static_assert(std::is_constructible_v<array<ValueType>, std::size_t, default_init_tag>);
    static_assert(std::is_constructible_v<array<ValueType>, std::size_t, ValueType>);
    static_assert(std::ranges::contiguous_range<std::vector<ValueType>> &&
                  std::is_constructible_v<array<ValueType>, std::vector<ValueType>>);
    static_assert(!std::ranges::contiguous_range<std::list<ValueType>> &&
                  std::is_constructible_v<array<ValueType>, std::list<ValueType>>);
    static_assert(std::is_constructible_v<array<ValueType>, std::initializer_list<ValueType>>);

    // deduction guide
    static_assert(std::same_as<decltype(array(std::declval<std::vector<ValueType>>())), array<ValueType>>);
    static_assert(std::same_as<decltype(array(std::declval<std::list<ValueType>>())), array<ValueType>>);
    static_assert(std::same_as<decltype(array(std::declval<std::array<ValueType, 1>>())), array<ValueType>>);
    static_assert(std::same_as<decltype(array(std::declval<std::initializer_list<ValueType>>())), array<ValueType>>);

    {
        auto a = array<ValueType>();
        EXPECT_EQ(a.size(), 0);
        EXPECT_EQ(0, gpu_memory_usage());
    }

    {
        constexpr auto s = std::size_t{1024};
        auto a = array<ValueType>(s);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s));
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1025};
        auto a = array<ValueType>(s, default_init);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1026};
        constexpr auto i = ValueType{42};
        auto a = array<ValueType>(s, i);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s, i));
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1027};
        constexpr auto i = ValueType{43};
        auto v = std::vector<ValueType>(s, i);
        auto a = array<ValueType>(v);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v2 = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1028};
        constexpr auto i = ValueType{44};
        auto v = std::array<ValueType, s>();
        std::ranges::fill(v, i);
        auto a = array<ValueType>(v);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v2 = a.to<std::array<ValueType, s>>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1029};
        constexpr auto i = ValueType{45};
        auto l = std::list<ValueType>(s, i);
        auto a = array<ValueType>(l);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto l2 = a.to<std::list>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, l.size());
        EXPECT_EQ(l, l2);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        auto a = array<ValueType>({-1, -2, -3, -4, -5});
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(5, a.size());
        EXPECT_EQ(5, v.size());
        EXPECT_EQ((std::vector{-1, -2, -3, -4, -5}), v);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        auto s = std::vector{-1, -2, -3, -4, -5};
        auto u = unified_array<ValueType>(s);
        EXPECT_EQ(sizeof(ValueType) * u.size(), gpu_memory_usage());

        auto a = array<ValueType>(u);
        auto d = a.to<unified_array>();
        auto v = d.to<std::vector>();
        EXPECT_EQ(s, v);
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, array_capture_shared_ptr)
{
    {
        using ValueType = int;
        constexpr auto s = 1024;
        auto arr = array<ValueType>(s);
        CALL_KERNEL_SYNC((capture_shared_memory<s><<<1, s, s * sizeof(ValueType)>>>(arr)));
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, unified_array_trivial)
{
    using ValueType = int;
    static_assert(std::is_default_constructible_v<unified_array<ValueType>>);
    static_assert(std::is_constructible_v<unified_array<ValueType>, const unified_array<ValueType>&>);
    static_assert(std::is_constructible_v<unified_array<ValueType>, unified_array<ValueType>&&>);
    static_assert(std::is_constructible_v<unified_array<ValueType>, std::size_t>);
    static_assert(std::is_constructible_v<unified_array<ValueType>, std::size_t, default_init_tag>);
    static_assert(std::is_constructible_v<unified_array<ValueType>, std::size_t, ValueType>);
    static_assert(std::ranges::contiguous_range<std::vector<ValueType>> &&
                  std::is_constructible_v<unified_array<ValueType>, std::vector<ValueType>>);
    static_assert(!std::ranges::contiguous_range<std::list<ValueType>> &&
                  std::is_constructible_v<unified_array<ValueType>, std::list<ValueType>>);
    static_assert(std::is_constructible_v<unified_array<ValueType>, std::initializer_list<ValueType>>);
    static_assert(std::is_constructible_v<unified_array<ValueType>, array<ValueType>>);

    // deduction guide
    static_assert(
        std::same_as<decltype(unified_array(std::declval<std::vector<ValueType>>())), unified_array<ValueType>>);
    static_assert(
        std::same_as<decltype(unified_array(std::declval<std::list<ValueType>>())), unified_array<ValueType>>);
    static_assert(
        std::same_as<decltype(unified_array(std::declval<std::array<ValueType, 1>>())), unified_array<ValueType>>);
    static_assert(std::same_as<decltype(unified_array(std::declval<std::initializer_list<ValueType>>())),
                               unified_array<ValueType>>);

    {
        auto a = unified_array<ValueType>();
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());
        EXPECT_EQ(a.size(), 0);
        EXPECT_EQ(0, gpu_memory_usage());
    }
    {
        constexpr auto s = std::size_t{1024};
        auto a = unified_array<ValueType>(s);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s));
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1025};
        auto a = unified_array<ValueType>(s, default_init);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1026};
        constexpr auto i = ValueType{42};
        auto a = unified_array<ValueType>(s, i);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s, i));
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1027};
        constexpr auto i = ValueType{43};
        auto v = std::vector<ValueType>(s, i);
        auto a = unified_array<ValueType>(v);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v2 = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1028};
        constexpr auto i = ValueType{44};
        auto v = std::array<ValueType, s>();
        std::ranges::fill(v, i);
        auto a = unified_array<ValueType>(v);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v2 = a.to<std::array<ValueType, s>>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto s = std::size_t{1029};
        constexpr auto i = ValueType{45};
        auto l = std::list<ValueType>(s, i);
        auto a = unified_array<ValueType>(l);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto l2 = a.to<std::list>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, l.size());
        EXPECT_EQ(l, l2);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        auto a = unified_array<ValueType>{-1, -2, -3, -4, -5};
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto v = a.to<std::vector>();
        EXPECT_EQ(5, a.size());
        EXPECT_EQ(5, v.size());
        EXPECT_EQ((std::vector{-1, -2, -3, -4, -5}), v);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        auto s = std::vector{-1, -2, -3, -4, -5};
        auto a = array<ValueType>(s);
        EXPECT_EQ(sizeof(ValueType) * a.size(), gpu_memory_usage());

        auto u = unified_array<ValueType>(a);
        auto d = u.to<array>();
        auto v = d.to<std::vector>();
        EXPECT_EQ(s, v);
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, unified_array_struct)
{
    using ValueType = int;
    struct UserType
    {
        array<ValueType> a;
        unified_array<ValueType> u;
    };
    static_assert(!std::is_trivially_copyable_v<UserType>);

    {
        auto s1 = {1, 2, 3};
        auto s2 = {4, 5, 6};
        auto s3 = {7, 8, 9};
        auto s4 = {10, 11, 12};

        auto u = unified_array<UserType>{UserType{.a = s1, .u = s2}, UserType{.a = s3, .u = s4}};
        EXPECT_EQ(std::vector(s1), u[0].a.to<std::vector>());
        EXPECT_EQ(std::vector(s2), u[0].u.to<std::vector>());
        EXPECT_EQ(std::vector(s3), u[1].a.to<std::vector>());
        EXPECT_EQ(std::vector(s4), u[1].u.to<std::vector>());

        EXPECT_EQ(sizeof(UserType) * u.size() + sizeof(ValueType) * u[0].a.size() + sizeof(ValueType) * u[0].u.size() +
                      sizeof(ValueType) * u[1].a.size() + sizeof(ValueType) * u[1].u.size(),
                  gpu_memory_usage());
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, array_conversion)
{
    using IntType = int;
    using FloatType = double;

    {
        [[maybe_unused]] auto u1 = unified_array(std::vector<std::vector<std::list<array<IntType>>>>());
        static_assert(std::same_as<unified_array<unified_array<unified_array<unified_array<IntType>>>>, decltype(u1)>);
        [[maybe_unused]] auto u2 = unified_array<unified_array<array<IntType>>>();
        static_assert(std::same_as<std::vector<std::vector<std::vector<IntType>>>, decltype(u2.to<std::vector>())>);
        [[maybe_unused]] auto u3 = unified_array<std::vector<array<IntType>>>();
        static_assert(std::same_as<std::vector<std::vector<array<IntType>>>, decltype(u3.to<std::vector>())>);
    }

    // conversion constructors
    {
        auto ui = unified_array<IntType>{1, 2, 3};
        auto uf = unified_array<FloatType>(ui);
        auto ui2 = unified_array<IntType>(uf);
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }
    {
        auto ai = array<IntType>{1, 2, 3};
        auto uf = unified_array<FloatType>(ai);
        auto ai2 = array<IntType>(uf);
        EXPECT_EQ(ai.to<std::vector>(), ai2.to<std::vector>());
    }
    {
        auto ui = unified_array<IntType>{1, 2, 3};
        auto af = array<FloatType>(ui);
        auto ui2 = unified_array<IntType>(af);
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }

    // conversion using `to` method
    {
        auto ai = array<IntType>{1, 2, 3};
        auto af = ai.to<array<FloatType>>();
        auto ai2 = af.to<array<IntType>>();
        EXPECT_EQ(ai.to<std::vector>(), ai2.to<std::vector>());
    }
    {
        auto ui = unified_array<IntType>{1, 2, 3};
        auto uf = ui.to<unified_array<FloatType>>();
        auto ui2 = uf.to<unified_array<IntType>>();
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }
    {
        auto ai = array<IntType>{1, 2, 3};
        auto uf = ai.to<unified_array<FloatType>>();
        auto ai2 = uf.to<array<IntType>>();
        EXPECT_EQ(ai.to<std::vector>(), ai2.to<std::vector>());
    }
    {
        auto ui = unified_array<IntType>{1, 2, 3};
        auto af = ui.to<array<FloatType>>();
        auto ui2 = af.to<unified_array<IntType>>();
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }
}

TEST(gpu_ptr, value)
{
    using ValueType = int;
    static_assert(std::is_default_constructible_v<value<ValueType>>);
    static_assert(std::is_constructible_v<value<ValueType>, default_init_tag>);
    static_assert(std::is_constructible_v<value<ValueType>, const value<ValueType>&>);
    static_assert(std::is_constructible_v<value<ValueType>, value<ValueType>&&>);
    static_assert(std::is_constructible_v<value<ValueType>, const ValueType&>);

    // deduction guide
    static_assert(std::same_as<decltype(value(std::declval<ValueType>())), value<ValueType>>);

    {
        auto v = value<ValueType>();
        EXPECT_EQ(sizeof(ValueType), gpu_memory_usage());
        EXPECT_EQ(ValueType(), *v);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto i = ValueType{13};
        auto v = value(i);
        EXPECT_EQ(sizeof(ValueType), gpu_memory_usage());
        EXPECT_EQ(ValueType(i), *v);

        CALL_KERNEL_SYNC((increment_value<<<1, 1>>>(v)));
        EXPECT_EQ(ValueType(i + 1), *v);
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, value_struct)
{
    struct UserType
    {
        int i;
        double d;
    };
    static_assert(std::is_trivially_copyable_v<UserType>);

    {
        constexpr auto i = 10;
        constexpr auto d = 3.14;
        auto v = value<UserType>(UserType{.i = i, .d = d});
        EXPECT_EQ(sizeof(UserType), gpu_memory_usage());
        EXPECT_EQ(i, v->i);
        EXPECT_EQ(d, v->d);
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, value_capture_shared_ptr)
{
    {
        using ValueType = int;
        constexpr auto i = 1023;
        auto v = value<ValueType>(i);
        CALL_KERNEL_SYNC((capture_shared_memory<<<1, 1, sizeof(ValueType)>>>(v)));
        EXPECT_EQ(ValueType(i), *v);  // modification on GPU does not affect on CPU side
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, unified_value_trivial)
{
    using ValueType = int;
    static_assert(std::is_default_constructible_v<unified_value<ValueType>>);
    static_assert(std::is_constructible_v<unified_value<ValueType>, default_init_tag>);
    static_assert(std::is_constructible_v<unified_value<ValueType>, const unified_value<ValueType>&>);
    static_assert(std::is_constructible_v<unified_value<ValueType>, unified_value<ValueType>&&>);
    static_assert(std::is_constructible_v<unified_value<ValueType>, const ValueType&>);

    // deduction guide
    static_assert(std::same_as<decltype(unified_value(std::declval<ValueType>())), unified_value<ValueType>>);

    {
        auto v = unified_value<ValueType>();
        EXPECT_EQ(sizeof(ValueType), gpu_memory_usage());
        EXPECT_EQ(ValueType(), *v);
    }
    EXPECT_EQ(0, gpu_memory_usage());

    {
        constexpr auto i = ValueType{13};
        auto v = unified_value(i);
        EXPECT_EQ(sizeof(ValueType), gpu_memory_usage());
        EXPECT_EQ(ValueType(i), *v);

        CALL_KERNEL_SYNC((increment_value<<<1, 1>>>(v)));
        EXPECT_EQ(ValueType(i + 1), *v);
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

TEST(gpu_ptr, unified_value_struct)
{
    using ValueType = int;
    struct UserType
    {
        array<ValueType> a;
        unified_array<ValueType> u;
    };
    static_assert(!std::is_trivially_copyable_v<UserType>);

    {
        auto s1 = {1, 2, 3};
        auto s2 = {4, 5, 6};

        auto u = unified_value{UserType{.a = s1, .u = s2}};
        EXPECT_EQ(std::vector(s1), u->a.to<std::vector>());
        EXPECT_EQ(std::vector(s2), u->u.to<std::vector>());

        EXPECT_EQ(sizeof(UserType) + sizeof(ValueType) * u->a.size() + sizeof(ValueType) * u->u.size(),
                  gpu_memory_usage());
    }
    EXPECT_EQ(0, gpu_memory_usage());
}

template <typename T>
__global__ void readme1_func(array<T> arr)
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

TEST(gpu_ptr, readme1)
{
    auto arr = array<int>(5, 1);
    readme1_func<<<1, 1, sizeof(int) * 5>>>(arr);

    auto vec = std::vector<int>{2, 2, 2, 2, 2};
    EXPECT_EQ(vec, arr.to<std::vector>());
}

struct Data
{
    unified_array<int> ai = {1, 2, 3, 4, 5};
    unified_array<double> ad = {1.0, 2.0, 3.0, 4.0, 5.0};
};

__global__ void readme2_func(unified_array<Data> data_array)
{
    for (auto& d : data_array)
    {
        d.ai[0] -= 10;
        d.ad[0] -= 10.0;
    }
}

TEST(gpu_ptr, readme2)
{
    auto data_array = unified_array<Data>(5);

    for (auto& d : data_array)
    {
        d.ai[0] += 10;
        d.ad[0] += 10.0;
    }

    for (const auto& d : data_array)
    {
        EXPECT_EQ((std::vector<int>{11, 2, 3, 4, 5}), d.ai.to<std::vector>());
        EXPECT_EQ((std::vector<double>{11.0, 2.0, 3.0, 4.0, 5.0}), d.ad.to<std::vector>());
    }

    data_array.prefetch_to_gpu();
    readme2_func<<<1, 1>>>(data_array);
    data_array.prefetch_to_cpu();

    for (const auto& d : data_array)
    {
        EXPECT_EQ((std::vector<int>{1, 2, 3, 4, 5}), d.ai.to<std::vector>());
        EXPECT_EQ((std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}), d.ad.to<std::vector>());
    }
}

TEST(gpu_ptr, readme4)
{
    auto val = value<int>{5};
    EXPECT_EQ(5, *val);

    auto uval = unified_value<int>{5};
    EXPECT_EQ(5, *uval);
}
// NOLINTEND
