#include "gpu_ptr.hpp"

#include <gtest/gtest.h>
#include <array>
#include <list>
#include <vector>
// NOLINTBEGIN

using namespace gpu_smart_ptr;

#define CALL_KERNEL(cuda_func) \
    do {                       \
        (cuda_func);           \
    } while (0)
#define CALL_KERNEL_SYNC(cuda_func)                      \
    do {                                                 \
        (cuda_func);                                     \
        CHECK_GPU_ERROR(detail::gpuDeviceSynchronize()); \
    } while (0)

using int64cu_t = long long int;
using uint64cu_t = unsigned long long int;

constexpr auto gpu_max_threads = 1024;

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

template <std::size_t Size, typename T>
__global__ void capture_shared_memory(array_ptr<T> a)
{
    auto block = cooperative_groups::this_thread_block();
    extern __shared__ T sh[];
    __shared__ T sum;
    if (block.thread_rank() == 0)
    {
        sum = 0;
    }
    block.sync();
    a.reset(sh, Size);

    for (unsigned idx = block.thread_rank(); idx < Size; idx += block.size())
    {
        sh[idx] = T(1);
    }
    block.sync();

    for (unsigned idx = block.thread_rank(); idx < Size; idx += block.size())
    {
        atomic_add(sum, a[idx]);
    }
    block.sync();

    if (block.thread_rank() == 0)
    {
        assert(sum == Size);
    }
}

template <typename ValuePtr>
__global__ void increment_value(ValuePtr v)
{
    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() == 0) ++(*v);
}

template <typename ValueType>
__global__ void capture_shared_memory(value<ValueType> v)
{
    auto block = cooperative_groups::this_thread_block();
    extern __shared__ ValueType sh[];
    if (block.thread_rank() == 0)
    {
        auto temp = *v;
        v.reset(sh);
        *v = temp + 1;  // does not affect on CPU side
    }
}

TEST(gpu_smart_ptr, array_ptr)
{
    using ValueType = int;
    static_assert(std::is_default_constructible_v<array_ptr<ValueType>>);
    static_assert(std::is_constructible_v<array_ptr<ValueType>, const array_ptr<ValueType>&>);
    static_assert(std::is_constructible_v<array_ptr<ValueType>, array_ptr<ValueType>&&>);
    static_assert(std::is_constructible_v<array_ptr<ValueType>, std::size_t>);
    static_assert(std::is_constructible_v<array_ptr<ValueType>, std::size_t, default_init_tag>);
    static_assert(std::is_constructible_v<array_ptr<ValueType>, std::size_t, ValueType>);
    static_assert(std::ranges::contiguous_range<std::vector<ValueType>> &&
                  std::is_constructible_v<array_ptr<ValueType>, std::vector<ValueType>>);
    static_assert(!std::ranges::contiguous_range<std::list<ValueType>> &&
                  std::is_constructible_v<array_ptr<ValueType>, std::list<ValueType>>);
    static_assert(std::is_constructible_v<array_ptr<ValueType>, std::initializer_list<ValueType>>);

    // deduction guide
    static_assert(std::same_as<decltype(array_ptr(std::declval<std::vector<ValueType>>())), array_ptr<ValueType>>);
    static_assert(std::same_as<decltype(array_ptr(std::declval<std::list<ValueType>>())), array_ptr<ValueType>>);
    static_assert(std::same_as<decltype(array_ptr(std::declval<std::array<ValueType, 1>>())), array_ptr<ValueType>>);
    static_assert(
        std::same_as<decltype(array_ptr(std::declval<std::initializer_list<ValueType>>())), array_ptr<ValueType>>);

    {
        auto a = array_ptr<ValueType>();
        EXPECT_EQ(a.size(), 0);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
    }

    {
        constexpr auto s = std::size_t{1024};
        auto a = array_ptr<ValueType>(s);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s));
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1025};
        auto a = array_ptr<ValueType>(s, default_init);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1026};
        constexpr auto i = ValueType{42};
        auto a = array_ptr<ValueType>(s, i);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s, i));
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1027};
        constexpr auto i = ValueType{43};
        auto v = std::vector<ValueType>(s, i);
        auto a = array_ptr<ValueType>(v);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v2 = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1028};
        constexpr auto i = ValueType{44};
        auto v = std::array<ValueType, s>();
        std::ranges::fill(v, i);
        auto a = array_ptr<ValueType>(v);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v2 = a.to<std::array<ValueType, s>>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1029};
        constexpr auto i = ValueType{45};
        auto l = std::list<ValueType>(s, i);
        auto a = array_ptr<ValueType>(l);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto l2 = a.to<std::list>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, l.size());
        EXPECT_EQ(l, l2);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        auto a = array_ptr<ValueType>({-1, -2, -3, -4, -5});
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(5, a.size());
        EXPECT_EQ(5, v.size());
        EXPECT_EQ((std::vector{-1, -2, -3, -4, -5}), v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        auto s = std::vector{-1, -2, -3, -4, -5};
        auto u = unified_array_ptr<ValueType>(s);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * u.size()));

        auto a = array_ptr<ValueType>(u);
        auto d = a.to<unified_array_ptr>();
        auto v = d.to<std::vector>();
        EXPECT_EQ(s, v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, array_capture_shared_ptr)
{
    {
        using ValueType = int;
        constexpr auto s = 1024;
        auto arr = array_ptr<ValueType>(s);
        CALL_KERNEL_SYNC((capture_shared_memory<s><<<1, s, s * sizeof(ValueType)>>>(arr)));
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, unified_array_ptr_trivial)
{
    using ValueType = int;
    static_assert(std::is_default_constructible_v<unified_array_ptr<ValueType>>);
    static_assert(std::is_constructible_v<unified_array_ptr<ValueType>, const unified_array_ptr<ValueType>&>);
    static_assert(std::is_constructible_v<unified_array_ptr<ValueType>, unified_array_ptr<ValueType>&&>);
    static_assert(std::is_constructible_v<unified_array_ptr<ValueType>, std::size_t>);
    static_assert(std::is_constructible_v<unified_array_ptr<ValueType>, std::size_t, default_init_tag>);
    static_assert(std::is_constructible_v<unified_array_ptr<ValueType>, std::size_t, ValueType>);
    static_assert(std::ranges::contiguous_range<std::vector<ValueType>> &&
                  std::is_constructible_v<unified_array_ptr<ValueType>, std::vector<ValueType>>);
    static_assert(!std::ranges::contiguous_range<std::list<ValueType>> &&
                  std::is_constructible_v<unified_array_ptr<ValueType>, std::list<ValueType>>);
    static_assert(std::is_constructible_v<unified_array_ptr<ValueType>, std::initializer_list<ValueType>>);
    static_assert(std::is_constructible_v<unified_array_ptr<ValueType>, array_ptr<ValueType>>);

    // deduction guide
    static_assert(std::same_as<decltype(unified_array_ptr(std::declval<std::vector<ValueType>>())),
                               unified_array_ptr<ValueType>>);
    static_assert(
        std::same_as<decltype(unified_array_ptr(std::declval<std::list<ValueType>>())), unified_array_ptr<ValueType>>);
    static_assert(std::same_as<decltype(unified_array_ptr(std::declval<std::array<ValueType, 1>>())),
                               unified_array_ptr<ValueType>>);
    static_assert(std::same_as<decltype(unified_array_ptr(std::declval<std::initializer_list<ValueType>>())),
                               unified_array_ptr<ValueType>>);

    {
        auto a = unified_array_ptr<ValueType>();
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));
        EXPECT_EQ(a.size(), 0);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
    }
    {
        constexpr auto s = std::size_t{1024};
        auto a = unified_array_ptr<ValueType>(s);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s));
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1025};
        auto a = unified_array_ptr<ValueType>(s, default_init);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1026};
        constexpr auto i = ValueType{42};
        auto a = unified_array_ptr<ValueType>(s, i);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, std::vector<ValueType>(s, i));
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1027};
        constexpr auto i = ValueType{43};
        auto v = std::vector<ValueType>(s, i);
        auto a = unified_array_ptr<ValueType>(v);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v2 = a.to<std::vector>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1028};
        constexpr auto i = ValueType{44};
        auto v = std::array<ValueType, s>();
        std::ranges::fill(v, i);
        auto a = unified_array_ptr<ValueType>(v);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v2 = a.to<std::array<ValueType, s>>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, v.size());
        EXPECT_EQ(v, v2);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto s = std::size_t{1029};
        constexpr auto i = ValueType{45};
        auto l = std::list<ValueType>(s, i);
        auto a = unified_array_ptr<ValueType>(l);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto l2 = a.to<std::list>();
        EXPECT_EQ(s, a.size());
        EXPECT_EQ(s, l.size());
        EXPECT_EQ(l, l2);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        auto a = unified_array_ptr<ValueType>{-1, -2, -3, -4, -5};
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto v = a.to<std::vector>();
        EXPECT_EQ(5, a.size());
        EXPECT_EQ(5, v.size());
        EXPECT_EQ((std::vector{-1, -2, -3, -4, -5}), v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        auto s = std::vector{-1, -2, -3, -4, -5};
        auto a = array_ptr<ValueType>(s);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType) * a.size()));

        auto u = unified_array_ptr<ValueType>(a);
        auto d = u.to<array_ptr>();
        auto v = d.to<std::vector>();
        EXPECT_EQ(s, v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, unified_array_struct)
{
    using ValueType = int;
    struct UserType
    {
        array_ptr<ValueType> a;
        unified_array_ptr<ValueType> u;
    };
    static_assert(!std::is_trivially_copyable_v<UserType>);

    {
        auto s1 = {1, 2, 3};
        auto s2 = {4, 5, 6};
        auto s3 = {7, 8, 9};
        auto s4 = {10, 11, 12};

        auto u = unified_array_ptr<UserType>{UserType{.a = s1, .u = s2}, UserType{.a = s3, .u = s4}};
        EXPECT_EQ(std::vector(s1), u[0].a.to<std::vector>());
        EXPECT_EQ(std::vector(s2), u[0].u.to<std::vector>());
        EXPECT_EQ(std::vector(s3), u[1].a.to<std::vector>());
        EXPECT_EQ(std::vector(s4), u[1].u.to<std::vector>());

        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(UserType) * u.size() + sizeof(ValueType) * u[0].a.size() +
                                        sizeof(ValueType) * u[0].u.size() + sizeof(ValueType) * u[1].a.size() +
                                        sizeof(ValueType) * u[1].u.size()));
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, array_conversion)
{
    using IntType = int;
    using FloatType = double;

    {
        [[maybe_unused]] auto u1 = unified_array_ptr(std::vector<std::vector<std::list<array_ptr<IntType>>>>());
        static_assert(std::same_as<unified_array_ptr<unified_array_ptr<unified_array_ptr<unified_array_ptr<IntType>>>>,
                                   decltype(u1)>);
        [[maybe_unused]] auto u2 = unified_array_ptr<unified_array_ptr<array_ptr<IntType>>>();
        static_assert(std::same_as<std::vector<std::vector<std::vector<IntType>>>, decltype(u2.to<std::vector>())>);
        [[maybe_unused]] auto u3 = unified_array_ptr<std::vector<array_ptr<IntType>>>();
        static_assert(std::same_as<std::vector<std::vector<array_ptr<IntType>>>, decltype(u3.to<std::vector>())>);
    }

    // conversion constructors
    {
        auto ui = unified_array_ptr<IntType>{1, 2, 3};
        auto uf = unified_array_ptr<FloatType>(ui);
        auto ui2 = unified_array_ptr<IntType>(uf);
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }
    {
        auto ai = array_ptr<IntType>{1, 2, 3};
        auto uf = unified_array_ptr<FloatType>(ai);
        auto ai2 = array_ptr<IntType>(uf);
        EXPECT_EQ(ai.to<std::vector>(), ai2.to<std::vector>());
    }
    {
        auto ui = unified_array_ptr<IntType>{1, 2, 3};
        auto af = array_ptr<FloatType>(ui);
        auto ui2 = unified_array_ptr<IntType>(af);
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }

    // conversion using `to` method
    {
        auto ai = array_ptr<IntType>{1, 2, 3};
        auto af = ai.to<array_ptr<FloatType>>();
        auto ai2 = af.to<array_ptr<IntType>>();
        EXPECT_EQ(ai.to<std::vector>(), ai2.to<std::vector>());
    }
    {
        auto ui = unified_array_ptr<IntType>{1, 2, 3};
        auto uf = ui.to<unified_array_ptr<FloatType>>();
        auto ui2 = uf.to<unified_array_ptr<IntType>>();
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }
    {
        auto ai = array_ptr<IntType>{1, 2, 3};
        auto uf = ai.to<unified_array_ptr<FloatType>>();
        auto ai2 = uf.to<array_ptr<IntType>>();
        EXPECT_EQ(ai.to<std::vector>(), ai2.to<std::vector>());
    }
    {
        auto ui = unified_array_ptr<IntType>{1, 2, 3};
        auto af = ui.to<array_ptr<FloatType>>();
        auto ui2 = af.to<unified_array_ptr<IntType>>();
        EXPECT_EQ(ui.to<std::vector>(), ui2.to<std::vector>());
    }
}

TEST(gpu_smart_ptr, value)
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
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType)));
        EXPECT_EQ(ValueType(), *v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto i = ValueType{13};
        auto v = value(i);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType)));
        EXPECT_EQ(ValueType(i), *v);

        CALL_KERNEL_SYNC((increment_value<<<1, 1>>>(v)));
        EXPECT_EQ(ValueType(i + 1), *v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, value_struct)
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
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(UserType)));
        EXPECT_EQ(i, v->i);
        EXPECT_EQ(d, v->d);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, value_capture_shared_ptr)
{
    {
        using ValueType = int;
        constexpr auto i = 1023;
        auto v = value<ValueType>(i);
        CALL_KERNEL_SYNC((capture_shared_memory<<<1, 1, sizeof(ValueType)>>>(v)));
        EXPECT_EQ(ValueType(i), *v);  // modification on GPU does not affect on CPU side
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, unified_value_trivial)
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
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType)));
        EXPECT_EQ(ValueType(), *v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));

    {
        constexpr auto i = ValueType{13};
        auto v = unified_value(i);
        EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(sizeof(ValueType)));
        EXPECT_EQ(ValueType(i), *v);

        CALL_KERNEL_SYNC((increment_value<<<1, 1>>>(v)));
        EXPECT_EQ(ValueType(i + 1), *v);
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

TEST(gpu_smart_ptr, unified_value_struct)
{
    using ValueType = int;
    struct UserType
    {
        array_ptr<ValueType> a;
        unified_array_ptr<ValueType> u;
    };
    static_assert(!std::is_trivially_copyable_v<UserType>);

    {
        auto s1 = {1, 2, 3};
        auto s2 = {4, 5, 6};

        auto u = unified_value{UserType{.a = s1, .u = s2}};
        EXPECT_EQ(std::vector(s1), u->a.to<std::vector>());
        EXPECT_EQ(std::vector(s2), u->u.to<std::vector>());

        EXPECT_TRUE(
            GPU_MEMORY_USAGE_EQ(sizeof(UserType) + sizeof(ValueType) * u->a.size() + sizeof(ValueType) * u->u.size()));
    }
    EXPECT_TRUE(GPU_MEMORY_USAGE_EQ(0));
}

template <typename T1, typename T2>
requires std::same_as<std::remove_cvref_t<T1>, int> && std::same_as<std::remove_cvref_t<T2>, double>
class custom_tuple_base : public std::tuple<T1, T2>
{
    using base = std::tuple<T1, T2>;
    using base::base;

public:
    __host__ __device__ decltype(auto) get_int() const { return std::get<0>(*this); }
    __host__ __device__ decltype(auto) get_double() const { return std::get<1>(*this); }
    using base::operator=;
};
using custom_tuple = custom_tuple_base<int, double>;

template <typename SoaPtr>
__global__ void test_structure_of_arrays_kernel(SoaPtr x)
{
    using array_type = SoaPtr;

    // check local copy
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        assert(static_cast<int>(i) == x[i].get_int());
        assert(i + 0.5 == x[i].get_double());
    }

    for (std::size_t i = 0; const auto& e : x)
    {
        assert(static_cast<int>(i) == e.get_int());
        assert(i + 0.5 == e.get_double());
        ++i;
    }

    // copy array to shared memory
    __shared__ unsigned char __align__(alignof(array_type)) array_buffer[sizeof(array_type)];
    __shared__ array_type* array_ptr;
    const auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() == 0)
    {
        // placement new on shared memory
        array_ptr = new (array_buffer) array_type(x);
    }
    block.sync();
    auto& xr = *array_ptr;

    // data is shared on global memory
    for (std::size_t i = 0; i < xr.size(); ++i)
    {
        assert(static_cast<int>(i) == xr[i].get_int());
        assert(i + 0.5 == xr[i].get_double());
    }

    // move array buffer to shared memory
    extern __shared__ char buffer[];
    auto next = xr.template move_to<0>(reinterpret_cast<int*>(buffer));
    xr.template move_to<1>(reinterpret_cast<double*>(next));
    block.sync();

    // data is on shared memory
    for (auto i = block.thread_rank(); i < xr.size(); i += block.size())
    {
        // check
        assert(static_cast<int>(i) == xr[i].get_int());
        assert(i + 0.5 == xr[i].get_double());

        // modify
        xr[i].get_int() += 1;
    }
    block.sync();

    // shared memory is modified
    for (std::size_t i = 0; i < xr.size(); ++i)
    {
        assert(static_cast<int>(i + 1) == xr[i].get_int());
    }

    // global memory is not modified
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        assert(static_cast<int>(i) == x[i].get_int());
    }
}

TEST(gpu_smart_ptr, soa_ptr)
{
    static_assert(std::ranges::sized_range<soa_ptr<int, double>>);
    static_assert(std::ranges::random_access_range<soa_ptr<int, double>>);
    static_assert(std::ranges::sized_range<soa_ptr<std::tuple<int, double>>>);
    static_assert(std::ranges::random_access_range<soa_ptr<std::tuple<int, double>>>);
    static_assert(std::ranges::sized_range<soa_ptr<custom_tuple>>);
    static_assert(std::ranges::random_access_range<soa_ptr<custom_tuple>>);

    // This should work: need std::common_reference customization point?
    // static_assert(std::ranges::input_range<const soa_ptr<custom_tuple>>);

    auto v = std::vector<custom_tuple>{{0, 0.5}, {1, 1.5}, {2, 2.5}, {3, 3.5}};
    {
        auto x = soa_ptr<int, double>(v);
        auto v2 = x.to<std::vector<std::tuple<int, double>>>();

        EXPECT_EQ(v.size(), x.size());
        EXPECT_EQ(v.size(), v2.size());

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_EQ(v[i].get_int(), std::get<0>(v2[i]));
            EXPECT_EQ(v[i].get_double(), std::get<1>(v2[i]));
        }
    }
    {
        auto x = soa_ptr(v);
        auto v2 = x.to<std::vector>();

        EXPECT_EQ(v.size(), x.size());
        EXPECT_EQ(v.size(), v2.size());

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_EQ(v[i].get_int(), v2[i].get_int());
            EXPECT_EQ(v[i].get_double(), v2[i].get_double());
        }

        auto shared_size = x.alignment<0>(0).second;
        EXPECT_EQ(sizeof(int) * x.size(), shared_size);
        shared_size = x.alignment<1>(shared_size).second;
        EXPECT_EQ((sizeof(double) + sizeof(int)) * x.size(), shared_size);

        CALL_KERNEL_SYNC((test_structure_of_arrays_kernel<<<1, gpu_max_threads, shared_size>>>(x)));
    }

    auto vi = std::vector<int>{0, 1, 2, 3};
    auto vd = std::vector<double>{0.5, 1.5, 2.5, 3.5};
    {
        auto x = soa_ptr<custom_tuple>(vi, vd);
        auto v2 = x.to<std::vector>();

        EXPECT_EQ(v.size(), x.size());
        EXPECT_EQ(v.size(), v2.size());

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_EQ(v[i].get_int(), v2[i].get_int());
            EXPECT_EQ(v[i].get_double(), v2[i].get_double());
        }

        auto shared_size = x.alignment<0>(0).second;
        EXPECT_EQ(sizeof(int) * x.size(), shared_size);
        shared_size = x.alignment<1>(shared_size).second;
        EXPECT_EQ((sizeof(double) + sizeof(int)) * x.size(), shared_size);

        CALL_KERNEL_SYNC((test_structure_of_arrays_kernel<<<1, gpu_max_threads, shared_size>>>(x)));
    }
}

TEST(gpu_smart_ptr, unified_soa_ptr)
{
    static_assert(std::ranges::sized_range<unified_soa_ptr<int, double>>);
    static_assert(std::ranges::random_access_range<unified_soa_ptr<int, double>>);
    static_assert(std::ranges::sized_range<unified_soa_ptr<std::tuple<int, double>>>);
    static_assert(std::ranges::random_access_range<unified_soa_ptr<std::tuple<int, double>>>);
    static_assert(std::ranges::sized_range<unified_soa_ptr<custom_tuple>>);
    static_assert(std::ranges::random_access_range<unified_soa_ptr<custom_tuple>>);

    // This should work: need std::common_reference customization point?
    // static_assert(std::ranges::input_range<const unified_soa_ptr<custom_tuple>>);

    auto v = std::vector<custom_tuple>{{0, 0.5}, {1, 1.5}, {2, 2.5}, {3, 3.5}};
    {
        auto x = unified_soa_ptr<int, double>(v);
        auto v2 = x.to<std::vector<std::tuple<int, double>>>();

        EXPECT_EQ(v.size(), x.size());
        EXPECT_EQ(v.size(), v2.size());

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_EQ(v[i].get_int(), std::get<0>(v2[i]));
            EXPECT_EQ(v[i].get_double(), std::get<1>(v2[i]));
        }
    }
    {
        auto x = unified_soa_ptr(v);
        auto v2 = x.to<std::vector>();

        EXPECT_EQ(v.size(), x.size());
        EXPECT_EQ(v.size(), v2.size());

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_EQ(v[i].get_int(), v2[i].get_int());
            EXPECT_EQ(v[i].get_double(), v2[i].get_double());
        }

        auto shared_size = x.alignment<0>(0).second;
        EXPECT_EQ(sizeof(int) * x.size(), shared_size);
        shared_size = x.alignment<1>(shared_size).second;
        EXPECT_EQ((sizeof(double) + sizeof(int)) * x.size(), shared_size);

        CALL_KERNEL_SYNC((test_structure_of_arrays_kernel<<<1, gpu_max_threads, shared_size>>>(x)));
    }

    auto vi = std::vector<int>{0, 1, 2, 3};
    auto vd = std::vector<double>{0.5, 1.5, 2.5, 3.5};
    {
        auto x = unified_soa_ptr<custom_tuple>(vi, vd);
        auto v2 = x.to<std::vector>();

        EXPECT_EQ(v.size(), x.size());
        EXPECT_EQ(v.size(), v2.size());

        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_EQ(v[i].get_int(), v2[i].get_int());
            EXPECT_EQ(v[i].get_double(), v2[i].get_double());
        }

        auto shared_size = x.alignment<0>(0).second;
        EXPECT_EQ(sizeof(int) * x.size(), shared_size);
        shared_size = x.alignment<1>(shared_size).second;
        EXPECT_EQ((sizeof(double) + sizeof(int)) * x.size(), shared_size);

        CALL_KERNEL_SYNC((test_structure_of_arrays_kernel<<<1, gpu_max_threads, shared_size>>>(x)));
    }
}

TEST(gpu_smart_ptr, jagged_array)
{
    static_assert(std::ranges::random_access_range<jagged_array<unified_array_ptr<int>>>);
    static_assert(std::ranges::random_access_range<const jagged_array<unified_array_ptr<int>>>);
    static_assert(std::ranges::random_access_range<jagged_array<unified_soa_ptr<custom_tuple>>>);
    // static_assert(std::ranges::random_access_range<const jagged_array<unified_soa_ptr<custom_tuple>>>); // FIXME

    {
        auto x = jagged_array<unified_array_ptr<int>>(std::vector{3, 1, 4, 3, 0});
        EXPECT_EQ(11, x.size());
        EXPECT_EQ((std::vector<std::size_t>{3, 1, 4, 3, 0}), x.get_sizes());
        EXPECT_EQ((std::vector<std::size_t>{0, 3, 4, 8, 11, 11}), x.get_offsets().to<std::vector>());
    }
    {
        auto x = jagged_array<unified_soa_ptr<int>>(std::vector{3, 1, 4, 3, 0});
        EXPECT_EQ(11, x.size());
        EXPECT_EQ((std::vector<std::size_t>{3, 1, 4, 3, 0}), x.get_sizes());
        EXPECT_EQ((std::vector<std::size_t>{0, 3, 4, 8, 11, 11}), x.get_offsets().to<std::vector>());
    }
    {
        auto x = jagged_array<unified_soa_ptr<custom_tuple>>(std::vector{3, 1, 4, 3, 0});
        EXPECT_EQ(11, x.size());
        EXPECT_EQ((std::vector<std::size_t>{3, 1, 4, 3, 0}), x.get_sizes());
        EXPECT_EQ((std::vector<std::size_t>{0, 3, 4, 8, 11, 11}), x.get_offsets().to<std::vector>());
    }
    {
        auto x = jagged_array<unified_array_ptr<int>>(std::vector{
            std::vector{0, 1, 2}, std::vector{3}, std::vector{4, 5, 6, 7}, std::vector{8, 9, 10}, std::vector<int>{}});
        EXPECT_EQ(11, x.size());
        EXPECT_EQ((std::vector<std::size_t>{3, 1, 4, 3, 0}), x.get_sizes());
        EXPECT_EQ((std::vector<std::size_t>{0, 3, 4, 8, 11, 11}), x.get_offsets().to<std::vector>());

        EXPECT_EQ((std::vector{0, 1, 2}), std::vector(x.row(0).begin(), x.row(0).end()));
        EXPECT_EQ((std::vector{3}), std::vector(x.row(1).begin(), x.row(1).end()));
        EXPECT_EQ((std::vector{4, 5, 6, 7}), std::vector(x.row(2).begin(), x.row(2).end()));
        EXPECT_EQ((std::vector{8, 9, 10}), std::vector(x.row(3).begin(), x.row(3).end()));
        EXPECT_EQ((std::vector<int>{}), std::vector(x.row(4).begin(), x.row(4).end()));

        for (std::size_t i = 0; i < 11; ++i)
        {
            EXPECT_EQ(i, x[i]);
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            for (std::size_t j = 0; const auto t : x.row(i))
            {
                EXPECT_EQ(s + j, t);
                ++j;
            }
            s += x.size(i);
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            auto it = x.begin(i);
            for (std::size_t j = 0; j < x.size(i); ++j, ++it)
            {
                EXPECT_EQ(s + j, *it);
            }
            s += x.size(i);
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            for (std::size_t j = 0; j < x.size(i); ++j)
            {
                EXPECT_EQ(s + j, (x[{i, j}]));
            }
            s += x.size(i);
        }
    }
    {
        auto v = std::vector<std::vector<std::tuple<int, double>>>{{{0, 10}, {1, 11}, {2, 12}},
                                                                   {{3, 13}},
                                                                   {{4, 14}, {5, 15}, {6, 16}, {7, 17}},
                                                                   {{8, 18}, {9, 19}, {10, 20}},
                                                                   {}};
        auto x = jagged_array<unified_soa_ptr<int, double>>(v);
        EXPECT_EQ(11, x.size());
        EXPECT_EQ((std::vector<std::size_t>{3, 1, 4, 3, 0}), x.get_sizes());
        EXPECT_EQ((std::vector<std::size_t>{0, 3, 4, 8, 11, 11}), x.get_offsets().to<std::vector>());

        for (std::size_t i = 0; i < 11; ++i)
        {
            EXPECT_EQ(i, std::get<0>(x[i]));
            EXPECT_EQ(10.0 + i, std::get<1>(x[i]));
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            for (std::size_t j = 0; const auto t : x.row(i))
            {
                EXPECT_EQ(s + j, std::get<0>(t));
                EXPECT_EQ(10.0 + s + j, std::get<1>(t));
                ++j;
            }
            s += x.size(i);
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            auto it = x.begin(i);
            for (std::size_t j = 0; j < x.size(i); ++j, ++it)
            {
                EXPECT_EQ(s + j, std::get<0>(*it));
                EXPECT_EQ(10.0 + s + j, std::get<1>(*it));
            }
            s += x.size(i);
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            for (std::size_t j = 0; j < x.size(i); ++j)
            {
                EXPECT_EQ(s + j, std::get<0>(x[{i, j}]));
                EXPECT_EQ(10.0 + s + j, std::get<1>(x[{i, j}]));
            }
            s += x.size(i);
        }
    }
    {
        auto v = std::vector<std::vector<custom_tuple>>{{{0, 10.0}, {1, 11.0}, {2, 12.0}},
                                                        {{3, 13.0}},
                                                        {{4, 14.0}, {5, 15.0}, {6, 16.0}, {7, 17.0}},
                                                        {{8, 18.0}, {9, 19.0}, {10, 20.0}},
                                                        {}};
        const auto x = jagged_array<unified_soa_ptr<custom_tuple>>(v);
        EXPECT_EQ(11, x.size());
        EXPECT_EQ((std::vector<std::size_t>{3, 1, 4, 3, 0}), x.get_sizes());
        EXPECT_EQ((std::vector<std::size_t>{0, 3, 4, 8, 11, 11}), x.get_offsets().to<std::vector>());

        for (std::size_t i = 0; i < 11; ++i)
        {
            EXPECT_EQ(i, x[i].get_int());
            EXPECT_EQ(10.0 + i, x[i].get_double());
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            for (std::size_t j = 0; const auto t : x.row(i))
            {
                EXPECT_EQ(s + j, t.get_int());
                EXPECT_EQ(10.0 + s + j, t.get_double());
                ++j;
            }
            s += x.size(i);
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            auto it = x.begin(i);
            for (std::size_t j = 0; j < x.size(i); ++j, ++it)
            {
                EXPECT_EQ(s + j, it->get_int());
                EXPECT_EQ(10.0 + s + j, it->get_double());
            }
            s += x.size(i);
        }

        for (std::size_t i = 0, s = 0; i < x.num_rows(); ++i)
        {
            for (std::size_t j = 0; j < x.size(i); ++j)
            {
                EXPECT_EQ(s + j, (x[{i, j}].get_int()));
                EXPECT_EQ(10.0 + s + j, (x[{i, j}].get_double()));
            }
            s += x.size(i);
        }
    }
}
