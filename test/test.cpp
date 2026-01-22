#include "gpu_array.hpp"

#include <gtest/gtest.h>
#include <list>

// NOLINTBEGIN
using namespace gpu_array;

// Example of custom tuple type derived from std::tuple
// You may need to specialize std::common_type and std::basic_common_reference to satisfy range concepts
template <typename... Ts>
class custom_tuple_example : public std::tuple<Ts...>
{
    using base = std::tuple<Ts...>;
    using base::base;

public:
    template <std::size_t N>
    __host__ decltype(auto) get_string() const
    {
        return std::to_string(std::get<N>(*this));
    }
    using base::operator=;
    template <typename... Us>
    custom_tuple_example(const custom_tuple_example<Us...>& that) : base(that)
    {
    }
};

template <class... TTypes, class... UTypes>
requires requires { typename custom_tuple_example<std::common_type_t<TTypes, UTypes>...>; }
struct std::common_type<custom_tuple_example<TTypes...>, custom_tuple_example<UTypes...>>
{
    using type = custom_tuple_example<std::common_type_t<TTypes, UTypes>...>;
};

template <class... TTypes, class... UTypes, template <class> class TQual, template <class> class UQual>
requires requires { typename custom_tuple_example<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>; }
struct std::basic_common_reference<custom_tuple_example<TTypes...>, custom_tuple_example<UTypes...>, TQual, UQual>
{
    using type = custom_tuple_example<std::common_reference_t<TQual<TTypes>, UQual<UTypes>>...>;
};

TEST(Array, Construction)
{
    using ValueType = int;
    using OtherValueType = double;

    // initialization
    {
        auto arr = array<ValueType>();
        EXPECT_EQ(arr.size(), 0);
        EXPECT_EQ(arr.data(), nullptr);
    }
    {
        auto arr = array<ValueType>(0);
        EXPECT_EQ(arr.size(), 0);
        EXPECT_EQ(arr.data(), nullptr);
    }
    {
        auto arr = array<ValueType>(10);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
    }
    {
        auto arr = array<ValueType>(10, default_init);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
    }
    {
        auto arr = array<ValueType>(10, ValueType{42});
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto vec = arr.to<std::vector>();
        for (const auto v : vec)
        {
            EXPECT_EQ(v, 42);
        }
    }
    {
        // initializer list
        auto arr = array<ValueType>{10, ValueType{42}};
        EXPECT_EQ(arr.size(), 2);
        EXPECT_NE(arr.data(), nullptr);
        const auto vec = arr.to<std::vector>();
        EXPECT_EQ(vec[0], 10);
        EXPECT_EQ(vec[1], 42);
    }

    // copy construction and copy assignment (just copys pointer)
    {
        auto arr1 = array<ValueType>(10, ValueType{7});
        EXPECT_EQ(arr1.use_count(), 1);
        {
            auto arr2 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr2.use_count(), 2);
            EXPECT_EQ(arr2.size(), 10);
            EXPECT_EQ(arr2.data(), arr1.data());
        }
        EXPECT_EQ(arr1.use_count(), 1);

        {
            auto arr3 = array<ValueType>(10, ValueType{8});
            EXPECT_EQ(arr3.use_count(), 1);
            const auto old_ptr = arr3.data();
            arr3 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr3.use_count(), 2);
            EXPECT_EQ(arr3.size(), 10);
            EXPECT_EQ(arr3.data(), arr1.data());
            EXPECT_NE(arr3.data(), old_ptr);
        }
        EXPECT_EQ(arr1.use_count(), 1);
    }

    // construction from contiguous range
    {
        const auto vec = std::vector<ValueType>(10, ValueType{13});
        auto arr = array<ValueType>(vec);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto arr_vec = arr.to<std::vector>();
        for (const auto v : arr_vec) EXPECT_EQ(v, 13);
    }
    {
        const auto vec = std::vector<ValueType>(10, ValueType{13});
        auto arr = array<OtherValueType>(vec);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto arr_vec = arr.to<std::vector>();
        for (const auto v : arr_vec) EXPECT_EQ(v, OtherValueType{13});
    }

    // construction from non-contiguous range
    {
        const auto lst = std::list<ValueType>(10, ValueType{13});
        auto arr = array<ValueType>(lst);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto arr_vec = arr.to<std::vector<ValueType>>();
        for (const auto v : arr_vec) EXPECT_EQ(v, 13);
    }
    {
        const auto lst = std::list<ValueType>(10, ValueType{13});
        auto arr = array<OtherValueType>(lst);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto arr_vec = arr.to<std::vector>();
        for (const auto v : arr_vec) EXPECT_EQ(v, OtherValueType{13});
    }

    // copy construction is not allowed from GPU ptr with different value type
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(!std::is_constructible_v<array<OtherValueType>, const array<ValueType>&>);
        static_assert(!std::is_constructible_v<array<OtherValueType>, const managed_array<ValueType>&>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }

    // size check for construction from range
    {
        auto vec = std::vector<ValueType>(256, 1);
        EXPECT_THROW(({ [[maybe_unused]] auto arr = array<ValueType, std::uint8_t>(vec); }), std::runtime_error);
    }

    // deduction guides
    {
        auto arr = array(std::vector<int>(10, 1));
        static_assert(std::is_same_v<decltype(arr), array<int>>);
    }
}

TEST(Array, Export)
{
    using ValueType = int;
    using DstValueType = double;

    // export to range with same value type
    {
        const auto vec = std::vector<ValueType>(10, ValueType{21});
        auto arr = array<ValueType>(vec);
        const auto arr_vec = arr.to<std::vector>();
        EXPECT_EQ(arr_vec, vec);
    }
    {
        const auto lst = std::list<ValueType>(10, ValueType{21});
        auto arr = array<ValueType>(lst);
        const auto arr_list = arr.to<std::list>();
        EXPECT_EQ(arr_list, lst);
    }
    {
        auto stdarr = std::array<ValueType, 10>();
        for (auto& v : stdarr) v = 21;
        auto arr = array<ValueType>(stdarr);
        const auto arr_stdarr = arr.to<std::array<ValueType, 10>>();
        EXPECT_EQ(arr_stdarr, stdarr);
    }

    // export to range with different value type
    {
        const auto vec = std::vector<ValueType>(10, ValueType{31});
        auto arr = array<ValueType>(vec);
        const auto arr_vec = arr.to<std::vector<DstValueType>>();
        EXPECT_EQ(arr_vec.size(), vec.size());
        for (std::size_t i = 0; i < arr_vec.size(); ++i)
        {
            EXPECT_EQ(arr_vec[i], static_cast<DstValueType>(vec[i]));
        }
    }
    {
        const auto lst = std::list<ValueType>(10, ValueType{31});
        auto arr = array<ValueType>(lst);
        const auto arr_list = arr.to<std::list<DstValueType>>();
        EXPECT_EQ(arr_list.size(), lst.size());
        {
            auto it1 = arr_list.begin();
            auto it2 = lst.begin();
            for (; it1 != arr_list.end() && it2 != lst.end(); ++it1, ++it2)
            {
                EXPECT_EQ(*it1, static_cast<DstValueType>(*it2));
            }
        }
    }
    {
        auto stdarr = std::array<ValueType, 10>();
        for (auto& v : stdarr) v = 31;
        auto arr = array<ValueType>(stdarr);
        const auto arr_stdarr = arr.to<std::array<DstValueType, 10>>();
        for (std::size_t i = 0; i < arr_stdarr.size(); ++i)
        {
            EXPECT_EQ(arr_stdarr[i], static_cast<DstValueType>(stdarr[i]));
        }
    }

    // export to GPU ptr with the same value type
    {
        auto arr = array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<array>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        EXPECT_NE(arr2.data(), arr.data());
        const auto arr_vec = arr.to<std::vector>();
        const auto arr2_vec = arr2.to<std::vector>();
        EXPECT_EQ(arr_vec, arr2_vec);
        for (const auto& v : arr2_vec) EXPECT_EQ(v, 41);
    }
    {
        auto arr = array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<managed_array>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        EXPECT_NE(arr2.data(), arr.data());
        for (const auto& v : arr2) EXPECT_EQ(v, 41);
    }

    // export to GPU ptr with different value type
    {
        auto arr = array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<array<DstValueType>>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        const auto arr_vec = arr.to<std::vector<DstValueType>>();
        const auto arr2_vec = arr2.to<std::vector>();
        EXPECT_EQ(arr_vec, arr2_vec);
        for (const auto& v : arr2_vec) EXPECT_EQ(v, DstValueType{41});
    }
    {
        auto arr = array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<managed_array<DstValueType>>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        for (const auto& v : arr2) EXPECT_EQ(v, DstValueType{41});
    }

    // static_cast conversion (alias of to<...> except gpu ptr classes)
    {
        const auto vec = std::vector<ValueType>(10, ValueType{41});
        auto arr = array<ValueType>(vec);
        EXPECT_EQ(arr.to<std::vector>(), static_cast<std::vector<ValueType>>(arr));
        EXPECT_EQ(arr.to<std::vector<DstValueType>>(), static_cast<std::vector<DstValueType>>(arr));
    }
}

TEST(Array, RangeInterface)
{
    using ValueType = int;

    // concepts
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    static_assert(std::ranges::range<array<int>>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<array<int>>);
#else
    static_assert(!std::ranges::borrowed_range<array<int>>);
#endif
    static_assert(std::ranges::view<array<int>>);
    static_assert(std::ranges::output_range<array<int>, int>);
    static_assert(std::ranges::input_range<array<int>>);
    static_assert(std::ranges::forward_range<array<int>>);
    static_assert(std::ranges::bidirectional_range<array<int>>);
    static_assert(std::ranges::random_access_range<array<int>>);
    static_assert(std::ranges::sized_range<array<int>>);
    static_assert(std::ranges::contiguous_range<array<int>>);
    static_assert(std::ranges::common_range<array<int>>);
    static_assert(std::ranges::viewable_range<array<int>>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    // container-like interface
    {
        auto arr = array<ValueType>(10);
        EXPECT_NE(arr.data(), nullptr);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_FALSE(arr.empty());
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        EXPECT_EQ(&arr[0], arr.data());
        EXPECT_EQ(&arr.front(), &arr[0]);
        EXPECT_EQ(&arr.back(), &arr[9]);
        EXPECT_EQ(arr.begin(), &arr[0]);
        EXPECT_EQ(arr.end() - 1, &arr[9]);
        EXPECT_EQ(arr.rbegin(), std::reverse_iterator(arr.end()));
        EXPECT_EQ(arr.rend(), std::reverse_iterator(arr.begin()));
        EXPECT_EQ(arr.cbegin(), &arr[0]);
        EXPECT_EQ(arr.cend() - 1, &arr[9]);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
        arr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        auto vec = arr.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i) EXPECT_EQ(vec[i], static_cast<ValueType>(i));
    }
}

TEST(Array, SmartPointerInterface)
{
    using ValueType = int;

    // construction from raw device pointer
    ValueType* ptr = nullptr;
    GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), 3 * sizeof(ValueType)));
    auto arr = array(ptr, 3);
    EXPECT_EQ(arr.data(), ptr);
    EXPECT_EQ(arr.use_count(), 1);
    auto arr2 = arr;
    EXPECT_EQ(arr.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // reset with another raw device pointer
    GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), 19 * sizeof(ValueType)));
    arr.reset(ptr, 19);
    EXPECT_EQ(arr.data(), ptr);
    EXPECT_EQ(arr.use_count(), 1);
    EXPECT_EQ(arr2.use_count(), 1);
    EXPECT_EQ(static_cast<bool>(arr), true);

    // reset
    arr.reset();
    EXPECT_EQ(arr.data(), nullptr);
    EXPECT_EQ(arr.use_count(), 0);
    EXPECT_EQ(static_cast<bool>(arr), false);
}

TEST(ManagedArray, Construction)
{
    using ValueType = int;
    using OtherValueType = double;

    // initialization
    {
        auto arr = managed_array<ValueType>();
        EXPECT_EQ(arr.size(), 0);
        EXPECT_EQ(arr.data(), nullptr);
    }
    {
        auto arr = managed_array<ValueType>(0);
        EXPECT_EQ(arr.size(), 0);
        EXPECT_EQ(arr.data(), nullptr);
    }
    {
        auto arr = managed_array<ValueType>(10);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        for (const auto& v : arr) EXPECT_EQ(v, ValueType{});
    }
    {
        auto arr = managed_array<ValueType>(10, default_init);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
    }
    {
        auto arr = managed_array<ValueType>(10, ValueType{42});
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        for (const auto& v : arr) EXPECT_EQ(v, ValueType{42});
    }
    {
        // initializer list
        auto arr = managed_array<ValueType>{10, ValueType{42}};
        EXPECT_EQ(arr.size(), 2);
        EXPECT_NE(arr.data(), nullptr);
        EXPECT_EQ(arr[0], 10);
        EXPECT_EQ(arr[1], 42);
    }
    {
        // nested array
        auto arr = managed_array<managed_array<ValueType>>(2, managed_array<ValueType>{1, 2, 3});
        EXPECT_EQ(arr.size(), 2);
        for (const auto& v : arr)
        {
            EXPECT_EQ(v.size(), 3);
            EXPECT_EQ(v[0], 1);
            EXPECT_EQ(v[1], 2);
            EXPECT_EQ(v[2], 3);
        }
    }
    {
        // non-trivial value type
        struct NonTrivial
        {
            ValueType x;
            OtherValueType y;
            managed_array<ValueType> arr;
            NonTrivial() = default;
            explicit NonTrivial(ValueType v, OtherValueType w, std::size_t s) : x(v), y(w), arr(s, 99) {}
        };

        auto arr = managed_array<NonTrivial>(3, NonTrivial{7, 3.14, 5});
        EXPECT_EQ(arr.size(), 3);
        for (const auto& v : arr)
        {
            EXPECT_EQ(v.x, 7);
            EXPECT_EQ(v.y, 3.14);
            EXPECT_EQ(v.arr.size(), 5);
            for (const auto u : v.arr) EXPECT_EQ(u, 99);
        }
    }

    // copy construction and copy assignment (just copys pointer)
    {
        auto arr1 = managed_array<ValueType>(10, ValueType{7});
        EXPECT_EQ(arr1.use_count(), 1);
        {
            auto arr2 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr2.use_count(), 2);
            EXPECT_EQ(arr2.size(), 10);
            EXPECT_EQ(arr2.data(), arr1.data());
        }
        EXPECT_EQ(arr1.use_count(), 1);

        {
            auto arr3 = managed_array<ValueType>(10, ValueType{8});
            EXPECT_EQ(arr3.use_count(), 1);
            const auto old_ptr = arr3.data();
            arr3 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr3.use_count(), 2);
            EXPECT_EQ(arr3.size(), 10);
            EXPECT_EQ(arr3.data(), arr1.data());
            EXPECT_NE(arr3.data(), old_ptr);
        }
        EXPECT_EQ(arr1.use_count(), 1);
    }

    // construction from contiguous range
    {
        const auto vec = std::vector<ValueType>(10, ValueType{13});
        auto arr = managed_array<ValueType>(vec);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto arr_vec = arr.to<std::vector>();
        for (const auto v : arr_vec) EXPECT_EQ(v, 13);
    }
    {
        const auto vec = std::vector<ValueType>(10, ValueType{13});
        auto arr = managed_array<OtherValueType>(vec);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        for (const auto v : arr) EXPECT_EQ(v, OtherValueType{13});
    }

    // construction from non-contiguous range
    {
        const auto lst = std::list<ValueType>(10, ValueType{13});
        auto arr = managed_array<ValueType>(lst);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto arr_vec = arr.to<std::vector>();
        for (const auto v : arr_vec) EXPECT_EQ(v, 13);
    }
    {
        const auto lst = std::list<ValueType>(10, ValueType{13});
        auto arr = managed_array<OtherValueType>(lst);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_NE(arr.data(), nullptr);
        const auto arr_vec = arr.to<std::vector>();
        for (const auto v : arr_vec) EXPECT_EQ(v, OtherValueType{13});
    }

    // construction from nested range
    {
        auto v1 = std::vector<ValueType>(1, 1);
        auto v2 = std::vector<ValueType>(2, 2);
        auto v3 = std::vector<ValueType>(3, 3);
        const auto vec = std::vector<std::vector<ValueType>>{v1, v2, v3};
        auto arr = managed_array(vec);
        EXPECT_EQ(arr.size(), 3);
        EXPECT_NE(arr.data(), nullptr);
        EXPECT_EQ(arr[0].size(), 1);
        EXPECT_EQ(arr[1].size(), 2);
        EXPECT_EQ(arr[2].size(), 3);
        for (const auto v : arr[0]) EXPECT_EQ(v, 1);
        for (const auto v : arr[1]) EXPECT_EQ(v, 2);
        for (const auto v : arr[2]) EXPECT_EQ(v, 3);
    }

    // copy construction is not allowed from unmanaged GPU ptr with different value type
    {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(!std::is_constructible_v<managed_array<OtherValueType>, const array<ValueType>&>);
        static_assert(!std::is_constructible_v<managed_array<OtherValueType>, const managed_array<ValueType>&>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    }

    // size check for construction from range
    {
        auto vec = std::vector<ValueType>(256, 1);
        EXPECT_THROW(({ [[maybe_unused]] auto arr = managed_array<ValueType, std::uint8_t>(vec); }),
                     std::runtime_error);
    }

    // deduction guides
    {
        auto arr = managed_array(std::vector<int>(10, 1));
        static_assert(std::is_same_v<decltype(arr), managed_array<int>>);
    }
}

TEST(ManagedArray, Export)
{
    using ValueType = int;
    using DstValueType = double;

    // export to range with same value type
    {
        const auto vec = std::vector<ValueType>(10, ValueType{21});
        auto arr = managed_array<ValueType>(vec);
        const auto arr_vec = arr.to<std::vector>();
        EXPECT_EQ(arr_vec, vec);
    }
    {
        const auto lst = std::list<ValueType>(10, ValueType{21});
        auto arr = managed_array<ValueType>(lst);
        const auto arr_list = arr.to<std::list>();
        EXPECT_EQ(arr_list, lst);
    }
    {
        auto stdarr = std::array<ValueType, 10>();
        for (auto& v : stdarr) v = 21;
        auto arr = managed_array<ValueType>(stdarr);
        const auto arr_stdarr = arr.to<std::array<ValueType, 10>>();
        EXPECT_EQ(arr_stdarr, stdarr);
    }

    // export to range with different value type
    {
        const auto vec = std::vector<ValueType>(10, ValueType{31});
        auto arr = managed_array<ValueType>(vec);
        const auto arr_vec = arr.to<std::vector<DstValueType>>();
        EXPECT_EQ(arr_vec.size(), vec.size());
        for (std::size_t i = 0; i < arr_vec.size(); ++i)
        {
            EXPECT_EQ(arr_vec[i], static_cast<DstValueType>(vec[i]));
        }
    }
    {
        const auto lst = std::list<ValueType>(10, ValueType{31});
        auto arr = managed_array<ValueType>(lst);
        const auto arr_list = arr.to<std::list<DstValueType>>();
        EXPECT_EQ(arr_list.size(), lst.size());
        {
            auto it1 = arr_list.begin();
            auto it2 = lst.begin();
            for (; it1 != arr_list.end() && it2 != lst.end(); ++it1, ++it2)
            {
                EXPECT_EQ(*it1, static_cast<DstValueType>(*it2));
            }
        }
    }
    {
        auto stdarr = std::array<ValueType, 10>();
        for (auto& v : stdarr) v = 31;
        auto arr = managed_array<ValueType>(stdarr);
        const auto arr_stdarr = arr.to<std::array<DstValueType, 10>>();
        for (std::size_t i = 0; i < arr_stdarr.size(); ++i)
        {
            EXPECT_EQ(arr_stdarr[i], static_cast<DstValueType>(stdarr[i]));
        }
    }

    // export to GPU ptr with the same value type
    {
        auto arr = managed_array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<array>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        EXPECT_NE(arr2.data(), arr.data());
        const auto arr_vec = arr.to<std::vector>();
        const auto arr2_vec = arr2.to<std::vector>();
        EXPECT_EQ(arr_vec, arr2_vec);
        for (const auto& v : arr2_vec) EXPECT_EQ(v, 41);
    }
    {
        auto arr = managed_array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<managed_array>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        EXPECT_NE(arr2.data(), arr.data());
        for (const auto& v : arr2) EXPECT_EQ(v, 41);
    }

    // export to GPU ptr with different value type
    {
        auto arr = managed_array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<array<DstValueType>>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        const auto arr_vec = arr.to<std::vector<DstValueType>>();
        const auto arr2_vec = arr2.to<std::vector>();
        EXPECT_EQ(arr_vec, arr2_vec);
        for (const auto& v : arr2_vec) EXPECT_EQ(v, DstValueType{41});
    }
    {
        auto arr = managed_array<ValueType>(10, ValueType{41});
        auto arr2 = arr.to<managed_array<DstValueType>>();
        EXPECT_EQ(arr.use_count(), 1);
        EXPECT_EQ(arr2.use_count(), 1);
        EXPECT_EQ(arr2.size(), arr.size());
        for (const auto& v : arr2) EXPECT_EQ(v, DstValueType{41});
    }

    // export to nested range
    {
        auto v1 = std::vector<ValueType>(1, 1);
        auto v2 = std::vector<ValueType>(2, 2);
        auto v3 = std::vector<ValueType>(3, 3);
        const auto vec = std::vector<std::vector<ValueType>>{v1, v2, v3};
        auto arr = managed_array(vec);
        const auto arr_vec = arr.to<std::vector>();
        EXPECT_EQ(arr_vec.size(), vec.size());
        EXPECT_EQ(arr_vec, vec);
    }

    // export non-trivial value range
    {
        struct NonTrivial
        {
            using OtherValueType = double;
            ValueType x;
            OtherValueType y;
            managed_array<ValueType> arr;
            NonTrivial() = default;
            explicit NonTrivial(ValueType v, OtherValueType w, std::size_t s) : x(v), y(w), arr(s, 99) {}
        };
        auto arr = managed_array<NonTrivial>(3, NonTrivial{7, 3.14, 5});
        const auto arr_vec = arr.to<std::vector>();
        EXPECT_EQ(arr_vec.size(), 3);
        for (const auto& v : arr_vec)
        {
            EXPECT_EQ(v.x, 7);
            EXPECT_EQ(v.y, 3.14);
            EXPECT_EQ(v.arr.size(), 5);
            for (const auto u : v.arr) EXPECT_EQ(u, 99);
        }
    }

    // static_cast conversion (alias of to<...> except gpu ptr classes)
    {
        const auto vec = std::vector<ValueType>(10, ValueType{41});
        auto arr = managed_array<ValueType>(vec);
        EXPECT_EQ(arr.to<std::vector>(), static_cast<std::vector<ValueType>>(arr));
        EXPECT_EQ(arr.to<std::vector<DstValueType>>(), static_cast<std::vector<DstValueType>>(arr));
    }
}

TEST(ManagedArray, RangeInterface)
{
    using ValueType = int;

    // concepts
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    static_assert(std::ranges::range<managed_array<ValueType>>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<managed_array<ValueType>>);
#else
    static_assert(!std::ranges::borrowed_range<managed_array<ValueType>>);
#endif
    static_assert(std::ranges::view<managed_array<ValueType>>);
    static_assert(std::ranges::output_range<managed_array<ValueType>, ValueType>);
    static_assert(std::ranges::input_range<managed_array<ValueType>>);
    static_assert(std::ranges::forward_range<managed_array<ValueType>>);
    static_assert(std::ranges::bidirectional_range<managed_array<ValueType>>);
    static_assert(std::ranges::random_access_range<managed_array<ValueType>>);
    static_assert(std::ranges::sized_range<managed_array<ValueType>>);
    static_assert(std::ranges::contiguous_range<managed_array<ValueType>>);
    static_assert(std::ranges::common_range<managed_array<ValueType>>);
    static_assert(std::ranges::viewable_range<managed_array<ValueType>>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    // container-like interface
    {
        auto arr = managed_array<ValueType>(10);
        EXPECT_NE(arr.data(), nullptr);
        EXPECT_EQ(arr.size(), 10);
        EXPECT_FALSE(arr.empty());
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        EXPECT_EQ(&arr[0], arr.data());
        EXPECT_EQ(&arr.front(), &arr[0]);
        EXPECT_EQ(&arr.back(), &arr[9]);
        EXPECT_EQ(arr.begin(), &arr[0]);
        EXPECT_EQ(arr.end() - 1, &arr[9]);
        EXPECT_EQ(arr.rbegin(), std::reverse_iterator(arr.end()));
        EXPECT_EQ(arr.rend(), std::reverse_iterator(arr.begin()));
        EXPECT_EQ(arr.cbegin(), &arr[0]);
        EXPECT_EQ(arr.cend() - 1, &arr[9]);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
        arr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        for (std::size_t i = 0; i < arr.size(); ++i) EXPECT_EQ(arr[i], static_cast<ValueType>(i));
    }
}

TEST(ManagedArray, SmartPointerInterface)
{
    using ValueType = int;

    // construction from raw device pointer
    ValueType* ptr = nullptr;
    GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), 3 * sizeof(ValueType)));
    auto arr = managed_array<ValueType>(ptr, 3);
    EXPECT_EQ(arr.data(), ptr);
    EXPECT_EQ(arr.use_count(), 1);
    auto arr2 = arr;
    EXPECT_EQ(arr.use_count(), 2);
    EXPECT_EQ(arr2.use_count(), 2);

    // reset with another raw device pointer
    GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), 19 * sizeof(ValueType)));
    arr.reset(ptr, 19);
    EXPECT_EQ(arr.data(), ptr);
    EXPECT_EQ(arr.use_count(), 1);
    EXPECT_EQ(arr2.use_count(), 1);
    EXPECT_EQ(static_cast<bool>(arr), true);

    // reset
    arr.reset();
    EXPECT_EQ(arr.data(), nullptr);
    EXPECT_EQ(arr.use_count(), 0);
    EXPECT_EQ(static_cast<bool>(arr), false);
}

TEST(ManagedArray, MemoryManagement)
{
    // create nested managed_array
    auto elms = std::vector<managed_array<int>>();
    for (std::size_t i = 0; i < 10; ++i) elms.emplace_back(10, i);
    auto arr = managed_array(elms);
    // all elements of nested arrays should have different pointers
    EXPECT_FALSE(std::all_of(std::next(std::ranges::begin(arr)), std::ranges::end(arr),
                             [&arr](auto& a) { return a.data() == std::ranges::begin(arr)->data(); }));

    // check if prefetching works without errors
    {
        // prefetch 4 elements from the 3rd to device 0 with stream 0 recursively.
        arr.prefetch(3, 4, 0, 0, true);
        // prefetch 4 elements from the 3rd to current device with stream 0 recurslively.
        arr.prefetch(3, 4, 0, true);
        // all elements to device 0 with stream 0 recursively.
        arr.prefetch(0, 0, true);
        // all elements to current device with stream 0 recursively.
        arr.prefetch(0, true);
        // all elements to current device with stream 0 recursively.
        arr.prefetch();

        // prefetch 4 elements from the 3rd to host with stream 0 recursively.
        arr.prefetch_to_cpu(3, 4, 0, true);
        // all elements to host with stream 0 recursively.
        arr.prefetch_to_cpu(0, true);
        // all elements to host with stream 0 recursively.
        arr.prefetch_to_cpu();
    }

    // check if mem advice works without errors
    {
        // set memory advice for 4 elements from the 3rd to preferred location for device 0 recursively.
        arr.mem_advise(3, 4, api::gpuMemoryAdvise::SetPreferredLocation, 0, true);
        // set memory advice for 4 elements from the 3rd to preferred location for current device recursively.
        arr.mem_advise(3, 4, api::gpuMemoryAdvise::SetPreferredLocation, true);
        // set memory advice for all elements to preferred location for device 0 recursively.
        arr.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation, 0, true);
        // set memory advice for all elements to preferred location for current device recursively.
        arr.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation, true);
        // set memory advice for all elements to preferred location for current device recursively.
        arr.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation);

        // set memory advice for 4 elements from the 3rd to read mostly for host recursively.
        arr.mem_advise_to_cpu(3, 4, api::gpuMemoryAdvise::SetReadMostly, true);
        // set memory advice for all elements to read mostly for host recursively.
        arr.mem_advise_to_cpu(api::gpuMemoryAdvise::SetReadMostly, true);
        // set memory advice for all elements to read mostly for host recursively.
        arr.mem_advise_to_cpu(api::gpuMemoryAdvise::SetReadMostly);
    }
}

TEST(Value, Construction)
{
    using ValueType = int;

    // initialization
    {
        auto val = value<ValueType>();
        EXPECT_EQ(val.get(), nullptr);
    }
    {
        auto val = value<ValueType>(ValueType{});
        EXPECT_EQ(*val, 0);
    }
    {
        auto val = value<ValueType>(42.5);
        EXPECT_EQ(*val, 42);
    }
    {
        auto val = value<ValueType>(default_init);
        EXPECT_NE(val.get(), nullptr);
    }

    // copy construction and copy assignment (just copys pointer)
    {
        auto val1 = value<ValueType>(ValueType{7});
        EXPECT_EQ(val1.use_count(), 1);
        {
            auto val2 = val1;
            EXPECT_EQ(val1.use_count(), 2);
            EXPECT_EQ(val2.use_count(), 2);
            EXPECT_EQ(val2.get(), val1.get());
        }
        EXPECT_EQ(val1.use_count(), 1);

        {
            auto val3 = value<ValueType>(ValueType{8});
            EXPECT_EQ(val3.use_count(), 1);
            const auto old_ptr = val3.get();
            val3 = val1;
            EXPECT_EQ(val1.use_count(), 2);
            EXPECT_EQ(val3.use_count(), 2);
            EXPECT_EQ(val3.get(), val1.get());
            EXPECT_NE(val3.get(), old_ptr);
        }
        EXPECT_EQ(val1.use_count(), 1);
    }
}

TEST(Value, SmartPointerInterface)
{
    using ValueType = int;

    // construction from raw device pointer
    {
        ValueType* ptr = nullptr;
        GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(ValueType)));
        auto val = value(ptr);
        EXPECT_EQ(val.get(), ptr);
        EXPECT_EQ(val.use_count(), 1);
        auto val2 = val;
        EXPECT_EQ(val.use_count(), 2);
        EXPECT_EQ(val2.use_count(), 2);

        // reset with another raw device pointer
        GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(ValueType)));
        val.reset(ptr);
        EXPECT_EQ(val.get(), ptr);
        EXPECT_EQ(val.use_count(), 1);
        EXPECT_EQ(val2.use_count(), 1);
        EXPECT_EQ(static_cast<bool>(val), true);

        // reset
        val.reset();
        EXPECT_EQ(val.get(), nullptr);
        EXPECT_EQ(val.use_count(), 0);
        EXPECT_EQ(static_cast<bool>(val), false);
    }

    // operators
    {
        auto val = value<ValueType>(42);
        EXPECT_EQ(*val, 42);
    }
    {
        struct
        {
            ValueType x = 4;
        } obj;
        auto val_ptr = value(obj);
        EXPECT_EQ(val_ptr.use_count(), 1);
        EXPECT_EQ(val_ptr->x, 4);
    }
}

TEST(ManagedValue, Construction)
{
    using ValueType = int;

    // initialization
    {
        auto val = managed_value<ValueType>();
        EXPECT_EQ(val.get(), nullptr);
    }
    {
        auto val = managed_value<ValueType>(ValueType{});
        EXPECT_EQ(*val, 0);
    }
    {
        auto val = managed_value<ValueType>(42.5);
        EXPECT_EQ(*val, 42);
    }
    {
        auto val = managed_value<ValueType>(default_init);
        EXPECT_NE(val.get(), nullptr);
    }

    // copy construction and copy assignment (just copys pointer)
    {
        auto val1 = managed_value<ValueType>(ValueType{7});
        EXPECT_EQ(val1.use_count(), 1);
        {
            auto val2 = val1;
            EXPECT_EQ(val1.use_count(), 2);
            EXPECT_EQ(val2.use_count(), 2);
            EXPECT_EQ(val2.get(), val1.get());
        }
        EXPECT_EQ(val1.use_count(), 1);

        {
            auto val3 = managed_value<ValueType>(ValueType{8});
            EXPECT_EQ(val3.use_count(), 1);
            const auto old_ptr = val3.get();
            val3 = val1;
            EXPECT_EQ(val1.use_count(), 2);
            EXPECT_EQ(val3.use_count(), 2);
            EXPECT_EQ(val3.get(), val1.get());
            EXPECT_NE(val3.get(), old_ptr);
        }
        EXPECT_EQ(val1.use_count(), 1);
    }
}

TEST(ManagedValue, SmartPointerInterface)
{
    using ValueType = int;

    // construction from raw device pointer
    {
        ValueType* ptr = nullptr;
        GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(ValueType)));
        auto val = managed_value(ptr);
        EXPECT_EQ(val.get(), ptr);
        EXPECT_EQ(val.use_count(), 1);
        auto val2 = val;
        EXPECT_EQ(val.use_count(), 2);
        EXPECT_EQ(val2.use_count(), 2);

        // reset with another raw device pointer
        GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(ValueType)));
        val.reset(ptr);
        EXPECT_EQ(val.get(), ptr);
        EXPECT_EQ(val.use_count(), 1);
        EXPECT_EQ(val2.use_count(), 1);
        EXPECT_EQ(static_cast<bool>(val), true);

        // reset
        val.reset();
        EXPECT_EQ(val.get(), nullptr);
        EXPECT_EQ(val.use_count(), 0);
        EXPECT_EQ(static_cast<bool>(val), false);
    }

    // operators
    {
        auto val = managed_value<ValueType>(42);
        EXPECT_EQ(*val, 42);
    }
    {
        struct
        {
            ValueType x = 4;
        } obj;
        auto val_ptr = managed_value(obj);
        EXPECT_EQ(val_ptr.use_count(), 1);
        EXPECT_EQ(val_ptr->x, 4);
    }
}

TEST(StructureOfArrays, Construction)
{
    using tuple_elem_type0 = int;
    using tuple_elem_type1 = double;
    using tuple_type = std::tuple<tuple_elem_type0, tuple_elem_type1>;
    using custom_tuple_type = custom_tuple_example<tuple_elem_type0, tuple_elem_type1>;

    // initizalization
    {
        auto soa = structure_of_arrays<tuple_elem_type0, tuple_elem_type1>();
        EXPECT_EQ(soa.size(), 0);
    }
    {
        auto soa = structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(10);
        EXPECT_EQ(soa.size(), 10);
        auto vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(vec[i]), 0);
            EXPECT_EQ(std::get<1>(vec[i]), 0.0);
        }
    }
    {
        auto soa = structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(10, std::tuple(1, 2.5));
        EXPECT_EQ(soa.size(), 10);
        auto vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(vec[i]), 1);
            EXPECT_EQ(std::get<1>(vec[i]), 2.5);
        }
    }
    {
        auto soa = structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(10, default_init);
        EXPECT_EQ(soa.size(), 10);
    }
    {
        auto soa = structure_of_arrays<tuple_type>();
        EXPECT_EQ(soa.size(), 0);
    }
    {
        auto soa = structure_of_arrays<tuple_type>(10);
        EXPECT_EQ(soa.size(), 10);
        auto vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(vec[i]), 0);
            EXPECT_EQ(std::get<1>(vec[i]), 0.0);
        }
    }
    {
        auto soa = structure_of_arrays<tuple_type>(10, std::tuple(1, 2.5));
        EXPECT_EQ(soa.size(), 10);
        auto vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(vec[i]), 1);
            EXPECT_EQ(std::get<1>(vec[i]), 2.5);
        }
    }
    {
        auto soa = structure_of_arrays<tuple_type>(10, default_init);
        EXPECT_EQ(soa.size(), 10);
    }
    {
        auto soa = structure_of_arrays<custom_tuple_type>();
        EXPECT_EQ(soa.size(), 0);
    }
    {
        auto soa = structure_of_arrays<custom_tuple_type>(10);
        EXPECT_EQ(soa.size(), 10);
        auto vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(vec[i]), 0);
            EXPECT_EQ(std::get<1>(vec[i]), 0.0);
        }
    }
    {
        auto soa = structure_of_arrays<custom_tuple_type>(10, custom_tuple_type(1, 2.5));
        EXPECT_EQ(soa.size(), 10);
        auto vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(vec[i]), 1);
            EXPECT_EQ(std::get<1>(vec[i]), 2.5);
        }
    }
    {
        auto soa = structure_of_arrays<custom_tuple_type>(10, default_init);
        EXPECT_EQ(soa.size(), 10);
    }

    // copy construction and copy assignment (just copys pointer)
    {
        auto arr1 = structure_of_arrays<tuple_type>(10);
        EXPECT_EQ(arr1.use_count(), 1);
        {
            auto arr2 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr2.use_count(), 2);
            EXPECT_EQ(arr2.size(), 10);
            EXPECT_EQ(arr2.data<0>(), arr1.data<0>());
            EXPECT_EQ(arr2.data<1>(), arr1.data<1>());
        }
        EXPECT_EQ(arr1.use_count(), 1);

        {
            auto arr3 = structure_of_arrays<tuple_type>(10);
            EXPECT_EQ(arr3.use_count(), 1);
            const auto old_ptr0 = arr3.data<0>();
            const auto old_ptr1 = arr3.data<1>();
            arr3 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr3.use_count(), 2);
            EXPECT_EQ(arr3.size(), 10);
            EXPECT_EQ(arr3.data<0>(), arr1.data<0>());
            EXPECT_EQ(arr3.data<1>(), arr1.data<1>());
            EXPECT_NE(arr3.data<0>(), old_ptr0);
            EXPECT_NE(arr3.data<1>(), old_ptr1);
        }
        EXPECT_EQ(arr1.use_count(), 1);
    }

    // construction from range of std::tuple
    {
        auto vec = std::vector<tuple_type>();
        for (std::size_t i = 0; i < 10; ++i)
        {
            vec.emplace_back(static_cast<tuple_elem_type0>(i), static_cast<tuple_elem_type1>(i) + 0.5);
        }
        auto soa = structure_of_arrays(vec);
        EXPECT_EQ(soa.size(), 10);
        const auto soa_vec = soa.to<std::vector>();
        EXPECT_EQ(soa_vec, vec);
    }

    // construction from initializer_list of std::tuple
    {
        auto soa = structure_of_arrays<tuple_type>({{0, 0.5}, {1, 1.5}, {2, 2.5}});
        EXPECT_EQ(soa.size(), 3);
        const auto soa_vec = soa.to<std::vector>();
        EXPECT_EQ(std::get<0>(soa_vec[0]), 0);
        EXPECT_EQ(std::get<1>(soa_vec[0]), 0.5);
        EXPECT_EQ(std::get<0>(soa_vec[1]), 1);
        EXPECT_EQ(std::get<1>(soa_vec[1]), 1.5);
        EXPECT_EQ(std::get<0>(soa_vec[2]), 2);
        EXPECT_EQ(std::get<1>(soa_vec[2]), 2.5);
    }

    // construction from multiple ranges
    {
        auto vec0 = std::vector<tuple_elem_type0>();
        auto vec1 = std::vector<tuple_elem_type1>();
        for (std::size_t i = 0; i < 10; ++i)
        {
            vec0.push_back(static_cast<tuple_elem_type0>(i));
            vec1.push_back(static_cast<tuple_elem_type1>(i) + 0.5);
        }
        auto soa = structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(vec0, vec1);
        EXPECT_EQ(soa.size(), 10);
        const auto soa_vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < soa_vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(soa_vec[i]), vec0[i]);
            EXPECT_EQ(std::get<1>(soa_vec[i]), vec1[i]);
        }
    }

    // construction from multiple initializer_lists
    {
        auto soa = structure_of_arrays<tuple_elem_type0, tuple_elem_type1>({0, 1, 2}, {0.5, 1.5, 2.5});
        EXPECT_EQ(soa.size(), 3);
        const auto soa_vec = soa.to<std::vector>();
        EXPECT_EQ(std::get<0>(soa_vec[0]), 0);
        EXPECT_EQ(std::get<1>(soa_vec[0]), 0.5);
        EXPECT_EQ(std::get<0>(soa_vec[1]), 1);
        EXPECT_EQ(std::get<1>(soa_vec[1]), 1.5);
        EXPECT_EQ(std::get<0>(soa_vec[2]), 2);
        EXPECT_EQ(std::get<1>(soa_vec[2]), 2.5);
    }

    // construction from range of custom tuple
    {
        auto vec = std::vector<custom_tuple_example<tuple_elem_type0, tuple_elem_type1>>();
        for (std::size_t i = 0; i < 10; ++i)
        {
            vec.emplace_back(static_cast<tuple_elem_type0>(i), static_cast<tuple_elem_type1>(i) + 0.5);
        }
        auto soa = structure_of_arrays(vec);
        EXPECT_EQ(soa.size(), 10);
        const auto soa_vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < soa_vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(soa_vec[i]), static_cast<tuple_elem_type0>(i));
            EXPECT_EQ(soa_vec[i].get_string<0>(), std::to_string(i));
            EXPECT_EQ(std::get<1>(soa_vec[i]), static_cast<tuple_elem_type1>(i) + 0.5);
            EXPECT_EQ(soa_vec[i].get_string<1>(), std::to_string(i + 0.5));
        }
    }

    // deduction guides
    {
        auto soa = structure_of_arrays(10, custom_tuple_type{5, 7.5});
        static_assert(std::same_as<decltype(soa), structure_of_arrays<custom_tuple_type>>);
        auto soa2 = structure_of_arrays{custom_tuple_type{5, 7.5}, custom_tuple_type{6, 8.5}};
        static_assert(std::same_as<decltype(soa2), structure_of_arrays<custom_tuple_type>>);
        auto soa3 = structure_of_arrays(std::vector<custom_tuple_type>{{1, 2.5}, {3, 4.5}});
        static_assert(std::same_as<decltype(soa3), structure_of_arrays<custom_tuple_type>>);
        auto soa4 = structure_of_arrays(std::vector<tuple_elem_type0>{1, 3}, std::vector<tuple_elem_type1>{2.5, 4.5});
        static_assert(std::same_as<decltype(soa4), structure_of_arrays<tuple_elem_type0, tuple_elem_type1>>);
    }
}

TEST(StructureOfArrays, Export)
{
    using tuple_elem_type0 = int;
    using tuple_elem_type1 = double;
    using tuple_type = std::tuple<tuple_elem_type0, tuple_elem_type1>;

    // export to range with same value type
    {
        const auto vec = std::vector<tuple_type>(10, tuple_type{21, 21.0});
        auto arr = structure_of_arrays<tuple_type>(vec);
        const auto arr_vec = arr.to<std::vector>();
        EXPECT_EQ(arr_vec, vec);
    }
    {
        const auto lst = std::list<tuple_type>(10, tuple_type{21, 21.0});
        auto arr = structure_of_arrays<tuple_type>(lst);
        const auto arr_list = arr.to<std::list>();
        EXPECT_EQ(arr_list, lst);
    }
    {
        auto stdarr = std::array<tuple_type, 10>();
        for (auto& v : stdarr) v = tuple_type{21, 21.0};
        auto arr = structure_of_arrays<tuple_type>(stdarr);
        const auto arr_stdarr = arr.to<std::array<tuple_type, 10>>();
        EXPECT_EQ(arr_stdarr, stdarr);
    }
}

TEST(StructureOfArrays, RangeInterface)
{
    using tuple_type = std::tuple<int, double>;
    using soa_type1 = structure_of_arrays<int, double>;
    using soa_type2 = structure_of_arrays<tuple_type>;
    using soa_type3 = structure_of_arrays<custom_tuple_example<int, double>>;

    // concepts
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    static_assert(std::ranges::range<soa_type1>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<soa_type1>);
#else
    static_assert(!std::ranges::borrowed_range<soa_type1>);
#endif
    static_assert(std::ranges::view<soa_type1>);
    // FIXME: static_assert(std::ranges::output_range<soa_type1, tuple_type>);
    static_assert(std::ranges::input_range<soa_type1>);
    static_assert(std::ranges::forward_range<soa_type1>);
    static_assert(std::ranges::bidirectional_range<soa_type1>);
    static_assert(std::ranges::random_access_range<soa_type1>);
    static_assert(!std::ranges::contiguous_range<soa_type1>);
    static_assert(std::ranges::common_range<soa_type1>);
    static_assert(std::ranges::viewable_range<soa_type1>);

    static_assert(std::ranges::range<soa_type2>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<soa_type2>);
#else
    static_assert(!std::ranges::borrowed_range<soa_type2>);
#endif
    static_assert(std::ranges::view<soa_type2>);
    // FIXME: static_assert(std::ranges::output_range<soa_type2, tuple_type>);
    static_assert(std::ranges::input_range<soa_type2>);
    static_assert(std::ranges::forward_range<soa_type2>);
    static_assert(std::ranges::bidirectional_range<soa_type2>);
    static_assert(std::ranges::random_access_range<soa_type2>);
    static_assert(!std::ranges::contiguous_range<soa_type2>);
    static_assert(std::ranges::common_range<soa_type2>);
    static_assert(std::ranges::viewable_range<soa_type2>);

    static_assert(std::ranges::range<soa_type3>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<soa_type3>);
#else
    static_assert(!std::ranges::borrowed_range<soa_type3>);
#endif
    static_assert(std::ranges::view<soa_type3>);
    // FIXME: static_assert(std::ranges::output_range<soa_type3, tuple_type>);
    static_assert(std::ranges::input_range<soa_type3>);
    static_assert(std::ranges::forward_range<soa_type3>);
    static_assert(std::ranges::bidirectional_range<soa_type3>);
    static_assert(std::ranges::random_access_range<soa_type3>);
    static_assert(!std::ranges::contiguous_range<soa_type3>);
    static_assert(std::ranges::common_range<soa_type3>);
    static_assert(std::ranges::viewable_range<soa_type3>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

    // container-like interface
    {
        auto soa = soa_type2(10);
        EXPECT_NE(soa.data<0>(), nullptr);
        EXPECT_NE(soa.data<1>(), nullptr);
        EXPECT_EQ(soa.size(), 10);
        EXPECT_FALSE(soa.empty());
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(std::same_as<decltype(soa[0]), std::tuple<decltype(*soa.data<0>()), decltype(*soa.data<1>())>>);
        static_assert(
            std::same_as<decltype(*soa.begin()), std::tuple<decltype(*soa.data<0>()), decltype(*soa.data<1>())>>);
        static_assert(std::same_as<decltype(*(soa.end() - 1)),
                                   std::tuple<decltype(*(soa.data<0>() + 9)), decltype(*(soa.data<1>() + 9))>>);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
        soa = {std::tuple(0, 0.0), std::tuple(1, 1.0), std::tuple(2, 2.0), std::tuple(3, 3.0), std::tuple(4, 4.0)};
        auto vec = soa.to<std::vector>();
        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(vec[i]), static_cast<int>(i));
            EXPECT_EQ(std::get<1>(vec[i]), static_cast<double>(i));
        }
    }
}

TEST(StructureOfArrays, SmartPointerInterface)
{
    using soa_type = structure_of_arrays<int, double>;
    auto soa = soa_type(10);

    // reset
    EXPECT_EQ(soa.use_count(), 1);
    EXPECT_EQ(static_cast<bool>(soa), true);
    soa.reset();
    EXPECT_EQ(soa.use_count(), 0);
    EXPECT_EQ(static_cast<bool>(soa), false);
}

TEST(ManagedStructureOfArrays, Construction)
{
    using tuple_elem_type0 = int;
    using tuple_elem_type1 = double;
    using tuple_type = std::tuple<tuple_elem_type0, tuple_elem_type1>;
    using custom_tuple_type = custom_tuple_example<tuple_elem_type0, tuple_elem_type1>;

    // initizalization
    {
        auto soa = managed_structure_of_arrays<tuple_elem_type0, tuple_elem_type1>();
        EXPECT_EQ(soa.size(), 0);
    }
    {
        auto soa = managed_structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(10);
        EXPECT_EQ(soa.size(), 10);
        for (const auto& v : soa)
        {
            EXPECT_EQ(std::get<0>(v), 0);
            EXPECT_EQ(std::get<1>(v), 0.0);
        }
    }
    {
        auto soa = managed_structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(10, std::tuple(1, 2.5));
        EXPECT_EQ(soa.size(), 10);
        for (const auto& v : soa)
        {
            EXPECT_EQ(std::get<0>(v), 1);
            EXPECT_EQ(std::get<1>(v), 2.5);
        }
    }
    {
        auto soa = managed_structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(10, default_init);
        EXPECT_EQ(soa.size(), 10);
    }
    {
        auto soa = managed_structure_of_arrays<tuple_type>();
        EXPECT_EQ(soa.size(), 0);
    }
    {
        auto soa = managed_structure_of_arrays<tuple_type>(10);
        EXPECT_EQ(soa.size(), 10);
        for (const auto& v : soa)
        {
            EXPECT_EQ(std::get<0>(v), 0);
            EXPECT_EQ(std::get<1>(v), 0.0);
        }
    }
    {
        auto soa = managed_structure_of_arrays<tuple_type>(10, std::tuple(1, 2.5));
        EXPECT_EQ(soa.size(), 10);
        for (const auto& v : soa)
        {
            EXPECT_EQ(std::get<0>(v), 1);
            EXPECT_EQ(std::get<1>(v), 2.5);
        }
    }
    {
        auto soa = managed_structure_of_arrays<tuple_type>(10, default_init);
        EXPECT_EQ(soa.size(), 10);
    }
    {
        auto soa = managed_structure_of_arrays<custom_tuple_type>();
        EXPECT_EQ(soa.size(), 0);
    }
    {
        auto soa = managed_structure_of_arrays<custom_tuple_type>(10);
        EXPECT_EQ(soa.size(), 10);
        for (const auto& v : soa)
        {
            EXPECT_EQ(std::get<0>(v), 0);
            EXPECT_EQ(std::get<1>(v), 0.0);
        }
    }
    {
        auto soa = managed_structure_of_arrays<custom_tuple_type>(10, custom_tuple_type(1, 2.5));
        EXPECT_EQ(soa.size(), 10);
        for (const auto& v : soa)
        {
            EXPECT_EQ(std::get<0>(v), 1);
            EXPECT_EQ(std::get<1>(v), 2.5);
        }
    }
    {
        auto soa = managed_structure_of_arrays<custom_tuple_type>(10, default_init);
        EXPECT_EQ(soa.size(), 10);
    }

    // copy construction and copy assignment (just copys pointer)
    {
        auto arr1 = managed_structure_of_arrays<tuple_type>(10);
        EXPECT_EQ(arr1.use_count(), 1);
        {
            auto arr2 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr2.use_count(), 2);
            EXPECT_EQ(arr2.size(), 10);
            EXPECT_EQ(arr2.data<0>(), arr1.data<0>());
            EXPECT_EQ(arr2.data<1>(), arr1.data<1>());
        }
        EXPECT_EQ(arr1.use_count(), 1);

        {
            auto arr3 = managed_structure_of_arrays<tuple_type>(10);
            EXPECT_EQ(arr3.use_count(), 1);
            const auto old_ptr0 = arr3.data<0>();
            const auto old_ptr1 = arr3.data<1>();
            arr3 = arr1;
            EXPECT_EQ(arr1.use_count(), 2);
            EXPECT_EQ(arr3.use_count(), 2);
            EXPECT_EQ(arr3.size(), 10);
            EXPECT_EQ(arr3.data<0>(), arr1.data<0>());
            EXPECT_EQ(arr3.data<1>(), arr1.data<1>());
            EXPECT_NE(arr3.data<0>(), old_ptr0);
            EXPECT_NE(arr3.data<1>(), old_ptr1);
        }
        EXPECT_EQ(arr1.use_count(), 1);
    }

    // construction from range of std::tuple
    {
        auto vec = std::vector<tuple_type>();
        for (std::size_t i = 0; i < 10; ++i)
        {
            vec.emplace_back(static_cast<tuple_elem_type0>(i), static_cast<tuple_elem_type1>(i) + 0.5);
        }
        auto soa = managed_structure_of_arrays(vec);
        EXPECT_EQ(soa.size(), 10);
        for (std::size_t i = 0; i < 10; ++i)
        {
            EXPECT_EQ(std::get<0>(soa[i]), std::get<0>(vec[i]));
            EXPECT_EQ(std::get<1>(soa[i]), std::get<1>(vec[i]));
        }
    }

    // construction from initializer_list of std::tuple
    {
        auto soa = managed_structure_of_arrays<tuple_type>({{0, 0.5}, {1, 1.5}, {2, 2.5}});
        EXPECT_EQ(soa.size(), 3);
        EXPECT_EQ(std::get<0>(soa[0]), 0);
        EXPECT_EQ(std::get<1>(soa[0]), 0.5);
        EXPECT_EQ(std::get<0>(soa[1]), 1);
        EXPECT_EQ(std::get<1>(soa[1]), 1.5);
        EXPECT_EQ(std::get<0>(soa[2]), 2);
        EXPECT_EQ(std::get<1>(soa[2]), 2.5);
    }

    // construction from multiple ranges
    {
        auto vec0 = std::vector<tuple_elem_type0>();
        auto vec1 = std::vector<tuple_elem_type1>();
        for (std::size_t i = 0; i < 10; ++i)
        {
            vec0.push_back(static_cast<tuple_elem_type0>(i));
            vec1.push_back(static_cast<tuple_elem_type1>(i) + 0.5);
        }
        auto soa = managed_structure_of_arrays<tuple_elem_type0, tuple_elem_type1>(vec0, vec1);
        EXPECT_EQ(soa.size(), 10);
        for (std::size_t i = 0; const auto& v : soa)
        {
            EXPECT_EQ(std::get<0>(v), vec0[i]);
            EXPECT_EQ(std::get<1>(v), vec1[i]);
            ++i;
        }
    }

    // construction from multiple initializer_lists
    {
        auto soa = managed_structure_of_arrays<tuple_elem_type0, tuple_elem_type1>({0, 1, 2}, {0.5, 1.5, 2.5});
        EXPECT_EQ(soa.size(), 3);
        const auto soa_vec = soa.to<std::vector>();
        EXPECT_EQ(std::get<0>(soa_vec[0]), 0);
        EXPECT_EQ(std::get<1>(soa_vec[0]), 0.5);
        EXPECT_EQ(std::get<0>(soa_vec[1]), 1);
        EXPECT_EQ(std::get<1>(soa_vec[1]), 1.5);
        EXPECT_EQ(std::get<0>(soa_vec[2]), 2);
        EXPECT_EQ(std::get<1>(soa_vec[2]), 2.5);
    }

    // construction from range of custom tuple
    {
        auto vec = std::vector<custom_tuple_example<tuple_elem_type0, tuple_elem_type1>>();
        for (std::size_t i = 0; i < 10; ++i)
        {
            vec.emplace_back(static_cast<tuple_elem_type0>(i), static_cast<tuple_elem_type1>(i) + 0.5);
        }
        auto soa = managed_structure_of_arrays(vec);
        EXPECT_EQ(soa.size(), 10);
        for (std::size_t i = 0; i < soa.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(soa[i]), static_cast<tuple_elem_type0>(i));
            EXPECT_EQ(soa[i].get_string<0>(), std::to_string(i));
            EXPECT_EQ(std::get<1>(soa[i]), static_cast<tuple_elem_type1>(i) + 0.5);
            EXPECT_EQ(soa[i].get_string<1>(), std::to_string(i + 0.5));
        }
    }

    // deduction guides
    {
        auto soa = managed_structure_of_arrays(10, custom_tuple_type{5, 7.5});
        static_assert(std::same_as<decltype(soa), managed_structure_of_arrays<custom_tuple_type>>);
        auto soa2 = managed_structure_of_arrays{custom_tuple_type{5, 7.5}, custom_tuple_type{6, 8.5}};
        static_assert(std::same_as<decltype(soa2), managed_structure_of_arrays<custom_tuple_type>>);
        auto soa3 = managed_structure_of_arrays(std::vector<custom_tuple_type>{{1, 2.5}, {3, 4.5}});
        static_assert(std::same_as<decltype(soa3), managed_structure_of_arrays<custom_tuple_type>>);
        auto soa4 =
            managed_structure_of_arrays(std::vector<tuple_elem_type0>{1, 3}, std::vector<tuple_elem_type1>{2.5, 4.5});
        static_assert(std::same_as<decltype(soa4), managed_structure_of_arrays<tuple_elem_type0, tuple_elem_type1>>);
    }
}

TEST(ManagedStructureOfArrays, Export)
{
    using tuple_elem_type0 = int;
    using tuple_elem_type1 = double;
    using tuple_type = std::tuple<tuple_elem_type0, tuple_elem_type1>;

    // export to range with same value type
    {
        const auto vec = std::vector<tuple_type>(10, tuple_type{21, 21.0});
        auto arr = managed_structure_of_arrays<tuple_type>(vec);
        const auto arr_vec = arr.to<std::vector>();
        EXPECT_EQ(arr_vec, vec);
    }
    {
        const auto lst = std::list<tuple_type>(10, tuple_type{21, 21.0});
        auto arr = managed_structure_of_arrays<tuple_type>(lst);
        const auto arr_list = arr.to<std::list>();
        EXPECT_EQ(arr_list, lst);
    }
    {
        auto stdarr = std::array<tuple_type, 10>();
        for (auto& v : stdarr) v = tuple_type{21, 21.0};
        auto arr = managed_structure_of_arrays<tuple_type>(stdarr);
        const auto arr_stdarr = arr.to<std::array<tuple_type, 10>>();
        EXPECT_EQ(arr_stdarr, stdarr);
    }
}

TEST(ManagedStructureOfArrays, RangeInterface)
{
    using tuple_type = std::tuple<int, double>;
    using soa_type1 = managed_structure_of_arrays<int, double>;
    using soa_type2 = managed_structure_of_arrays<tuple_type>;
    using soa_type3 = managed_structure_of_arrays<custom_tuple_example<int, double>>;

    // concepts
    static_assert(std::ranges::range<soa_type1>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<soa_type1>);
#else
    static_assert(!std::ranges::borrowed_range<soa_type1>);
#endif
    static_assert(std::ranges::view<soa_type1>);
    // Since C++23: static_assert(std::ranges::output_range<soa_type1, tuple_type>);
    static_assert(std::ranges::input_range<soa_type1>);
    static_assert(std::ranges::forward_range<soa_type1>);
    static_assert(std::ranges::bidirectional_range<soa_type1>);
    static_assert(std::ranges::random_access_range<soa_type1>);
    static_assert(!std::ranges::contiguous_range<soa_type1>);
    static_assert(std::ranges::common_range<soa_type1>);
    static_assert(std::ranges::viewable_range<soa_type1>);

    static_assert(std::ranges::range<soa_type2>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<soa_type2>);
#else
    static_assert(!std::ranges::borrowed_range<soa_type2>);
#endif
    static_assert(std::ranges::view<soa_type2>);
    // Since C++23: static_assert(std::ranges::output_range<soa_type2, tuple_type>);
    static_assert(std::ranges::input_range<soa_type2>);
    static_assert(std::ranges::forward_range<soa_type2>);
    static_assert(std::ranges::bidirectional_range<soa_type2>);
    static_assert(std::ranges::random_access_range<soa_type2>);
    static_assert(!std::ranges::contiguous_range<soa_type2>);
    static_assert(std::ranges::common_range<soa_type2>);
    static_assert(std::ranges::viewable_range<soa_type2>);

    static_assert(std::ranges::range<soa_type3>);
#if defined(GPU_DEVICE_COMPILE)
    static_assert(std::ranges::borrowed_range<soa_type3>);
#else
    static_assert(!std::ranges::borrowed_range<soa_type3>);
#endif
    static_assert(std::ranges::view<soa_type3>);
    // FIXME: static_assert(std::ranges::output_range<soa_type3, tuple_type>);
    static_assert(std::ranges::input_range<soa_type3>);
    static_assert(std::ranges::forward_range<soa_type3>);
    static_assert(std::ranges::bidirectional_range<soa_type3>);
    static_assert(std::ranges::random_access_range<soa_type3>);
    static_assert(!std::ranges::contiguous_range<soa_type3>);
    static_assert(std::ranges::common_range<soa_type3>);
    static_assert(std::ranges::viewable_range<soa_type3>);

    // container-like interface
    {
        auto soa = soa_type2(10);
        EXPECT_NE(soa.data<0>(), nullptr);
        EXPECT_NE(soa.data<1>(), nullptr);
        EXPECT_EQ(soa.size(), 10);
        EXPECT_FALSE(soa.empty());
        EXPECT_EQ(soa[0], std::tuple(*soa.data<0>(), *soa.data<1>()));
        EXPECT_EQ(*soa.begin(), std::tuple(*soa.data<0>(), *soa.data<1>()));
        EXPECT_EQ(*(soa.end() - 1), std::tuple(*(soa.data<0>() + 9), *(soa.data<1>() + 9)));

        soa = {std::tuple(0, 0.0), std::tuple(1, 1.0), std::tuple(2, 2.0), std::tuple(3, 3.0), std::tuple(4, 4.0)};
        for (std::size_t i = 0; i < soa.size(); ++i)
        {
            EXPECT_EQ(std::get<0>(soa[i]), static_cast<int>(i));
            EXPECT_EQ(std::get<1>(soa[i]), static_cast<double>(i));
        }
    }
}

TEST(ManagedStructureOfArrays, SmartPointerInterface)
{
    using soa_type = managed_structure_of_arrays<int, double>;
    auto soa = soa_type(10);

    // reset
    EXPECT_EQ(soa.use_count(), 1);
    EXPECT_EQ(static_cast<bool>(soa), true);
    soa.reset();
    EXPECT_EQ(soa.use_count(), 0);
    EXPECT_EQ(static_cast<bool>(soa), false);
}

TEST(ManagedStructureOfArrays, MemoryManagement)
{
    // create nested managed_array
    auto elms = std::vector<std::tuple<int, managed_array<double>>>();
    for (std::size_t i = 0; i < 10; ++i) elms.emplace_back(i, managed_array<double>(i, 99));
    auto soa = managed_structure_of_arrays(elms);

    // check if prefetching works without errors
    {
        // prefetch 4 elements from the 3rd to device 0 with stream 0 recursively.
        soa.prefetch(3, 4, 0, 0, true);
        // prefetch 4 elements from the 3rd to current device with stream 0 recurslively.
        soa.prefetch(3, 4, 0, true);
        // all elements to device 0 with stream 0 recursively.
        soa.prefetch(0, 0, true);
        // all elements to current device with stream 0 recursively.
        soa.prefetch(0, true);
        // all elements to current device with stream 0 recursively.
        soa.prefetch();

        // prefetch 4 elements from the 3rd to host with stream 0 recursively.
        soa.prefetch_to_cpu(3, 4, 0, true);
        // all elements to host with stream 0 recursively.
        soa.prefetch_to_cpu(0, true);
        // all elements to host with stream 0 recursively.
        soa.prefetch_to_cpu();
    }

    // check if mem advice works without errors
    {
        // set memory advice for 4 elements from the 3rd to preferred location for device 0 recursively.
        soa.mem_advise(3, 4, api::gpuMemoryAdvise::SetPreferredLocation, 0, true);
        // set memory advice for 4 elements from the 3rd to preferred location for current device recursively.
        soa.mem_advise(3, 4, api::gpuMemoryAdvise::SetPreferredLocation, true);
        // set memory advice for all elements to preferred location for device 0 recursively.
        soa.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation, 0, true);
        // set memory advice for all elements to preferred location for current device recursively.
        soa.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation, true);
        // set memory advice for all elements to preferred location for current device recursively.
        soa.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation);

        // set memory advice for 4 elements from the 3rd to read mostly for host recursively.
        soa.mem_advise_to_cpu(3, 4, api::gpuMemoryAdvise::SetReadMostly, true);
        // set memory advice for all elements to read mostly for host recursively.
        soa.mem_advise_to_cpu(api::gpuMemoryAdvise::SetReadMostly, true);
        // set memory advice for all elements to read mostly for host recursively.
        soa.mem_advise_to_cpu(api::gpuMemoryAdvise::SetReadMostly);
    }
}

TEST(JaggedArray, Construction)
{
    // construction from sizes
    {
        auto jagged = jagged_array<managed_array<int>>(std::vector<std::size_t>{1, 3, 2});
        EXPECT_EQ(jagged.size(), 6);
        EXPECT_EQ(jagged.num_rows(), 3);
        EXPECT_EQ(jagged.size(0), 1);
        EXPECT_EQ(jagged.size(1), 3);
        EXPECT_EQ(jagged.size(2), 2);
        for (const auto& v : jagged) EXPECT_EQ(v, 0);
    }

    // construction from sizes of initializer_list
    {
        auto jagged = jagged_array<managed_array<int>>({1, 3, 2});
        EXPECT_EQ(jagged.size(), 6);
        EXPECT_EQ(jagged.num_rows(), 3);
        EXPECT_EQ(jagged.size(0), 1);
        EXPECT_EQ(jagged.size(1), 3);
        EXPECT_EQ(jagged.size(2), 2);
        for (const auto& v : jagged) EXPECT_EQ(v, 0);
    }

    // construction from nested containers for jagged managed array
    {
        auto vec_vec =
            std::vector<std::vector<int>>{std::vector<int>(1, 0), std::vector<int>(2, 1), std::vector<int>(3, 2),
                                          std::vector<int>(4, 3), std::vector<int>(5, 4)};
        auto vec_arr = std::vector<managed_array<int>>{managed_array<int>(1, 0), managed_array<int>(2, 1),
                                                       managed_array<int>(3, 2), managed_array<int>(4, 3),
                                                       managed_array<int>(5, 4)};
        auto arr_arr = managed_array<managed_array<int, std::uint8_t>>{
            managed_array<int, std::uint8_t>(1, 0), managed_array<int, std::uint8_t>(2, 1),
            managed_array<int, std::uint8_t>(3, 2), managed_array<int, std::uint8_t>(4, 3),
            managed_array<int, std::uint8_t>(5, 4)};

        auto jagged_vec_vec = jagged_array<managed_array<int>>(vec_vec);  // no deduction guide
        auto jagged_vec_arr = jagged_array(vec_arr);
        auto jagged_arr_arr = jagged_array(arr_arr);

        // deduction guides
        static_assert(std::same_as<decltype(jagged_vec_vec), jagged_array<managed_array<int>>>);
        static_assert(std::same_as<decltype(jagged_vec_arr), jagged_array<managed_array<int>>>);
        static_assert(std::same_as<decltype(jagged_arr_arr), jagged_array<managed_array<int, std::uint8_t>>>);

        EXPECT_EQ(jagged_vec_vec.size(), 15);
        EXPECT_EQ(jagged_vec_arr.size(), 15);
        EXPECT_EQ(jagged_arr_arr.size(), 15);
        for (std::size_t i = 0; i < jagged_vec_vec.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_vec_vec.size(i), i + 1);
            for (const auto& v : jagged_vec_vec.row(i)) EXPECT_EQ(v, i);
        }
        for (std::size_t i = 0; i < jagged_vec_arr.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_vec_arr.size(i), i + 1);
            for (const auto& v : jagged_vec_arr.row(i)) EXPECT_EQ(v, i);
        }
        for (std::size_t i = 0; i < jagged_arr_arr.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr_arr.size(i), i + 1);
            for (const auto& v : jagged_arr_arr.row(i)) EXPECT_EQ(v, i);
        }
    }

    // construction from nested containers for jagged structure of arrays
    {
        using tuple_type = std::tuple<int, double>;
        auto vec_tpl = std::vector<std::vector<tuple_type>>{
            std::vector<tuple_type>(1, {0, 0.5}), std::vector<tuple_type>(2, {1, 1.5}),
            std::vector<tuple_type>(3, {2, 2.5}), std::vector<tuple_type>(4, {3, 3.5}),
            std::vector<tuple_type>(5, {4, 4.5})};
        auto vec_soa = std::vector<managed_structure_of_arrays<tuple_type>>{
            managed_structure_of_arrays<tuple_type>(1, {0, 0.5}), managed_structure_of_arrays<tuple_type>(2, {1, 1.5}),
            managed_structure_of_arrays<tuple_type>(3, {2, 2.5}), managed_structure_of_arrays<tuple_type>(4, {3, 3.5}),
            managed_structure_of_arrays<tuple_type>(5, {4, 4.5})};
        auto arr_sos = managed_array<managed_structure_of_arrays<tuple_type, std::uint8_t>, std::uint32_t>{
            managed_structure_of_arrays<tuple_type, std::uint8_t>(1, {0, 0.5}),
            managed_structure_of_arrays<tuple_type, std::uint8_t>(2, {1, 1.5}),
            managed_structure_of_arrays<tuple_type, std::uint8_t>(3, {2, 2.5}),
            managed_structure_of_arrays<tuple_type, std::uint8_t>(4, {3, 3.5}),
            managed_structure_of_arrays<tuple_type, std::uint8_t>(5, {4, 4.5})};

        auto jagged_vec_tpl1 = jagged_array<managed_array<tuple_type>>(vec_tpl);                // no deduction guide
        auto jagged_vec_tpl2 = jagged_array<managed_structure_of_arrays<tuple_type>>(vec_tpl);  // no deduction guide
        auto jagged_vec_soa = jagged_array(vec_soa);
        auto jagged_arr_soa = jagged_array(arr_sos);

        // deduction guides
        static_assert(std::same_as<decltype(jagged_vec_soa), jagged_array<managed_structure_of_arrays<tuple_type>>>);
        static_assert(std::same_as<decltype(jagged_arr_soa),
                                   jagged_array<managed_structure_of_arrays<tuple_type, std::uint8_t>>>);

        EXPECT_EQ(jagged_vec_tpl1.size(), 15);
        EXPECT_EQ(jagged_vec_tpl2.size(), 15);
        EXPECT_EQ(jagged_vec_soa.size(), 15);
        EXPECT_EQ(jagged_arr_soa.size(), 15);

        for (std::size_t i = 0; i < jagged_vec_tpl2.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_vec_tpl2.size(i), i + 1);
            for (const auto& v : jagged_vec_tpl2.row(i)) EXPECT_EQ(v, tuple_type(i, static_cast<double>(i) + 0.5));
        }
        for (std::size_t i = 0; i < jagged_vec_soa.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_vec_soa.size(i), i + 1);
            for (const auto& v : jagged_vec_soa.row(i)) EXPECT_EQ(v, tuple_type(i, static_cast<double>(i) + 0.5));
        }
        for (std::size_t i = 0; i < jagged_arr_soa.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr_soa.size(i), i + 1);
            for (const auto& v : jagged_arr_soa.row(i)) EXPECT_EQ(v, tuple_type(i, static_cast<double>(i) + 0.5));
        }
    }

    // construction from container of sizes and flat range (managed array)
    {
        auto flat_range = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
        auto sizes = std::vector<std::uint32_t>{1, 2, 3, 4, 5};

        auto jagged_arr = jagged_array<managed_array<int>>(sizes, flat_range);  // no deduction guide

        EXPECT_EQ(jagged_arr.size(), 15);
        for (std::size_t i = 0, j = 0; i < jagged_arr.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr.size(i), sizes[i]);
            for (const auto& v : jagged_arr.row(i)) EXPECT_EQ(v, j++);
        }
    }

    // construction from container of sizes and flat range (structure of arrays)
    {
        auto flat_range = std::vector<std::tuple<int, double>>{
            {0, 0.5}, {1, 1.5}, {2, 2.5},   {3, 3.5},   {4, 4.5},   {5, 5.5},   {6, 6.5},  {7, 7.5},
            {8, 8.5}, {9, 9.5}, {10, 10.5}, {11, 11.5}, {12, 12.5}, {13, 13.5}, {14, 14.5}};
        auto sizes = std::vector<std::uint32_t>{1, 2, 3, 4, 5};

        auto jagged_arr = jagged_array<managed_structure_of_arrays<std::tuple<int, double>>>(
            sizes, flat_range);  // no deduction guide

        EXPECT_EQ(jagged_arr.size(), 15);
        for (std::size_t i = 0, j = 0; i < jagged_arr.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr.size(i), sizes[i]);
            for (const auto& v : jagged_arr.row(i))
            {
                EXPECT_EQ(v, (std::tuple<int, double>(j, static_cast<double>(j) + 0.5)));
                ++j;
            }
        }
    }

    // construction from nested initializer_list (managed array)
    {
        auto jagged_arr = jagged_array<managed_array<int>>(
            {{0}, {1, 1}, {2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4, 4}});  // no deduction guide

        EXPECT_EQ(jagged_arr.size(), 15);
        for (std::size_t i = 0; i < jagged_arr.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr.size(i), i + 1);
            for (const auto& v : jagged_arr.row(i)) EXPECT_EQ(v, static_cast<int>(i));
        }
    }

    // construction from nested initializer_list (managed structure of arrays)
    {
        using tuple_type = std::tuple<int, double>;
        auto jagged_arr = jagged_array<managed_structure_of_arrays<tuple_type>>{
            {{0, 0.5}},
            {{1, 1.5}, {1, 1.5}},
            {{2, 2.5}, {2, 2.5}, {2, 2.5}},
            {{3, 3.5}, {3, 3.5}, {3, 3.5}, {3, 3.5}},
            {{4, 4.5}, {4, 4.5}, {4, 4.5}, {4, 4.5}, {4, 4.5}}};  // no deduction guide
        EXPECT_EQ(jagged_arr.size(), 15);
        for (std::size_t i = 0; i < jagged_arr.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr.size(i), i + 1);
            for (const auto& v : jagged_arr.row(i))
            {
                EXPECT_EQ(v, tuple_type(static_cast<int>(i), static_cast<double>(i) + 0.5));
            }
        }
    }

    // wrap managed array with sizes
    {
        auto sizes = std::vector<std::uint32_t>{3, 1, 4, 1, 5};
        auto arr = managed_array<double>(14, 99.0);
        auto jagged_arr_wrap = jagged_array(sizes, arr);

        // use count: jagged_arr_wrap wraps managed_array arr without copy
        EXPECT_EQ(arr.use_count(), 2);

        // deduction guide
        static_assert(std::same_as<decltype(jagged_arr_wrap), jagged_array<managed_array<double>>>);

        EXPECT_EQ(jagged_arr_wrap.size(), arr.size());
        EXPECT_EQ(std::accumulate(sizes.begin(), sizes.end(), 0), arr.size());
        for (std::size_t i = 0; i < jagged_arr_wrap.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr_wrap.size(i), sizes[i]);
            for (const auto& v : jagged_arr_wrap.row(i)) EXPECT_EQ(v, 99.0);
        }
    }

    // wrap managed structure of arrays with sizes
    {
        using tuple_type = std::tuple<int, double>;
        auto sizes = std::vector<std::uint32_t>{3, 1, 4, 1, 5};
        auto arr = managed_structure_of_arrays<tuple_type>(14, tuple_type(42, 99.0));
        auto jagged_arr_wrap = jagged_array(sizes, arr);

        // use count: jagged_arr_wrap wraps managed_structure_of_arrays arr without copy
        EXPECT_EQ(arr.use_count(), 2);

        // deduction guide
        static_assert(std::same_as<decltype(jagged_arr_wrap), jagged_array<managed_structure_of_arrays<tuple_type>>>);

        EXPECT_EQ(jagged_arr_wrap.size(), arr.size());
        EXPECT_EQ(std::accumulate(sizes.begin(), sizes.end(), 0), arr.size());
        for (std::size_t i = 0; i < jagged_arr_wrap.num_rows(); ++i)
        {
            EXPECT_EQ(jagged_arr_wrap.size(i), sizes[i]);
            for (const auto& v : jagged_arr_wrap.row(i))
            {
                EXPECT_EQ(std::get<0>(v), 42);
                EXPECT_EQ(std::get<1>(v), 99.0);
            }
        }
    }

    // wrap managed array with sizes initializer_list
    {
        auto arr = managed_array<double>(14, 99.0);
        auto jagged_arr_wrap = jagged_array({3, 1, 4, 1, 5}, arr);

        // use count: jagged_arr_wrap wraps managed_array arr without copy
        EXPECT_EQ(arr.use_count(), 2);

        // deduction guide
        static_assert(std::same_as<decltype(jagged_arr_wrap), jagged_array<managed_array<double>>>);

        EXPECT_EQ(jagged_arr_wrap.size(), arr.size());
        for (std::size_t i = 0; i < jagged_arr_wrap.num_rows(); ++i)
        {
            for (const auto& v : jagged_arr_wrap.row(i)) EXPECT_EQ(v, 99.0);
        }
    }
}

TEST(JaggedArray, Export)
{
    // Skip: inherited
}

TEST(Jagged, RangeInterface)
{
    // Skip: inherited
}

TEST(JaggedArray, MemoryManagement)
{
    // create nested managed_array
    auto vec_vec = std::vector<std::vector<int>>{std::vector<int>(1, 0), std::vector<int>(2, 1), std::vector<int>(3, 2),
                                                 std::vector<int>(4, 3), std::vector<int>(5, 4)};
    auto jagged_arr = jagged_array<managed_array<int>>(vec_vec);

    // check if prefetching works without errors
    {
        // prefetch 4 elements from the 3rd to device 0 with stream 0 recursively.
        jagged_arr.prefetch(3, 4, 0, 0, true);
        // prefetch 4 elements from the 3rd to current device with stream 0 recurslively.
        jagged_arr.prefetch(3, 4, 0, true);
        // all elements to device 0 with stream 0 recursively.
        jagged_arr.prefetch(0, 0, true);
        // all elements to current device with stream 0 recursively.
        jagged_arr.prefetch(0, true);
        // all elements to current device with stream 0 recursively.
        jagged_arr.prefetch();

        // prefetch 4 elements from the 3rd to host with stream 0 recursively.
        jagged_arr.prefetch_to_cpu(3, 4, 0, true);
        // all elements to host with stream 0 recursively.
        jagged_arr.prefetch_to_cpu(0, true);
        // all elements to host with stream 0 recursively.
        jagged_arr.prefetch_to_cpu();
    }

    // check if mem advice works without errors
    {
        // set memory advice for 4 elements from the 3rd to preferred location for device 0 recursively.
        jagged_arr.mem_advise(3, 4, api::gpuMemoryAdvise::SetPreferredLocation, 0, true);
        // set memory advice for 4 elements from the 3rd to preferred location for current device recursively.
        jagged_arr.mem_advise(3, 4, api::gpuMemoryAdvise::SetPreferredLocation, true);
        // set memory advice for all elements to preferred location for device 0 recursively.
        jagged_arr.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation, 0, true);
        // set memory advice for all elements to preferred location for current device recursively.
        jagged_arr.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation, true);
        // set memory advice for all elements to preferred location for current device recursively.
        jagged_arr.mem_advise(api::gpuMemoryAdvise::SetPreferredLocation);

        // set memory advice for 4 elements from the 3rd to read mostly for host recursively.
        jagged_arr.mem_advise_to_cpu(3, 4, api::gpuMemoryAdvise::SetReadMostly, true);
        // set memory advice for all elements to read mostly for host recursively.
        jagged_arr.mem_advise_to_cpu(api::gpuMemoryAdvise::SetReadMostly, true);
        // set memory advice for all elements to read mostly for host recursively.
        jagged_arr.mem_advise_to_cpu(api::gpuMemoryAdvise::SetReadMostly);
    }
}
// NOLINTEND
