/*===================================================*
|  GPU smart pointer (gpu-smart-ptr) version v0.2.0  |
|  https://github.com/yosh-matsuda/gpu-ptr           |
|                                                    |
|  Copyright (c) 2026 Yoshiki Matsuda @yosh-matsuda  |
|                                                    |
|  This software is released under the MIT License.  |
|  https://opensource.org/license/mit/               |
====================================================*/

#pragma once
#include "gpu_runtime_api.hpp"

#if defined(GPU_DEVICE_COMPILE)
#if !defined(NDEBUG)
#define GPU_NDEBUG_DEVICE_CODE_STL
#define NDEBUG
#endif
#endif

#include <algorithm>
#include <concepts>
#include <memory>
#include <numeric>
#include <ranges>
#include <tuple>

#if !defined(GPU_DEVICE_COMPILE)
#include <iostream>
#include <stdexcept>
#include <vector>
#endif

#if defined(GPU_PTR_DEBUG)
#include <nameof.hpp>
#endif

#if defined(GPU_NDEBUG_DEVICE_CODE_STL)
#undef GPU_NDEBUG_DEVICE_CODE_STL
#undef NDEBUG
#endif

#include <cassert>

#if defined(GPU_DEVICE_COMPILE) || !defined(__NVCC__)
#define SIGSEGV_DEPRECATED
#else
#define SIGSEGV_DEPRECATED [[deprecated("Cannot access GPU memory directly")]]
#endif

namespace gpu_ptr
{
#if defined(GPU_USE_32BIT_SIZE_TYPE_DEFAULT)
    // Use 32-bit size_type for reducing register usage on GPU
    using size_type_default = std::uint32_t;
#else
    using size_type_default = std::size_t;
#endif

    template <typename ValueType, typename SizeType = size_type_default>
    requires std::is_trivially_copyable_v<ValueType>
    class array;
    template <typename ValueType, typename SizeType = size_type_default>
    class managed_array;

    inline constexpr struct default_init_tag
    {
        struct tag
        {
        };
        constexpr explicit default_init_tag(tag) {}
    } default_init{default_init_tag::tag{}};

    namespace detail
    {

#if defined(GPU_PTR_DEBUG)
        inline std::size_t gpu_memory_usage = 0UL;
#define INCR_GPU_MEORY_USAGE(x) (gpu_ptr::api::gpu_memory_usage += (x))
#define DECR_GPU_MEORY_USAGE(x) (gpu_ptr::api::gpu_memory_usage -= (x))
#define MEMORY_USAGE_EQ(x) (gpu_ptr::api::gpu_memory_usage == (x))
#else
#define INCR_GPU_MEORY_USAGE(x) void(x)
#define DECR_GPU_MEORY_USAGE(x) void(x)
#define MEMORY_USAGE_EQ(x) (true)
#endif

        template <typename>
        constexpr bool always_false_v = false;

        __host__ inline int get_device_count()
        {
            int device_count;
            GPU_CHECK_ERROR(api::gpuGetDeviceCount(&device_count));
            return device_count;
        }

        __host__ inline int get_device_id()
        {
            int device_id;
            GPU_CHECK_ERROR(api::gpuGetDevice(&device_id));
            assert(device_id >= 0 && device_id < get_device_count());
            return device_id;
        }

        template <bool Unified, typename SizeType, typename... ValueTypes>
        requires (sizeof...(ValueTypes) > 0)
        class base
        {
        public:
            using size_type = SizeType;
            using difference_type = std::ptrdiff_t;

            __host__ __device__ size_type size() const noexcept { return size_; }
            __host__ __device__ bool empty() const noexcept { return size_ == 0; }
            __host__ __device__ std::uint32_t use_count() const noexcept
            {
                return base::ref_count_ != nullptr ? *base::ref_count_ : 0U;
            }

        protected:
            size_type size_ = 0U;
            std::tuple<ValueTypes*...> data_;
            std::uint32_t* ref_count_ = nullptr;  // reference counter, not used on GPU

            template <std::size_t N>
            using element_type = std::tuple_element_t<N, std::tuple<ValueTypes...>>;

            __host__ __device__ void init()
            {
                size_ = 0U;
                base::tuple_for_each([](auto*& ptr) { ptr = nullptr; });
                ref_count_ = nullptr;
            }
            __host__ void free()
            {
#ifndef GPU_DEVICE_COMPILE
                assert((size_ == 0) == (ref_count_ == nullptr));
                if (ref_count_ == nullptr) return;

                // delete objects
                if (--*ref_count_ == 0)
                {
#if defined(GPU_PTR_DEBUG)
                    std::cout << " gpuFree: " << ((std::string(NAMEOF_FULL_TYPE(ValueTypes)) + ' ') + ...)
                              << ", size: " << size_ << ", refcount: " << *ref_count_ << '\n';
#endif

                    // do not throw in destructor
                    try
                    {
                        if constexpr (Unified)
                        {
                            // call destructor explicitly for unified memory
                            base::tuple_for_each([this](auto* ptr) { std::destroy_n(ptr, size_); });
                        }

                        base::tuple_for_each([](auto* ptr) { GPU_CHECK_ERROR(api::gpuFree(ptr)); });
                        (DECR_GPU_MEORY_USAGE(sizeof(ValueTypes) * size_), ...);  // for debug

#ifndef _CLANGD                                                                   // clangd crashes with this code
                        delete ref_count_;
#endif
                    }
                    catch (std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                        std::terminate();
                    }
                    catch (...)
                    {
                        std::cerr << "gpuFree failed\n";
                        std::terminate();
                    }
                }

                // init variables
                init();
#endif
            }

            __host__ __device__ base& operator=(const base& r)
            {
#ifndef GPU_DEVICE_COMPILE
                free();
#endif

                size_ = r.size_;
                data_ = r.data_;
                ref_count_ = r.ref_count_;
#ifndef GPU_DEVICE_COMPILE
                if (ref_count_ != nullptr)
                {
                    ++*ref_count_;
#if defined(GPU_PTR_DEBUG)
                    std::cout << " copied: " << ((std::string(NAMEOF_FULL_TYPE(ValueTypes)) + ' ') + ...)
                              << ", size: " << size_ << ", refcount: " << *ref_count_ << '\n';
#endif
                }
#endif
                return *this;
            }
            __host__ __device__ base& operator=(base&& r) noexcept
            {
#ifndef GPU_DEVICE_COMPILE
                free();
#endif

                size_ = r.size_;
                data_ = r.data_;
                ref_count_ = r.ref_count_;

                r.init();
                return *this;
            }

            __host__ __device__ base() : data_(static_cast<ValueTypes*>(nullptr)...) {};
            __host__ __device__ base(const base& r) : size_(r.size_), data_(r.data_), ref_count_(r.ref_count_)
            {
#ifndef GPU_DEVICE_COMPILE
                if (ref_count_ != nullptr)
                {
                    ++*ref_count_;
#if defined(GPU_PTR_DEBUG)
                    std::cout << " copied: " << ((std::string(NAMEOF_FULL_TYPE(ValueTypes)) + ' ') + ...)
                              << ", size: " << size_ << ", refcount: " << *ref_count_ << '\n';
#endif
                }
#endif
            }
            __host__ __device__ base(base&& r) noexcept : size_(r.size_), data_(r.data_), ref_count_(r.ref_count_)
            {
                r.init();
            }
            __device__ explicit base(ValueTypes*... ptr, size_type size)
                : size_(size), data_(static_cast<ValueTypes*>(size != 0 ? ptr : nullptr)...)
            {
                assert(size_ == 0 || ((ptr != nullptr) && ...));
            }
            __host__ explicit base(std::size_t size)
                : size_(static_cast<size_type>(size)), ref_count_(size_ == 0 ? nullptr : new std::uint32_t(1))
            {
                // check range size overflow
                if (std::numeric_limits<size_type>::max() < size)
                {
                    throw std::runtime_error("size overflow");
                }

                // called from derived class constructor
                (INCR_GPU_MEORY_USAGE(sizeof(ValueTypes) * size_), ...);  // for debug
            }
#ifndef GPU_DEVICE_COMPILE
            __host__ ~base() { free(); }
#endif

            __host__ __device__ void tuple_for_each(auto&& f) const
            {
                std::apply([&f](const auto&... args) { (f(args), ...); }, data_);
            }
            __host__ __device__ void tuple_for_each(auto&& f)
            {
                std::apply([&f](auto&... args) { (f(args), ...); }, data_);
            }
        };

        template <bool Unified, typename SizeType, typename... ValueType>
        constexpr void is_ptr_helper(const detail::base<Unified, SizeType, ValueType...>&)
        {
        }
        template <typename SizeType, typename... ValueType>
        constexpr void is_managed_ptr_helper(const detail::base<true, SizeType, ValueType...>&)
        {
        }
        template <typename SizeType, typename... ValueType>
        constexpr void is_unmanaged_ptr_helper(const detail::base<false, SizeType, ValueType...>&)
        {
        }
        template <bool Unified, typename SizeType, typename ValueType>
        constexpr void is_array_helper(const detail::base<Unified, SizeType, ValueType>&)
        {
        }
        template <typename ValueType, typename SizeType>
        constexpr void is_managed_array_helper(const detail::base<true, SizeType, ValueType>&)
        {
        }
        template <typename ValueType, typename SizeType>
        constexpr void is_unmanaged_array_helper(const detail::base<false, SizeType, ValueType>&)
        {
        }

        template <class Derived>
        concept gpu_ptr = requires(Derived d) { is_ptr_helper(d); };
        template <class Derived>
        concept gpu_managed_ptr = requires(Derived d) { is_managed_ptr_helper(d); };
        template <class Derived>
        concept gpu_unmanaged_ptr = requires(Derived d) { is_unmanaged_ptr_helper(d); };
        template <class Derived>
        concept gpu_array_ptr = requires(Derived d) { is_array_helper(d); };
        template <class Derived>
        concept gpu_managed_array_ptr = requires(Derived d) { is_managed_array_helper(d); };
        template <class Derived>
        concept gpu_unmanaged_array_ptr = requires(Derived d) { is_unmanaged_array_helper(d); };

        template <typename T>
        concept array_convertible = (!gpu_ptr<T>) && std::ranges::forward_range<T> && std::ranges::sized_range<T>;
        template <typename T>
        concept array_convertible_for_copy =
            (!gpu_unmanaged_ptr<T>) && std::ranges::forward_range<T> && std::ranges::sized_range<T>;

        template <typename T>
        struct unified_array_deduced
        {
            using type = T;
        };
        template <typename T>
        using unified_array_deduced_t = typename unified_array_deduced<T>::type;

        template <array_convertible T>
        struct unified_array_deduced<T>
        {
            using type = managed_array<unified_array_deduced_t<std::ranges::range_value_t<T>>>;
        };

        template <gpu_array_ptr T, template <typename...> typename U>
        struct to_range_deduced
        {
            using type = U<std::ranges::range_value_t<T>>;
        };
        template <typename T, template <typename...> typename U>
        using to_range_deduced_t = typename to_range_deduced<T, U>::type;

        template <gpu_array_ptr Array, template <typename...> typename U>
        requires gpu_array_ptr<std::ranges::range_value_t<Array>>
        struct to_range_deduced<Array, U>
        {
            using type = U<to_range_deduced_t<std::ranges::range_value_t<Array>, U>>;
        };

        inline constexpr struct join_init_tag
        {
            struct tag
            {
            };
            constexpr explicit join_init_tag(tag) {}
        } join_init{join_init_tag::tag{}};
    }  // namespace detail

    using detail::gpu_array_ptr;
    using detail::gpu_managed_array_ptr;
    using detail::gpu_managed_ptr;
    using detail::gpu_ptr;
    using detail::gpu_unmanaged_array_ptr;
    using detail::gpu_unmanaged_ptr;

    template <typename ValueType, typename SizeType>
    requires std::is_trivially_copyable_v<ValueType>
    class array : public detail::base<false, SizeType, ValueType>
    {
        using base = detail::base<false, SizeType, ValueType>;

    public:
        using size_type = SizeType;
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator = value_type*;
        using const_iterator = const value_type*;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        SIGSEGV_DEPRECATED __host__ __device__ const_reference operator[](size_type i) const noexcept
        {
            return data()[i];
        };
        SIGSEGV_DEPRECATED __host__ __device__ reference operator[](size_type i) noexcept { return data()[i]; }
        SIGSEGV_DEPRECATED __host__ __device__ iterator begin() noexcept { return data(); }
        SIGSEGV_DEPRECATED __host__ __device__ iterator end() noexcept { return data() + base::size_; }
        SIGSEGV_DEPRECATED __host__ __device__ std::reverse_iterator<iterator> rbegin() noexcept
        {
            return std::reverse_iterator<iterator>(end());
        }
        SIGSEGV_DEPRECATED __host__ __device__ std::reverse_iterator<iterator> rend() noexcept
        {
            return std::reverse_iterator<iterator>(begin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator begin() const noexcept { return data(); }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator end() const noexcept { return data() + base::size_; }
        SIGSEGV_DEPRECATED __host__ __device__ std::reverse_iterator<const_iterator> rbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(end());
        }
        SIGSEGV_DEPRECATED __host__ __device__ std::reverse_iterator<const_iterator> rend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(begin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator cbegin() const noexcept { return data(); }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator cend() const noexcept { return data() + base::size_; }
        SIGSEGV_DEPRECATED __host__ __device__ std::reverse_iterator<const_iterator> crbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cend());
        }
        SIGSEGV_DEPRECATED __host__ __device__ std::reverse_iterator<const_iterator> crend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cbegin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ reference front() noexcept { return *begin(); }
        SIGSEGV_DEPRECATED __host__ __device__ const_reference front() const noexcept { return *begin(); }
        SIGSEGV_DEPRECATED __host__ __device__ reference back() noexcept { return *(data() + base::size_ - 1); }
        SIGSEGV_DEPRECATED __host__ __device__ const_reference back() const noexcept
        {
            return *(data() + base::size_ - 1);
        }
        __host__ __device__ pointer data() noexcept { return std::get<0>(base::data_); }
        __host__ __device__ pointer data() const noexcept { return std::get<0>(base::data_); }

        array() = default;
        __host__ __device__ array(const array& r) : base(r) {}
        __host__ __device__ array(array&& r) noexcept : base(std::move(r)) {}

        __host__ explicit array(std::size_t size) : base(size)
        {
            if (base::size_ == 0) return;
            auto buf = std::make_unique<value_type[]>(base::size_);
            GPU_CHECK_ERROR(
                api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            GPU_CHECK_ERROR(api::gpuMemcpy(data(), buf.get(), sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
        }

        __host__ array(std::size_t size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(
                api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            if constexpr (!std::is_trivially_default_constructible_v<value_type>)
            {
                auto buf = std::make_unique<value_type[]>(base::size_);
                GPU_CHECK_ERROR(
                    api::gpuMemcpy(data(), buf.get(), sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
            }
        }

        __host__ array(std::size_t size, const value_type& value) : base(size)
        {
            if (base::size_ == 0) return;

            auto al = std::allocator<value_type>();
            auto buf = al.allocate(base::size_);
            std::ranges::uninitialized_fill(buf, buf + base::size_, value);
            GPU_CHECK_ERROR(
                api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            GPU_CHECK_ERROR(api::gpuMemcpy(data(), buf, sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
            al.deallocate(buf, base::size_);
        }

        template <gpu_ptr T>
        __host__ explicit array(const T&) = delete;

        template <detail::array_convertible T, std::same_as<value_type> U = std::ranges::range_value_t<T>>
        requires std::ranges::contiguous_range<T>
        __host__ explicit array(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;

            GPU_CHECK_ERROR(
                api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            GPU_CHECK_ERROR(
                api::gpuMemcpy(data(), std::ranges::data(r), sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
        }

        template <detail::array_convertible T, typename U = std::ranges::range_value_t<T>>
        requires std::is_constructible_v<value_type, U> &&
                 (!std::same_as<value_type, U> || !std::ranges::contiguous_range<T>)
        __host__ explicit array(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;

            auto al = std::allocator<value_type>();
            auto buf = al.allocate(base::size_);
            for (auto i = std::size_t{0}; const auto& v : r) std::ranges::construct_at(buf + i++, v);

            GPU_CHECK_ERROR(
                api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            GPU_CHECK_ERROR(api::gpuMemcpy(data(), buf, sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
            al.deallocate(buf, base::size_);
        }

        __host__ array(std::initializer_list<value_type> r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(
                api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            GPU_CHECK_ERROR(
                api::gpuMemcpy(data(), std::ranges::data(r), sizeof(ValueType) * base::size_, gpuMemcpyHostToDevice));
        }

        __host__ __device__ array& operator=(const array& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ array& operator=(array&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }
        __host__ array& operator=(std::initializer_list<ValueType> r) noexcept
        {
            base::operator=(array(r));
            return *this;
        }

        template <gpu_array_ptr T>
        requires std::same_as<value_type, typename T::value_type>
        __host__ T to() const
        {
            if constexpr (gpu_unmanaged_array_ptr<T>)
            {
                auto result = T(base::size_, default_init);
                if (base::size_ == 0) return result;

                assert(data() != nullptr);
                GPU_CHECK_ERROR(
                    api::gpuMemcpy(result.data(), data(), sizeof(value_type) * base::size_, gpuMemcpyDeviceToDevice));
                return result;
            }
            else if constexpr (gpu_managed_array_ptr<T>)
            {
                auto result = T(base::size_, default_init);
                if (base::size_ == 0) return result;

                assert(data() != nullptr);
                GPU_CHECK_ERROR(
                    api::gpuMemcpy(result.data(), data(), sizeof(value_type) * base::size_, gpuMemcpyDeviceToHost));
                return result;
            }
            else
            {
                static_assert(detail::always_false_v<T>, "unsupported gpu_ptr type");
            }
        }

        template <detail::array_convertible T>
        requires std::same_as<value_type, std::ranges::range_value_t<T>> &&
                 ((std::ranges::contiguous_range<T> &&
                   (std::is_constructible_v<T, size_type> || std::is_default_constructible_v<T>)) ||
                  (std::is_default_constructible_v<T> &&
                   requires { std::declval<T>().push_back(std::declval<value_type>()); }))
        __host__ T to() const
        {
            if constexpr (std::ranges::contiguous_range<T>)
            {
                auto result = []([[maybe_unused]] size_type size) -> T {
                    if constexpr (std::is_constructible_v<T, size_type>)
                    {
                        // std::vector like type
                        return T(size);
                    }
                    else
                    {
                        // std::array like type
                        return T();
                    }
                }(this->size_);

                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");
                GPU_CHECK_ERROR(api::gpuMemcpy(std::ranges::data(result), data(), sizeof(value_type) * base::size_,
                                               gpuMemcpyDeviceToHost));
                return result;
            }
            else
            {
                // std::list like type
                static_assert(requires { std::declval<T>().push_back(std::declval<ValueType>()); });
                auto buf = std::make_unique_for_overwrite<ValueType[]>(base::size_);
                GPU_CHECK_ERROR(
                    api::gpuMemcpy(buf.get(), data(), sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));

                auto result = T();
                if constexpr (requires { std::declval<T>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }
                std::move(buf.get(), buf.get() + base::size_, std::back_inserter(result));
                return result;
            }
        }

        template <gpu_array_ptr T>
        requires std::is_constructible_v<value_type, typename T::value_type> &&
                 (!std::same_as<value_type, typename T::value_type>)
        __host__ T to() const
        {
            using U = typename T::value_type;
            auto buf = std::make_unique_for_overwrite<value_type[]>(base::size_);
            GPU_CHECK_ERROR(api::gpuMemcpy(buf.get(), data(), sizeof(value_type) * base::size_, gpuMemcpyDeviceToHost));

            if constexpr (gpu_unmanaged_array_ptr<T>)
            {
                auto result = T(base::size_, default_init);
                if (base::size_ == 0) return result;

                auto buf2 = std::make_unique_for_overwrite<U[]>(base::size_);
                assert(buf.get() != nullptr);
                assert(buf2.get() != nullptr);
                for (auto i = std::size_t{0}; i < base::size_; ++i)
                {
                    std::ranges::construct_at(buf2.get() + i, *(buf.get() + i));
                }

                assert(result.data() != nullptr);
                GPU_CHECK_ERROR(
                    api::gpuMemcpy(result.data(), buf2.get(), sizeof(U) * base::size_, gpuMemcpyHostToDevice));
                return result;
            }
            else if constexpr (gpu_managed_array_ptr<T>)
            {
                auto result = T(base::size_, default_init);
                if (base::size_ == 0) return result;

                assert(result.data() != nullptr);
                for (auto i = std::size_t{0}; i < base::size_; ++i)
                {
                    std::ranges::construct_at(result.data() + i, *(buf.get() + i));
                }
                return result;
            }
            else
            {
                static_assert(detail::always_false_v<T>, "unsupported gpu_ptr type");
            }
        }

        template <detail::array_convertible T>
        requires std::is_default_constructible_v<T> &&
                 requires(const value_type& v) { static_cast<std::ranges::range_value_t<T>>(v); } &&
                 (!std::same_as<value_type, std::ranges::range_value_t<T>>)
        __host__ T to() const
        {
            using U = std::ranges::range_value_t<T>;
            auto buf = std::make_unique_for_overwrite<value_type[]>(base::size_);
            GPU_CHECK_ERROR(api::gpuMemcpy(buf.get(), data(), sizeof(value_type) * base::size_, gpuMemcpyDeviceToHost));

            if constexpr (requires { std::declval<T>().push_back(std::declval<U>()); })
            {
                // back insertable type
                auto result = T();
                if constexpr (requires { std::declval<T>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }
                std::transform(buf.get(), buf.get() + base::size_, std::back_inserter(result),
                               [](const value_type& v) { return static_cast<U>(v); });
                return result;
            }
            else
            {
                // std::array like type
                auto result = T();
                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");
                std::transform(buf.get(), buf.get() + base::size_, std::ranges::begin(result),
                               [](const value_type& v) { return static_cast<U>(v); });
                return result;
            }
        }

        template <template <typename...> typename T>
        __host__ auto to() const
        {
            return to<T<value_type>>();
        }

        template <typename T>
        requires (!gpu_ptr<T>) && requires(const array& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

#if defined(GPU_OVERLOAD_HOST)
        __host__ array(pointer ptr, std::size_t size) : base(size)
        {
            if (size == 0) return;

            auto attr = api::gpuPointerAttributes{};
            GPU_CHECK_ERROR(api::gpuPointerGetAttributes(&attr, ptr));
            if (static_cast<api::gpuMemoryType>(attr.type) != api::gpuMemoryType::Device)
            {
                throw std::runtime_error("pointer type mismatch: expected device memory pointer");
            }
            std::get<0>(base::data_) = ptr;
        }
#endif
#if defined(GPU_OVERLOAD_DEVICE)
        __device__ array(pointer ptr, size_type size) : base(ptr, size)
        {
            if (size > 0) assert(ptr != nullptr);
        }
#endif

#if defined(GPU_OVERLOAD_HOST)
        __host__ void reset(pointer ptr, std::size_t size) { *this = array(ptr, size); }
#endif
#if defined(GPU_OVERLOAD_DEVICE)
        __device__ void reset(pointer ptr, size_type size) { *this = array(ptr, size); }
#endif
        __host__ __device__ void reset() noexcept
        {
#ifdef GPU_DEVICE_COMPILE
            reset(nullptr, 0);
#else
            base::free();
#endif
        }

        __host__ __device__ explicit operator bool() const noexcept { return data() != nullptr; }
    };

    template <typename ValueType, typename SizeType>
    class managed_array : public detail::base<true, SizeType, ValueType>
    {
        using base = detail::base<true, SizeType, ValueType>;
        static constexpr auto has_prefetch =
            requires(const ValueType& a, int device_id, api::gpuStream_t s) { a.prefetch(device_id, s); };
        static constexpr auto has_mem_advise = requires(const ValueType& a, api::gpuMemoryAdvise advise,
                                                        int device_id) { a.mem_advise(advise, device_id); };

    protected:
        // for jagged array initialization
        template <detail::array_convertible_for_copy Range>
        requires std::is_constructible_v<ValueType, std::ranges::range_value_t<std::ranges::range_value_t<Range>>>
        __host__ explicit managed_array(const Range& nested_array, detail::join_init_tag)
            : base(std::accumulate(std::ranges::begin(nested_array), std::ranges::end(nested_array), std::size_t{0},
                                   [](auto acc, const auto& r) { return acc + std::ranges::size(r); }))
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                  sizeof(ValueType) * base::size_));
            assert(data() != nullptr);

            for (auto i = std::size_t{0}; const auto& array : nested_array)
            {
                for (const auto& v : array) std::ranges::construct_at(data() + i++, v);
            }
        }

    public:
        using size_type = SizeType;
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator = value_type*;
        using const_iterator = const value_type*;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        __host__ __device__ const_reference operator[](size_type i) const noexcept { return data()[i]; };
        __host__ __device__ reference operator[](size_type i) noexcept { return data()[i]; }
        __host__ __device__ iterator begin() noexcept { return data(); }
        __host__ __device__ iterator end() noexcept { return data() + base::size_; }
        __host__ __device__ std::reverse_iterator<iterator> rbegin() noexcept
        {
            return std::reverse_iterator<iterator>(end());
        }
        __host__ __device__ std::reverse_iterator<iterator> rend() noexcept
        {
            return std::reverse_iterator<iterator>(begin());
        }
        __host__ __device__ const_iterator begin() const noexcept { return data(); }
        __host__ __device__ const_iterator end() const noexcept { return data() + base::size_; }
        __host__ __device__ std::reverse_iterator<const_iterator> rbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(end());
        }
        __host__ __device__ std::reverse_iterator<const_iterator> rend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(begin());
        }
        __host__ __device__ const_iterator cbegin() const noexcept { return data(); }
        __host__ __device__ const_iterator cend() const noexcept { return data() + base::size_; }
        __host__ __device__ std::reverse_iterator<const_iterator> crbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cend());
        }
        __host__ __device__ std::reverse_iterator<const_iterator> crend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cbegin());
        }
        __host__ __device__ reference front() noexcept { return *begin(); }
        __host__ __device__ const_reference front() const noexcept { return *begin(); }
        __host__ __device__ reference back() noexcept { return *(data() + base::size_ - 1); }
        __host__ __device__ const_reference back() const noexcept { return *(data() + base::size_ - 1); }
        __host__ __device__ pointer data() noexcept { return std::get<0>(base::data_); }
        __host__ __device__ pointer data() const noexcept { return std::get<0>(base::data_); }

        managed_array() = default;
        __host__ __device__ managed_array(const managed_array& r) : base(r) {}
        __host__ __device__ managed_array(managed_array&& r) noexcept : base(std::move(r)) {}

        __host__ explicit managed_array(std::size_t size) : base(size)
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                  sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            std::ranges::uninitialized_value_construct(*this);
        }

        __host__ explicit managed_array(std::size_t size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                  sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            std::ranges::uninitialized_default_construct(*this);
        }

        __host__ managed_array(std::size_t size, const value_type& value) : base(size)
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                  sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            std::ranges::uninitialized_fill(*this, value);
        }

        template <detail::array_convertible T, typename U = std::ranges::range_value_t<T>>
        requires std::is_constructible_v<value_type, U>
        __host__ explicit managed_array(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                  sizeof(value_type) * base::size_));
            assert(data() != nullptr);

            for (auto i = std::size_t{0}; const auto& v : r)
            {
                std::ranges::construct_at(data() + i++, v);
            }
        }

        __host__ managed_array(std::initializer_list<value_type> r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                  sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            for (auto i = std::size_t{0}; const auto& v : r) std::ranges::construct_at(data() + i++, v);
        }

        __host__ __device__ managed_array& operator=(const managed_array& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ managed_array& operator=(managed_array&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }
        __host__ managed_array& operator=(std::initializer_list<value_type> r) noexcept
        {
            base::operator=(managed_array(r));
            return *this;
        }

        __host__ void prefetch(size_type n, size_type len, int device_id, api::gpuStream_t stream = 0,
                               bool recursive = true) const
        {
            assert(n + len <= base::size_);
            if (len == 0) return;
            GPU_CHECK_ERROR(api::gpuMemPrefetchAsync(data() + n, sizeof(value_type) * len, device_id, stream));
            if constexpr (has_prefetch)
            {
                if (recursive)
                    for (auto i = n; i < n + len; ++i) data()[i].prefetch(device_id, stream, recursive);
            }
        }
        __host__ void prefetch(size_type n, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(n, len, detail::get_device_id(), stream, recursive);
        }
        __host__ void prefetch(int device_id, api::gpuStream_t stream = 0, bool recursive = true) const
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMemPrefetchAsync(data(), sizeof(value_type) * base::size_, device_id, stream));
            if constexpr (has_prefetch)
            {
                if (recursive)
                    for (std::size_t i = 0; i < base::size_; ++i) data()[i].prefetch(device_id, stream, recursive);
            }
        }
        __host__ void prefetch(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(detail::get_device_id(), stream, recursive);
        }

        __host__ void prefetch_to_cpu(size_type n, size_type len, api::gpuStream_t stream = 0,
                                      bool recursive = true) const
        {
            prefetch(n, len, gpuCpuDeviceId, stream, recursive);
        }
        __host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(gpuCpuDeviceId, stream, recursive);
        }

        __host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, int device_id,
                                 bool recursive = true) const
        {
            assert(n + len <= base::size_);
            if (len == 0) return;
            GPU_CHECK_ERROR(api::gpuMemAdvise(data() + n, sizeof(value_type) * len, advise, device_id));
            if constexpr (has_mem_advise)
            {
                if (recursive)
                    for (auto i = n; i < n + len; ++i) data()[i].mem_advise(advise, device_id, recursive);
            }
        }
        __host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(n, len, advise, detail::get_device_id(), recursive);
        }
        __host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMemAdvise(data(), sizeof(value_type) * base::size_, advise, device_id));
            if constexpr (has_mem_advise)
            {
                if (recursive)
                    for (std::size_t i = 0; i < base::size_; ++i) data()[i].mem_advise(advise, device_id, recursive);
            }
        }
        __host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, detail::get_device_id(), recursive);
        }

        __host__ void mem_advise_to_cpu(size_type n, size_type len, api::gpuMemoryAdvise advise,
                                        bool recursive = true) const
        {
            mem_advise(n, len, advise, gpuCpuDeviceId, recursive);
        }
        __host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, gpuCpuDeviceId, recursive);
        }

        template <gpu_array_ptr T>
        requires std::is_constructible_v<value_type, typename T::value_type>
        __host__ T to() const
        {
            using U = typename T::value_type;
            if constexpr (gpu_unmanaged_array_ptr<T>)
            {
                auto result = T(base::size_, default_init);
                if (base::size_ == 0) return result;

                if constexpr (std::is_same_v<value_type, U>)
                {
                    assert(result.data() != nullptr);
                    GPU_CHECK_ERROR(
                        api::gpuMemcpy(result.data(), data(), sizeof(U) * base::size_, gpuMemcpyHostToDevice));
                }
                else
                {
                    auto buf = std::make_unique_for_overwrite<U[]>(base::size_);
                    assert(buf.get() != nullptr);
                    for (auto i = std::size_t{0}; i < base::size_; ++i)
                    {
                        std::ranges::construct_at(buf.get() + i, *(data() + i));
                    }

                    assert(result.data() != nullptr);
                    GPU_CHECK_ERROR(
                        api::gpuMemcpy(result.data(), buf.get(), sizeof(U) * base::size_, gpuMemcpyHostToDevice));
                }
                return result;
            }
            else if constexpr (gpu_managed_array_ptr<T>)
            {
                auto result = T(base::size_, default_init);
                if (base::size_ == 0) return result;

                assert(result.data() != nullptr);
                for (auto i = std::size_t{0}; const auto& v : *this)
                {
                    std::ranges::construct_at(result.data() + i++, v);
                }
                return result;
            }
            else
            {
                static_assert(detail::always_false_v<T>, "unsupported gpu_ptr type");
            }
        }

        template <detail::array_convertible T>
        requires (gpu_array_ptr<T> || std::is_default_constructible_v<T>) &&
                 requires(const value_type& v) { static_cast<std::ranges::range_value_t<T>>(v); }
        __host__ T to() const
        {
            using U = std::ranges::range_value_t<T>;
            if constexpr (requires { std::declval<T>().push_back(std::declval<U>()); })
            {
                // back insertable type
                auto result = T();
                if constexpr (requires { std::declval<T>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }
                std::ranges::transform(*this, std::back_inserter(result),
                                       [](const value_type& v) { return static_cast<U>(v); });
                return result;
            }
            else
            {
                // std::array like type
                auto result = T();
                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");
                std::ranges::transform(*this, std::ranges::begin(result),
                                       [](const value_type& v) { return static_cast<U>(v); });
                return result;
            }
        }

        template <template <typename...> typename U>
        __host__ auto to() const
        {
            return to<detail::to_range_deduced_t<managed_array, U>>();
        }

        template <typename T>
        requires (!gpu_ptr<T>) && requires(const managed_array& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

#if defined(GPU_OVERLOAD_HOST)
        __host__ managed_array(pointer ptr, std::size_t size) : base(size)
        {
            if (size == 0) return;

            auto attr = api::gpuPointerAttributes{};
            GPU_CHECK_ERROR(api::gpuPointerGetAttributes(&attr, ptr));
            if (static_cast<api::gpuMemoryType>(attr.type) != api::gpuMemoryType::Managed)
            {
                throw std::runtime_error("pointer type mismatch: expected managed memory pointer");
            }
            std::get<0>(base::data_) = ptr;
        }
#endif
#if defined(GPU_OVERLOAD_DEVICE)
        __device__ managed_array(pointer ptr, size_type size) : base(ptr, size)
        {
            if (size > 0) assert(ptr != nullptr);
        }
#endif

#if defined(GPU_OVERLOAD_HOST)
        __host__ void reset(pointer ptr, std::size_t size) { *this = managed_array(ptr, size); }
#endif
#if defined(GPU_OVERLOAD_DEVICE)
        __device__ void reset(pointer ptr, size_type size) { *this = managed_array(ptr, size); }
#endif
        __host__ __device__ void reset() noexcept
        {
#ifdef GPU_DEVICE_COMPILE
            reset(nullptr, 0);
#else
            base::free();
#endif
        }

        __host__ __device__ explicit operator bool() const noexcept { return data() != nullptr; }
    };

    // deduction guide for arrays
    template <detail::array_convertible T>
    array(const T& r) -> array<std::ranges::range_value_t<T>>;
    template <detail::array_convertible T>
    managed_array(const T& r) -> managed_array<detail::unified_array_deduced_t<std::ranges::range_value_t<T>>>;

    template <typename ValueType>
    requires std::is_trivially_copyable_v<ValueType>
    class value : protected detail::base<false, std::uint32_t, ValueType>
    {
        using base = detail::base<false, std::uint32_t, ValueType>;
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

    public:
        using element_type = value_type;
        using base::use_count;

        __host__ __device__ value(const value& r) : base(r) {};
        __host__ __device__ value(value&& r) noexcept : base(std::move(r)) {};

        __host__ value() noexcept : base() {}

        __host__ explicit value(default_init_tag) : base(1)
        {
            GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type)));
            assert(get() != nullptr);
            if constexpr (!std::is_trivially_default_constructible_v<value_type>)
            {
                auto temp = value_type();
                GPU_CHECK_ERROR(api::gpuMemcpy(get(), &temp, sizeof(value_type), gpuMemcpyHostToDevice));
            }
        }

        __host__ explicit value(const value_type& r) : base(1)
        {
            GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type)));
            assert(get() != nullptr);
            GPU_CHECK_ERROR(api::gpuMemcpy(get(), &r, sizeof(value_type), gpuMemcpyHostToDevice));
        }

        template <typename... Args>
        requires std::is_constructible_v<value_type, Args...>
        __host__ explicit value(Args&&... args) : base(1)
        {
            auto temp = value_type(std::forward<Args>(args)...);
            GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type)));
            assert(get() != nullptr);
            GPU_CHECK_ERROR(api::gpuMemcpy(get(), &temp, sizeof(value_type), gpuMemcpyHostToDevice));
        }

        __host__ __device__ explicit value(pointer ptr)
#if !defined(GPU_DEVICE_COMPILE)
            : base(ptr == nullptr ? 0 : 1)
        {
            if (ptr == nullptr) return;
            auto attr = api::gpuPointerAttributes{};
            GPU_CHECK_ERROR(api::gpuPointerGetAttributes(&attr, ptr));
            if (static_cast<api::gpuMemoryType>(attr.type) != api::gpuMemoryType::Device)
            {
                throw std::runtime_error("pointer type mismatch: expected device memory pointer");
            }
            std::get<0>(base::data_) = ptr;
        }
#else
            : base(ptr, ptr == nullptr ? 0 : 1)
        {
        }
#endif

        __host__ __device__ value& operator=(const value& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ value& operator=(value&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }

        __host__ __device__ pointer get() noexcept { return std::get<0>(base::data_); }
        __host__ __device__ const_pointer get() const noexcept { return std::get<0>(base::data_); }
        __host__ __device__ explicit operator bool() const noexcept { return std::get<0>(base::data_) != nullptr; }

#if defined(GPU_OVERLOAD_DEVICE)
        __device__ const_reference operator*() const noexcept
        {
            assert(get() != nullptr);
            return *get();
        }
        __device__ reference operator*() noexcept
        {
            assert(get() != nullptr);
            return *get();
        }
        __device__ const_pointer operator->() const noexcept
        {
            assert(get() != nullptr);
            return get();
        }
        __device__ pointer operator->() noexcept
        {
            assert(get() != nullptr);
            return get();
        }
#endif
#if defined(GPU_OVERLOAD_HOST)
        __host__ element_type operator*() const
        {
            value_type temp;
            GPU_CHECK_ERROR(api::gpuMemcpy(&temp, get(), sizeof(value_type), gpuMemcpyDeviceToHost));
            return temp;
        }
        __host__ auto operator->() const
        {
            // proxy object
            struct
            {
                value_type t;
                __forceinline__ const_pointer operator->() const { return &t; }
            } proxy;

            GPU_CHECK_ERROR(api::gpuMemcpy(&proxy.t, get(), sizeof(value_type), gpuMemcpyDeviceToHost));
            return proxy;
        }
#endif
        __host__ __device__ void reset(pointer ptr) { *this = value(ptr); }
        __host__ __device__ void reset() noexcept
        {
#ifdef GPU_DEVICE_COMPILE
            reset(nullptr);
#else
            base::free();
#endif
        }
    };

    template <typename ValueType>
    class managed_value : protected detail::base<true, std::uint32_t, ValueType>
    {
        using base = detail::base<true, std::uint32_t, ValueType>;
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        static constexpr auto has_prefetch =
            requires(const ValueType& a, int device_id, api::gpuStream_t s) { a.prefetch(device_id, s); };
        static constexpr auto has_mem_advise = requires(const ValueType& a, api::gpuMemoryAdvise advise,
                                                        int device_id) { a.mem_advise(advise, device_id); };

    public:
        using element_type = ValueType;
        using base::use_count;

        __host__ __device__ managed_value(const managed_value& r) : base(r) {};
        __host__ __device__ managed_value(managed_value&& r) noexcept : base(std::move(r)) {};

        __host__ managed_value() noexcept : base() {}

        __host__ explicit managed_value(default_init_tag) : base(1)
        {
            GPU_CHECK_ERROR(
                api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::uninitialized_default_construct_n(get(), 1);
        }

        __host__ explicit managed_value(const ValueType& r) : base(1)
        {
            GPU_CHECK_ERROR(
                api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::construct_at(get(), r);
        }

        __host__ explicit managed_value(ValueType&& r) : base(1)
        {
            GPU_CHECK_ERROR(
                api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::construct_at(get(), std::move(r));
        }

        template <typename... Args>
        requires std::is_constructible_v<ValueType, Args...>
        __host__ explicit managed_value(Args&&... args) : base(1)
        {
            GPU_CHECK_ERROR(
                api::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::construct_at(get(), std::forward<Args>(args)...);
        }

        __host__ __device__ explicit managed_value(pointer ptr)
#if !defined(GPU_DEVICE_COMPILE)
            : base(ptr == nullptr ? 0 : 1)
        {
            if (ptr == nullptr) return;
            auto attr = api::gpuPointerAttributes{};
            GPU_CHECK_ERROR(api::gpuPointerGetAttributes(&attr, ptr));
            if (static_cast<api::gpuMemoryType>(attr.type) != api::gpuMemoryType::Managed)
            {
                throw std::runtime_error("pointer type mismatch: expected managed memory pointer");
            }
            std::get<0>(base::data_) = ptr;
        }
#else
            : base(ptr, ptr == nullptr ? 0 : 1)
        {
        }
#endif

        __host__ __device__ managed_value& operator=(const managed_value& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ managed_value& operator=(managed_value&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }

        __host__ __device__ const_reference operator*() const noexcept
        {
            assert(get() != nullptr);
            return *get();
        }
        __host__ __device__ reference operator*() noexcept
        {
            assert(get() != nullptr);
            return *get();
        }
        __host__ __device__ const_pointer operator->() const noexcept
        {
            assert(get() != nullptr);
            return get();
        }
        __host__ __device__ pointer operator->() noexcept
        {
            assert(get() != nullptr);
            return get();
        }

        __host__ __device__ pointer get() noexcept { return std::get<0>(base::data_); }
        __host__ __device__ const_pointer get() const noexcept { return std::get<0>(base::data_); }
        __host__ __device__ explicit operator bool() const noexcept { return std::get<0>(base::data_) != nullptr; }

        __host__ void prefetch(int device_id, api::gpuStream_t stream = 0, bool recursive = true) const
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMemPrefetchAsync(get(), sizeof(ValueType), device_id, stream));
            if constexpr (has_prefetch)
                if (recursive) get()->prefetch(device_id, stream, recursive);
        }
        __host__ void prefetch(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(detail::get_device_id(), stream, recursive);
        }

        __host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(gpuCpuDeviceId, stream, recursive);
        }

        __host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const
        {
            if (base::size_ == 0) return;
            GPU_CHECK_ERROR(api::gpuMemAdvise(get(), sizeof(ValueType), advise, device_id));
            if constexpr (has_mem_advise)
                if (recursive) get()->mem_advise(advise, device_id, recursive);
        }
        __host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, detail::get_device_id(), recursive);
        }

        __host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, gpuCpuDeviceId, recursive);
        }
        __host__ __device__ void reset(pointer ptr) { *this = managed_value(ptr); }
        __host__ __device__ void reset() noexcept
        {
#ifdef GPU_DEVICE_COMPILE
            reset(nullptr);
#else
            base::free();
#endif
        }
    };

    namespace detail
    {
        template <std::size_t N, typename Tuple, typename... Ts>
        constexpr bool assignable_to_tuple_helper_n()
        {
            return requires(const Tuple& t1, std::tuple<Ts...>& t2) {
                std::get<N>(t1);
                std::get<N>(t2);
                requires std::assignable_from<decltype(std::get<N>(t2)), decltype(std::get<N>(t1))>;
            };
        }
        template <typename Tuple, typename... Ts>
        constexpr bool assignable_to_tuple_helper()
        {
            return []<std::size_t... N>(std::index_sequence<N...>) {
                return (assignable_to_tuple_helper_n<N, Tuple, Ts...>() && ...);
            }(std::make_index_sequence<sizeof...(Ts)>());
        }
        template <typename Tuple, typename... Ts>
        concept assignable_to_tuple = assignable_to_tuple_helper<Tuple, Ts...>();
    }  // namespace detail

    template <template <typename...> typename Tuple, typename... Ts>
    class structure_of_arrays_iterator
    {
        std::tuple<Ts*...> ptrs_;

    public:
        using difference_type = std::ptrdiff_t;
        using size_type = std::size_t;
        using value_type = Tuple<Ts...>;
        using iterator_concept = std::random_access_iterator_tag;

        structure_of_arrays_iterator() = default;
        structure_of_arrays_iterator(const structure_of_arrays_iterator&) = default;
        structure_of_arrays_iterator(structure_of_arrays_iterator&&) noexcept = default;

        structure_of_arrays_iterator& operator=(const structure_of_arrays_iterator&) = default;
        structure_of_arrays_iterator& operator=(structure_of_arrays_iterator&&) noexcept = default;

        __host__ __device__ explicit structure_of_arrays_iterator(std::tuple<Ts*...> ptrs) : ptrs_(ptrs) {}

        __host__ __device__ Tuple<Ts&...> operator*() const
        {
            return std::apply([](auto*... ptrs) { return Tuple<Ts&...>(*ptrs...); }, ptrs_);
        }
        __host__ __device__ Tuple<Ts&...> operator[](size_type n) const
        {
            return std::apply([n](auto*... ptrs) { return Tuple<Ts&...>(ptrs[n]...); }, ptrs_);
        }
        __host__ __device__ auto operator->() const
        {
            struct
            {
                Tuple<Ts&...> t;
                __host__ __device__ inline auto* operator->() { return &t; }
            } cap{**this};
            return cap;
        }
        __host__ __device__ structure_of_arrays_iterator& operator++()
        {
            std::apply([](auto*&... ptrs) { (++ptrs, ...); }, ptrs_);
            return *this;
        }
        __host__ __device__ structure_of_arrays_iterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        __host__ __device__ structure_of_arrays_iterator& operator+=(difference_type n)
        {
            std::apply([n](auto*&... ptrs) { ((ptrs += n), ...); }, ptrs_);
            return *this;
        }
        __host__ __device__ structure_of_arrays_iterator& operator--()
        {
            std::apply([](auto*&... ptrs) { (--ptrs, ...); }, ptrs_);
            return *this;
        }
        __host__ __device__ structure_of_arrays_iterator operator--(int)
        {
            auto tmp = *this;
            --(*this);
            return tmp;
        }
        __host__ __device__ structure_of_arrays_iterator& operator-=(difference_type n)
        {
            std::apply([n](auto*&... ptrs) { ((ptrs -= n), ...); }, ptrs_);
            return *this;
        }

        __host__ __device__ friend difference_type operator-(const structure_of_arrays_iterator& lhs,
                                                             const structure_of_arrays_iterator& rhs)
        {
            return std::get<0>(lhs.ptrs_) - std::get<0>(rhs.ptrs_);
        }
        __host__ __device__ friend structure_of_arrays_iterator operator+(const structure_of_arrays_iterator& lhs,
                                                                          difference_type n)
        {
            return structure_of_arrays_iterator(
                std::apply([n](auto*... ptrs) { return std::tuple{ptrs + n...}; }, lhs.ptrs_));
        }
        __host__ __device__ friend structure_of_arrays_iterator operator+(structure_of_arrays_iterator&& lhs,
                                                                          difference_type n)
        {
            lhs += n;
            return std::move(lhs);
        }
        __host__ __device__ friend structure_of_arrays_iterator operator+(difference_type n,
                                                                          const structure_of_arrays_iterator& rhs)
        {
            return rhs + n;
        }
        __host__ __device__ friend structure_of_arrays_iterator operator+(difference_type n,
                                                                          structure_of_arrays_iterator&& rhs)
        {
            return std::move(rhs) + n;
        }
        __host__ __device__ friend structure_of_arrays_iterator operator-(const structure_of_arrays_iterator& lhs,
                                                                          difference_type n)
        {
            return structure_of_arrays_iterator(
                std::apply([n](auto*... ptrs) { return std::tuple{ptrs - n...}; }, lhs.ptrs_));
        }
        __host__ __device__ friend structure_of_arrays_iterator operator-(structure_of_arrays_iterator&& lhs,
                                                                          difference_type n)
        {
            lhs -= n;
            return std::move(lhs);
        }

        __host__ __device__ friend bool operator==(const structure_of_arrays_iterator& lhs,
                                                   const structure_of_arrays_iterator& rhs)
        {
            return std::get<0>(lhs.ptrs_) == std::get<0>(rhs.ptrs_);
        }
        __host__ __device__ friend std::strong_ordering operator<=>(const structure_of_arrays_iterator& lhs,
                                                                    const structure_of_arrays_iterator& rhs)
        {
            return std::get<0>(lhs.ptrs_) <=> std::get<0>(rhs.ptrs_);
        }
        __host__ __device__ friend auto iter_move(const structure_of_arrays_iterator& x)
        {
            return std::apply(
                [](auto*... ptrs) {
                    using RetType = std::remove_cvref_t<decltype(x)>::value_type;
                    return RetType(std::move(*ptrs)...);
                },
                x.ptrs_);
        }
        __host__ __device__ friend void iter_swap(const structure_of_arrays_iterator& lhs,
                                                  const structure_of_arrays_iterator& rhs) noexcept
        {
            constexpr std::size_t size = std::tuple_size_v<std::remove_cvref_t<decltype(lhs.ptrs_)>>;
            [&lhs, &rhs]<std::size_t... N>(std::index_sequence<N...>) {
                (std::swap(*std::get<N>(lhs.ptrs_), *std::get<N>(rhs.ptrs_)), ...);
            }(std::make_index_sequence<size>());
        }
    };

    template <typename... Ts>
    class structure_of_arrays : public structure_of_arrays<std::tuple<Ts...>, size_type_default>
    {
        using base = structure_of_arrays<std::tuple<Ts...>, size_type_default>;
        using base::base;

    public:
        using size_type = size_type_default;
        template <std::size_t N>
        using element_type = base::template element_type<N>;
        using base::operator=;
    };

    template <template <typename...> typename Tuple, typename... Ts>
    class structure_of_arrays<Tuple<Ts...>> : public structure_of_arrays<Tuple<Ts...>, size_type_default>
    {
        using base = structure_of_arrays<Tuple<Ts...>, size_type_default>;
        using base::base;

    public:
        using size_type = size_type_default;
        template <std::size_t N>
        using element_type = base::template element_type<N>;
        using base::operator=;
    };

    template <template <typename...> typename Tuple, typename... Ts, typename SizeType>
    requires (sizeof...(Ts) > 0) && std::constructible_from<Tuple<Ts...>, const Ts&...> &&
             std::constructible_from<Tuple<Ts&...>, Ts&...> &&
             std::constructible_from<Tuple<const Ts&...>, const Ts&...> && (std::is_trivially_copyable_v<Ts> && ...)
    class structure_of_arrays<Tuple<Ts...>, SizeType> : public detail::base<false, SizeType, Ts...>
    {
        static constexpr auto num_arrays = sizeof...(Ts);
        using base = detail::base<false, SizeType, Ts...>;

        using tuple_value_type = std::tuple<Ts...>;
        using tuple_pointer_type = std::tuple<Ts*...>;
        using tuple_const_pointer_type = std::tuple<const Ts*...>;
        using ret_tuple_value_type = Tuple<Ts...>;
        using ret_tuple_reference_type = Tuple<Ts&...>;
        using ret_tuple_const_reference_type = Tuple<const Ts&...>;
        using iterator_type = structure_of_arrays_iterator<Tuple, Ts...>;
        using const_iterator_type = structure_of_arrays_iterator<Tuple, const Ts...>;

    public:
        using size_type = SizeType;
        template <std::size_t N>
        using element_type = std::tuple_element_t<N, tuple_value_type>;

        SIGSEGV_DEPRECATED __host__ __device__ auto begin() noexcept { return iterator_type(base::data_); }
        SIGSEGV_DEPRECATED __host__ __device__ auto end() noexcept
        {
            return std::apply(
                [this](auto&... ptrs) { return iterator_type(tuple_pointer_type{(ptrs + base::size_)...}); },
                base::data_);
        }
        SIGSEGV_DEPRECATED __host__ __device__ auto begin() const noexcept
        {
            return std::apply([](auto&... ptrs) { return const_iterator_type(tuple_const_pointer_type{(ptrs)...}); },
                              base::data_);
        }
        SIGSEGV_DEPRECATED __host__ __device__ auto end() const noexcept
        {
            return std::apply(
                [this](auto&... ptrs) {
                    return const_iterator_type(tuple_const_pointer_type{(ptrs + base::size_)...});
                },
                base::data_);
        }
        SIGSEGV_DEPRECATED __host__ __device__ auto operator[](size_type i) &
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_reference_type{*(ptrs + i)...}; }, base::data_);
        }
        SIGSEGV_DEPRECATED __host__ __device__ auto operator[](size_type i) const&
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_const_reference_type{*(ptrs + i)...}; },
                              base::data_);
        }
        SIGSEGV_DEPRECATED __host__ __device__ auto operator[](size_type i) &&
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_value_type{*(ptrs + i)...}; }, base::data_);
        }
        template <std::size_t N>
        __host__ __device__ auto* data() noexcept
        {
            return std::get<N>(base::data_);
        }
        template <std::size_t N>
        __host__ __device__ const auto* data() const noexcept
        {
            return std::get<N>(base::data_);
        }

        structure_of_arrays() = default;
        __host__ __device__ structure_of_arrays(const structure_of_arrays& r) : base(r) {}
        __host__ __device__ structure_of_arrays(structure_of_arrays&& r) noexcept : base(std::move(r)) {}

        __host__ explicit structure_of_arrays(std::size_t size) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                auto buf = std::make_unique<T[]>(base::size_);
                GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                GPU_CHECK_ERROR(api::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            });
        }

        __host__ explicit structure_of_arrays(std::size_t size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                if constexpr (!std::is_trivially_default_constructible_v<T>)
                {
                    auto buf = std::make_unique<T[]>(base::size_);
                    GPU_CHECK_ERROR(api::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
                }
            });
        }

        __host__ structure_of_arrays(std::size_t size, const ret_tuple_value_type& value) : base(size)
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this]<typename T>(T*& ptr, const T& v) {
                GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                auto buf = std::make_unique_for_overwrite<T[]>(base::size_);
                std::ranges::uninitialized_fill_n(buf.get(), base::size_, v);
                GPU_CHECK_ERROR(api::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            };

            [this, &value, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), std::get<N>(value)), ...);
            }(std::make_index_sequence<num_arrays>());
        }

        template <detail::array_convertible_for_copy Range>
        requires detail::assignable_to_tuple<std::ranges::range_value_t<Range>, Ts...>
        __host__ explicit structure_of_arrays(const Range& array) : base(std::ranges::size(array))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this]<typename T>(T*& ptr, auto view_range) {
                auto buf = std::make_unique_for_overwrite<T[]>(base::size_);
                std::ranges::copy(view_range, buf.get());
                GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                GPU_CHECK_ERROR(api::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            };

            [this, &array, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_),
                           array | std::views::transform([](const auto& v) { return (std::get<N>(v)); })),
                 ...);
            }(std::make_index_sequence<num_arrays>());
        }

        __host__ structure_of_arrays(std::initializer_list<ret_tuple_value_type> list) : base(std::ranges::size(list))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this]<typename T>(T*& ptr, auto view_range) {
                auto buf = std::make_unique_for_overwrite<T[]>(base::size_);
                std::ranges::copy(view_range, buf.get());
                GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                GPU_CHECK_ERROR(api::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            };

            [this, &list, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_),
                           list | std::views::transform([](const auto& v) { return (std::get<N>(v)); })),
                 ...);
            }(std::make_index_sequence<num_arrays>());
        }

        template <detail::array_convertible_for_copy... Ranges>
        requires (sizeof...(Ranges) == num_arrays) &&
                 detail::assignable_to_tuple<std::tuple<std::ranges::range_value_t<Ranges>...>, Ts...>
        __host__ explicit structure_of_arrays(const Ranges&... arrays) : base(std::max({std::ranges::size(arrays)...}))
        {
            if (base::size_ == 0) return;

            auto sizes = std::array{std::ranges::size(arrays)...};
            if (!std::ranges::all_of(sizes, [s = base::size_](auto x) { return x == s; }))
            {
                throw std::invalid_argument("the sizes of arrays are not equal");
            }

            const auto alloc_ptr = [this]<typename T>(T*& ptr, const auto& range) {
                auto buf = std::make_unique_for_overwrite<T[]>(base::size_);
                std::ranges::copy(range, buf.get());
                GPU_CHECK_ERROR(api::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                GPU_CHECK_ERROR(api::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            };

            auto arrays_tuple = std::tuple<const Ranges&...>(arrays...);
            [this, &arrays_tuple, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), std::get<N>(arrays_tuple)), ...);
            }(std::make_index_sequence<num_arrays>());
        }

        __host__ __device__ structure_of_arrays& operator=(const structure_of_arrays& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ structure_of_arrays& operator=(structure_of_arrays&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }
        __host__ structure_of_arrays& operator=(std::initializer_list<ret_tuple_value_type> r) noexcept
        {
            base::operator=(structure_of_arrays(r));
            return *this;
        }

        template <detail::array_convertible Range>
        requires std::is_default_constructible_v<Range> &&
                 std::is_constructible_v<std::ranges::range_value_t<Range>, Ts...>
        __host__ Range to() const
        {
            using U = std::ranges::range_value_t<Range>;
            const auto copy_buffer = [this]<typename T>(const T* ptr) {
                auto buf = std::make_unique_for_overwrite<T[]>(base::size_);
                GPU_CHECK_ERROR(api::gpuMemcpy(buf.get(), ptr, sizeof(T) * base::size_, gpuMemcpyDeviceToHost));
                return buf;
            };

            const auto tmp_bufs = std::apply(
                [copy_buffer](const auto*... ptrs) {
                    return std::tuple<std::unique_ptr<Ts[]>...>(copy_buffer(ptrs)...);
                },
                base::data_);

            auto result = Range();

            if constexpr (requires { std::declval<Range>().push_back(std::declval<U>()); })
            {
                // back insert-able type
                if constexpr (requires { std::declval<Range>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }

                for (size_type i = 0; i < base::size_; ++i)
                {
                    std::apply([&result, i](const auto&... bufs) { result.push_back(U{bufs[i]...}); }, tmp_bufs);
                }

                return result;
            }
            else
            {
                // std::array like type
                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");

                for (size_type i = 0; i < base::size_; ++i)
                {
                    std::apply([&result, i](const auto&... bufs) { result[i] = U{bufs[i]...}; }, tmp_bufs);
                }

                return result;
            }
        }

        template <template <typename...> typename T>
        __host__ auto to() const
        {
            return to<T<Tuple<Ts...>>>();
        }

        template <typename T>
        requires (!gpu_ptr<T>) && requires(const structure_of_arrays& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

        template <std::size_t N>
        __device__ void reset(std::tuple_element_t<N, tuple_pointer_type> ptr)
        {
            assert(base::size_ == 0 || ptr != nullptr);

            // reset specified pointer only
            std::get<N>(base::data_) = base::size_ == 0 ? nullptr : ptr;
        }

        template <std::size_t N, gpu_array_ptr T>
        __device__ void reset(const T& array)
        {
            assert(array.size() == base::size_);
            reset<N>(array.data());
        }

        __host__ void reset() { base::free(); }

        __host__ __device__ explicit operator bool() const noexcept
        {
            // If not empty, all pointers are valid
            return base::size_ != 0;
        }
    };

    template <typename... Ts>
    class managed_structure_of_arrays : public managed_structure_of_arrays<std::tuple<Ts...>, size_type_default>
    {
        using base = managed_structure_of_arrays<std::tuple<Ts...>, size_type_default>;
        using base::base;

    public:
        using size_type = size_type_default;
        template <std::size_t N>
        using element_type = base::template element_type<N>;
        using base::operator=;
    };

    template <template <typename...> typename Tuple, typename... Ts>
    class managed_structure_of_arrays<Tuple<Ts...>>
        : public managed_structure_of_arrays<Tuple<Ts...>, size_type_default>
    {
        using base = managed_structure_of_arrays<Tuple<Ts...>, size_type_default>;
        using base::base;

    public:
        using size_type = size_type_default;
        template <std::size_t N>
        using element_type = base::template element_type<N>;
        using base::operator=;
    };

    template <template <typename...> typename Tuple, typename... Ts, typename SizeType>
    requires (sizeof...(Ts) > 0) && std::constructible_from<Tuple<Ts...>, const Ts&...> &&
             std::constructible_from<Tuple<Ts&...>, Ts&...> &&
             std::constructible_from<Tuple<const Ts&...>, const Ts&...>
    class managed_structure_of_arrays<Tuple<Ts...>, SizeType> : public detail::base<true, SizeType, Ts...>
    {
        static constexpr auto num_arrays = sizeof...(Ts);
        using base = detail::base<true, SizeType, Ts...>;

        using tuple_value_type = std::tuple<Ts...>;
        using tuple_pointer_type = std::tuple<Ts*...>;
        using tuple_const_pointer_type = std::tuple<const Ts*...>;
        using ret_tuple_value_type = Tuple<Ts...>;
        using ret_tuple_reference_type = Tuple<Ts&...>;
        using ret_tuple_const_reference_type = Tuple<const Ts&...>;
        using iterator_type = structure_of_arrays_iterator<Tuple, Ts...>;
        using const_iterator_type = structure_of_arrays_iterator<Tuple, const Ts...>;

        template <typename T>
        static constexpr auto has_prefetch =
            requires(const T& a, int device_id, api::gpuStream_t s) { a.prefetch(device_id, s); };
        template <typename T>
        static constexpr auto has_mem_advise =
            requires(const T& a, api::gpuMemoryAdvise advise, int device_id) { a.mem_advise(advise, device_id); };

    protected:
        // for jagged array initialization
        template <detail::array_convertible_for_copy Range, typename U = std::ranges::range_value_t<Range>>
        requires (detail::assignable_to_tuple<std::ranges::range_value_t<U>, Ts...>)
        __host__ explicit managed_structure_of_arrays(const Range& nested_array, detail::join_init_tag)
            : base(std::accumulate(std::ranges::begin(nested_array), std::ranges::end(nested_array), std::size_t{0},
                                   [](auto acc, const auto& r) { return acc + std::ranges::size(r); }))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this, &nested_array]<typename T>(T*& ptr, auto f) {
                GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);

                for (auto i = std::size_t{0}; const auto& array : nested_array)
                {
                    for (const auto& v : array) std::ranges::construct_at(ptr + i++, f(v));
                }
            };

            [this, &alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), [](const auto& v) -> const auto& { return (std::get<N>(v)); }),
                 ...);
            }(std::make_index_sequence<num_arrays>());
        }

    public:
        using size_type = SizeType;
        template <std::size_t N>
        using element_type = std::tuple_element_t<N, tuple_value_type>;

        __host__ __device__ auto begin() noexcept { return iterator_type(base::data_); }
        __host__ __device__ auto end() noexcept
        {
            return std::apply(
                [this](auto&... ptrs) { return iterator_type(tuple_pointer_type{(ptrs + base::size_)...}); },
                base::data_);
        }
        __host__ __device__ auto begin() const noexcept
        {
            return std::apply([](auto&... ptrs) { return const_iterator_type(tuple_const_pointer_type{(ptrs)...}); },
                              base::data_);
        }
        __host__ __device__ auto end() const noexcept
        {
            return std::apply(
                [this](auto&... ptrs) {
                    return const_iterator_type(tuple_const_pointer_type{(ptrs + base::size_)...});
                },
                base::data_);
        }
        __host__ __device__ auto operator[](size_type i) &
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_reference_type{*(ptrs + i)...}; }, base::data_);
        }
        __host__ __device__ auto operator[](size_type i) const&
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_const_reference_type{*(ptrs + i)...}; },
                              base::data_);
        }
        __host__ __device__ auto operator[](size_type i) &&
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_value_type{*(ptrs + i)...}; }, base::data_);
        }
        template <std::size_t N>
        __host__ __device__ auto* data() noexcept
        {
            return std::get<N>(base::data_);
        }
        template <std::size_t N>
        __host__ __device__ const auto* data() const noexcept
        {
            return std::get<N>(base::data_);
        }

        managed_structure_of_arrays() = default;
        __host__ __device__ managed_structure_of_arrays(const managed_structure_of_arrays& r) : base(r) {}
        __host__ __device__ managed_structure_of_arrays(managed_structure_of_arrays&& r) noexcept : base(std::move(r))
        {
        }

        __host__ explicit managed_structure_of_arrays(std::size_t size) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                std::ranges::uninitialized_value_construct(ptr, ptr + base::size_);
            });
        }

        __host__ explicit managed_structure_of_arrays(std::size_t size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                std::ranges::uninitialized_default_construct(ptr, ptr + base::size_);
            });
        }

        __host__ managed_structure_of_arrays(std::size_t size, const ret_tuple_value_type& value) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
            });

            [this, &value]<std::size_t... N>(std::index_sequence<N...>) {
                (std::ranges::uninitialized_fill_n(std::get<N>(base::data_), base::size_, std::get<N>(value)), ...);
            }(std::make_index_sequence<num_arrays>());
        }

        template <detail::array_convertible_for_copy Range>
        requires detail::assignable_to_tuple<std::ranges::range_value_t<Range>, Ts...>
        __host__ explicit managed_structure_of_arrays(const Range& array) : base(std::ranges::size(array))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this, &array]<typename T>(T*& ptr, auto f) {
                GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);

                for (auto i = std::size_t{0}; const auto& v : array)
                {
                    std::ranges::construct_at(ptr + i++, f(v));
                }
            };

            [this, &alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), [](const auto& e) -> const auto& { return (std::get<N>(e)); }),
                 ...);
            }(std::make_index_sequence<num_arrays>());
        }

        __host__ managed_structure_of_arrays(std::initializer_list<ret_tuple_value_type> list)
            : base(std::ranges::size(list))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this, &list]<typename T>(T*& ptr, auto f) {
                GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);

                for (auto i = std::size_t{0}; const auto& v : list)
                {
                    std::ranges::construct_at(ptr + i++, f(v));
                }
            };

            [this, &alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), [](const auto& e) -> const auto& { return (std::get<N>(e)); }),
                 ...);
            }(std::make_index_sequence<num_arrays>());
        }

        template <detail::array_convertible_for_copy... Ranges>
        requires (sizeof...(Ranges) == num_arrays) &&
                 detail::assignable_to_tuple<std::tuple<std::ranges::range_value_t<Ranges>...>, Ts...>
        __host__ explicit managed_structure_of_arrays(const Ranges&... arrays)
            : base(std::max({std::ranges::size(arrays)...}))
        {
            if (base::size_ == 0) return;

            auto sizes = std::array{std::ranges::size(arrays)...};
            if (!std::ranges::all_of(sizes, [s = base::size_](auto x) { return x == s; }))
            {
                throw std::invalid_argument("the sizes of arrays are not equal");
            }

            const auto alloc_ptr = [this]<typename T>(T*& ptr, const auto& range) {
                GPU_CHECK_ERROR(api::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                for (auto i = std::size_t{0}; const auto& v : range) std::ranges::construct_at(ptr + i++, v);
            };

            auto arrays_tuple = std::tuple<const Ranges&...>(arrays...);
            [this, &arrays_tuple, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), std::get<N>(arrays_tuple)), ...);
            }(std::make_index_sequence<num_arrays>());
        }

        __host__ __device__ managed_structure_of_arrays& operator=(const managed_structure_of_arrays& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ managed_structure_of_arrays& operator=(managed_structure_of_arrays&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }
        __host__ managed_structure_of_arrays& operator=(std::initializer_list<ret_tuple_value_type> r) noexcept
        {
            base::operator=(managed_structure_of_arrays(r));
            return *this;
        }

        __host__ void prefetch(size_type n, size_type len, int device_id, api::gpuStream_t stream = 0,
                               bool recursive = true) const
        {
            assert(n + len <= base::size_);
            if (len == 0) return;
            base::tuple_for_each([n, len, device_id, stream, recursive]<typename T>(T* ptr) {
                GPU_CHECK_ERROR(api::gpuMemPrefetchAsync(ptr + n, sizeof(T) * len, device_id, stream));
                if constexpr (has_prefetch<T>)
                {
                    if (recursive)
                        for (auto i = n; i < n + len; ++i) ptr[i].prefetch(device_id, stream, recursive);
                }
            });
        }
        __host__ void prefetch(size_type n, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(n, len, detail::get_device_id(), stream, recursive);
        }
        __host__ void prefetch(int device_id, api::gpuStream_t stream = 0, bool recursive = true) const
        {
            if (base::size_ == 0) return;
            base::tuple_for_each([this, device_id, stream, recursive]<typename T>(T* ptr) {
                GPU_CHECK_ERROR(api::gpuMemPrefetchAsync(ptr, sizeof(T) * base::size_, device_id, stream));
                if constexpr (has_prefetch<T>)
                {
                    if (recursive)
                        for (std::size_t i = 0; i < base::size_; ++i) ptr[i].prefetch(device_id, stream, recursive);
                }
            });
        }
        __host__ void prefetch(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(detail::get_device_id(), stream, recursive);
        }
        __host__ void prefetch_to_cpu(size_type n, size_type len, api::gpuStream_t stream = 0,
                                      bool recursive = true) const
        {
            prefetch(n, len, gpuCpuDeviceId, stream, recursive);
        }
        __host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(gpuCpuDeviceId, stream, recursive);
        }

        __host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, int device_id,
                                 bool recursive = true) const
        {
            assert(n + len <= base::size_);
            if (len == 0) return;
            base::tuple_for_each([n, len, device_id, advise, recursive]<typename T>(T* ptr) {
                GPU_CHECK_ERROR(api::gpuMemAdvise(ptr + n, sizeof(T) * len, advise, device_id));
                if constexpr (has_mem_advise<T>)
                {
                    if (recursive)
                        for (auto i = n; i < n + len; ++i) ptr[i].mem_advise(advise, device_id, recursive);
                }
            });
        }
        __host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(n, len, advise, detail::get_device_id(), recursive);
        }
        __host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const
        {
            if (base::size_ == 0) return;
            base::tuple_for_each([this, device_id, advise, recursive]<typename T>(T* ptr) {
                GPU_CHECK_ERROR(api::gpuMemAdvise(ptr, sizeof(T) * base::size_, advise, device_id));
                if constexpr (has_mem_advise<T>)
                {
                    if (recursive)
                        for (std::size_t i = 0; i < base::size_; ++i) ptr[i].mem_advise(advise, device_id, recursive);
                }
            });
        }
        __host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, detail::get_device_id(), recursive);
        }

        __host__ void mem_advise_to_cpu(std::size_t n, std::size_t len, api::gpuMemoryAdvise advise,
                                        bool recursive = true) const
        {
            mem_advise(n, len, advise, gpuCpuDeviceId, recursive);
        }
        __host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, gpuCpuDeviceId, recursive);
        }

        template <detail::array_convertible Range>
        requires std::is_default_constructible_v<Range> &&
                 std::is_constructible_v<std::ranges::range_value_t<Range>, Ts...>
        __host__ Range to() const
        {
            using U = std::ranges::range_value_t<Range>;
            auto result = Range();

            if constexpr (requires { std::declval<Range>().push_back(std::declval<U>()); })
            {
                // back insertable type
                if constexpr (requires { std::declval<Range>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }

                for (size_type i = 0; i < base::size_; ++i)
                {
                    std::apply([&result, i](const auto&... bufs) { result.push_back(U{bufs[i]...}); }, base::data_);
                }

                return result;
            }
            else
            {
                // std::array like type
                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");

                for (size_type i = 0; i < base::size_; ++i)
                {
                    std::apply([&result, i](const auto&... bufs) { result[i] = U{bufs[i]...}; }, base::data_);
                }

                return result;
            }
        }

        template <template <typename...> typename T>
        __host__ auto to() const
        {
            return to<T<Tuple<Ts...>>>();
        }

        template <typename T>
        requires (!gpu_ptr<T>) && requires(const managed_structure_of_arrays& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

        template <std::size_t N>
        __device__ void reset(std::tuple_element_t<N, tuple_pointer_type> ptr)
        {
            assert(base::size_ == 0 || ptr != nullptr);

            // reset specified pointer only
            std::get<N>(base::data_) = base::size_ == 0 ? nullptr : ptr;
        }

        template <std::size_t N, gpu_array_ptr T>
        __device__ void reset(const T& array)
        {
            assert(array.size() == base::size_);
            reset<N>(array.data());
        }

        __host__ void reset() { base::free(); }

        __host__ __device__ explicit operator bool() const noexcept
        {
            // If not empty, all pointers are valid
            return base::size_ != 0;
        }
    };

    // deduction guides
    template <template <typename...> typename Tuple, typename... Ts>
    structure_of_arrays(std::size_t, const Tuple<Ts...>&) -> structure_of_arrays<Tuple<Ts...>>;
    template <template <typename...> typename Tuple, typename... Ts>
    structure_of_arrays(std::initializer_list<Tuple<Ts...>>) -> structure_of_arrays<Tuple<Ts...>>;
    template <detail::array_convertible_for_copy... Range>
    structure_of_arrays(const Range&... array) -> structure_of_arrays<std::ranges::range_value_t<Range>...>;
    template <detail::array_convertible_for_copy Range>
    structure_of_arrays(const Range& array) -> structure_of_arrays<std::ranges::range_value_t<Range>>;
    template <template <typename...> typename Tuple, typename... Ts>
    managed_structure_of_arrays(std::size_t, const Tuple<Ts...>&) -> managed_structure_of_arrays<Tuple<Ts...>>;
    template <template <typename...> typename Tuple, typename... Ts>
    managed_structure_of_arrays(std::initializer_list<Tuple<Ts...>>) -> managed_structure_of_arrays<Tuple<Ts...>>;
    template <detail::array_convertible_for_copy... Range>
    managed_structure_of_arrays(const Range&... array)
        -> managed_structure_of_arrays<std::ranges::range_value_t<Range>...>;
    template <detail::array_convertible_for_copy Range>
    managed_structure_of_arrays(const Range& array) -> managed_structure_of_arrays<std::ranges::range_value_t<Range>>;

    template <gpu_managed_ptr ArrayType>
    class jagged_array : public ArrayType
    {
        using base = ArrayType;
        using size_type = typename base::size_type;
        using offsets_type = managed_array<size_type>;
        using iterator_type = std::ranges::iterator_t<ArrayType>;
        using const_iterator_type = decltype(std::ranges::cbegin(std::declval<ArrayType&>()));

        offsets_type offsets_;
        static constexpr auto has_data = requires(const ArrayType& a) { a.data(); };
        static constexpr auto has_prefetch = requires(const ArrayType& a) { a.prefetch(); };

    public:
        jagged_array() = default;
        jagged_array(const jagged_array&) = default;
        jagged_array(jagged_array&&) noexcept = default;

        template <detail::array_convertible_for_copy Range>
        requires std::constructible_from<size_type, std::ranges::range_value_t<Range>>
        __host__ explicit jagged_array(const Range& sizes)
            : base(std::accumulate(std::ranges::begin(sizes), std::ranges::end(sizes), std::size_t{0})),
              offsets_(std::ranges::size(sizes) + 1U, default_init)
        {
            offsets_[0] = 0;
            for (size_type i = 0; const auto& s : sizes)
            {
                offsets_[i + 1] = offsets_[i] + static_cast<size_type>(s);
                ++i;
            }
        }

        template <detail::array_convertible_for_copy Range>
        requires detail::array_convertible_for_copy<std::ranges::range_value_t<Range>>
        __host__ explicit jagged_array(const Range& nested_array)
            : base(nested_array, detail::join_init), offsets_(std::ranges::size(nested_array) + 1U, default_init)
        {
            offsets_[0] = 0;
            for (size_type i = 0; const auto& a : nested_array)
            {
                offsets_[i + 1] = static_cast<size_type>(offsets_[i] + std::ranges::size(a));
                ++i;
            }
        }

        __host__ jagged_array(
            std::initializer_list<std::initializer_list<std::ranges::range_value_t<ArrayType>>> nested_list)
            : base(nested_list, detail::join_init), offsets_(std::ranges::size(nested_list) + 1U, default_init)
        {
            offsets_[0] = 0;
            for (size_type i = 0; const auto& a : nested_list)
            {
                offsets_[i + 1] = static_cast<size_type>(offsets_[i] + std::ranges::size(a));
                ++i;
            }
        }

        template <detail::array_convertible_for_copy Range1, detail::array_convertible_for_copy Range2>
        requires std::constructible_from<size_type, std::ranges::range_value_t<Range1>>
        __host__ jagged_array(const Range1& sizes, const Range2& array)
            : base(array), offsets_(std::ranges::size(sizes) + 1U, default_init)
        {
            if (std::accumulate(std::ranges::begin(sizes), std::ranges::end(sizes), std::size_t{0}) != base::size())
            {
                throw std::invalid_argument("the total size of sizes does not match the size of array");
            }

            offsets_[0] = 0;
            for (size_type i = 0; const auto& s : sizes)
            {
                offsets_[i + 1] = offsets_[i] + static_cast<size_type>(s);
                ++i;
            }
        }

        template <detail::array_convertible_for_copy Range>
        __host__ jagged_array(std::initializer_list<size_type> sizes, const Range& array)
            : base(array), offsets_(std::ranges::size(sizes) + 1U, default_init)
        {
            if (std::accumulate(std::ranges::begin(sizes), std::ranges::end(sizes), std::size_t{0}) != base::size())
            {
                throw std::invalid_argument("the total size of sizes does not match the size of array");
            }

            offsets_[0] = 0;
            for (size_type i = 0; const auto& s : sizes)
            {
                offsets_[i + 1] = offsets_[i] + static_cast<size_type>(s);
                ++i;
            }
        }

        jagged_array& operator=(const jagged_array&) = default;
        jagged_array& operator=(jagged_array&&) noexcept = default;

        using base::begin;
        using base::empty;
        using base::end;
        using base::size;
        using base::operator[];

        __host__ __device__ auto begin(size_type i) noexcept { return base::begin() + offsets_[i]; }
        __host__ __device__ auto end(size_type i) noexcept { return base::begin() + offsets_[i + 1]; }
        __host__ __device__ auto begin(size_type i) const noexcept { return base::begin() + offsets_[i]; }
        __host__ __device__ auto end(size_type i) const noexcept { return base::begin() + offsets_[i + 1]; }

        __host__ __device__ auto size(size_type i) const noexcept
        {
            assert(i < num_rows());
            return offsets_[i + 1] - offsets_[i];
        }
        __host__ __device__ auto data() noexcept
        requires has_data
        {
            return base::data();
        }
        __host__ __device__ auto data() const noexcept
        requires has_data
        {
            return base::data();
        }
        __host__ __device__ auto data(size_type i) noexcept
        requires has_data
        {
            assert(i < num_rows());
            return base::data() + offsets_[i];
        }
        __host__ __device__ auto data(size_type i) const noexcept
        requires has_data
        {
            assert(i < num_rows());
            return base::data() + offsets_[i];
        }

        __host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) &
        {
            assert(idx[0] < num_rows());
            assert(idx[1] < size(idx[0]));
            return (*this)[offsets_[idx[0]] + idx[1]];
        }
        __host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) const&
        {
            assert(idx[0] < num_rows());
            assert(idx[1] < size(idx[0]));
            return (*this)[offsets_[idx[0]] + idx[1]];
        }
        __host__ __device__ decltype(auto) operator[](std::array<size_type, 2> idx) &&
        {
            assert(idx[0] < num_rows());
            assert(idx[1] < size(idx[0]));
            return (*this)[offsets_[idx[0]] + idx[1]];
        }
#ifdef __cpp_multidimensional_subscript
        __host__ __device__ decltype(auto) operator[](size_type i, size_type j) & { return this->operator[]({i, j}); }
        __host__ __device__ decltype(auto) operator[](size_type i, size_type j) const&
        {
            return this->operator[]({i, j});
        }
        __host__ __device__ decltype(auto) operator[](size_type i, size_type j) && { return this->operator[]({i, j}); }
#endif

        __host__ __device__ auto row(size_type i) noexcept
        {
            assert(i < num_rows());
            return std::ranges::subrange<iterator_type, iterator_type, std::ranges::subrange_kind::sized>(
                base::begin() + offsets_[i], base::begin() + offsets_[i + 1], offsets_[i + 1] - offsets_[i]);
        }
        __host__ __device__ auto row(size_type i) const noexcept
        {
            assert(i < num_rows());
            return std::ranges::subrange<const_iterator_type, const_iterator_type, std::ranges::subrange_kind::sized>(
                base::begin() + offsets_[i], base::begin() + offsets_[i + 1], offsets_[i + 1] - offsets_[i]);
        }

        __host__ __device__ auto num_rows() const noexcept { return offsets_.size() - 1; }

        __host__ void prefetch(size_type n, size_type len, int device_id, api::gpuStream_t stream = 0,
                               bool recursive = true) const
        {
            if constexpr (has_prefetch) base::prefetch(n, len, device_id, stream, recursive);
            offsets_.prefetch(device_id, stream);
        }
        __host__ void prefetch(size_type n, size_type len, api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(n, len, detail::get_device_id(), stream, recursive);
        }
        __host__ void prefetch(int device_id, api::gpuStream_t stream = 0, bool recursive = true) const
        {
            if constexpr (has_prefetch) base::prefetch(device_id, stream, recursive);
            offsets_.prefetch(device_id, stream);
        }
        __host__ void prefetch(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(detail::get_device_id(), stream, recursive);
        }

        __host__ void prefetch_to_cpu(size_type n, size_type len, api::gpuStream_t stream = 0,
                                      bool recursive = true) const
        {
            prefetch(n, len, gpuCpuDeviceId, stream, recursive);
        }
        __host__ void prefetch_to_cpu(api::gpuStream_t stream = 0, bool recursive = true) const
        {
            prefetch(gpuCpuDeviceId, stream, recursive);
        }

        __host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, int device_id,
                                 bool recursive = true) const
        {
            if constexpr (has_prefetch) base::mem_advise(n, len, advise, device_id, recursive);
            offsets_.mem_advise(advise, device_id);
        }
        __host__ void mem_advise(size_type n, size_type len, api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(n, len, advise, detail::get_device_id(), recursive);
        }
        __host__ void mem_advise(api::gpuMemoryAdvise advise, int device_id, bool recursive = true) const
        {
            if constexpr (has_prefetch) base::mem_advise(advise, device_id, recursive);
            offsets_.mem_advise(advise, device_id);
        }
        __host__ void mem_advise(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, detail::get_device_id(), recursive);
        }

        __host__ void mem_advise_to_cpu(size_type n, size_type len, api::gpuMemoryAdvise advise,
                                        bool recursive = true) const
        {
            mem_advise(n, len, advise, gpuCpuDeviceId, recursive);
        }
        __host__ void mem_advise_to_cpu(api::gpuMemoryAdvise advise, bool recursive = true) const
        {
            mem_advise(advise, gpuCpuDeviceId, recursive);
        }

#if defined(GPU_PTR_DEBUG)
        [[deprecated("for debug")]] [[nodiscard]] const auto& get_offsets() const noexcept { return offsets_; }
        [[deprecated("for debug")]] [[nodiscard]] auto get_sizes() const noexcept
        {
            auto result = std::vector<size_type>();
            result.reserve(num_rows());
            for (size_type i = 0; i < num_rows(); ++i) result.emplace_back(offsets_[i + 1] - offsets_[i]);
            return result;
        }
#endif
    };

    // deduction guides
    template <detail::array_convertible_for_copy Range>
    requires gpu_managed_ptr<std::ranges::range_value_t<Range>>
    jagged_array(const Range& nested_array) -> jagged_array<std::ranges::range_value_t<Range>>;
    template <detail::array_convertible_for_copy Range1, detail::array_convertible_for_copy Range2>
    requires gpu_managed_ptr<Range2>
    jagged_array(const Range1&, const Range2&) -> jagged_array<Range2>;
    template <detail::array_convertible_for_copy Range>
    requires gpu_managed_ptr<Range>
    jagged_array(std::initializer_list<size_type_default>, const Range&) -> jagged_array<Range>;

}  // namespace gpu_ptr

// speciialization for std::ranges
template <typename... Ts>
inline constexpr bool std::ranges::enable_view<gpu_ptr::array<Ts...>> = true;
template <typename... Ts>
inline constexpr bool std::ranges::enable_view<gpu_ptr::managed_array<Ts...>> = true;
template <typename... Ts>
inline constexpr bool std::ranges::enable_view<gpu_ptr::structure_of_arrays<Ts...>> = true;
template <typename... Ts>
inline constexpr bool std::ranges::enable_view<gpu_ptr::managed_structure_of_arrays<Ts...>> = true;
#if defined(GPU_DEVICE_COMPILE)
template <typename... Ts>
inline constexpr bool std::ranges::enable_borrowed_range<gpu_ptr::array<Ts...>> = true;
template <typename... Ts>
inline constexpr bool std::ranges::enable_borrowed_range<gpu_ptr::managed_array<Ts...>> = true;
template <typename... Ts>
inline constexpr bool std::ranges::enable_borrowed_range<gpu_ptr::structure_of_arrays<Ts...>> = true;
template <typename... Ts>
inline constexpr bool std::ranges::enable_borrowed_range<gpu_ptr::managed_structure_of_arrays<Ts...>> = true;
#endif

#undef SIGSEGV_DEPRECATED
#undef INCR_GPU_MEORY_USAGE
#undef DECR_GPU_MEORY_USAGE
#undef MEMORY_USAGE_EQ
