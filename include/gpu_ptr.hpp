/*===================================================*
|  GPU smart pointer (gpu-smart-ptr) version v0.0.1  |
|  https://github.com/yosh-matsuda/gpu-ptr           |
|                                                    |
|  Copyright (c) 2024 Yoshiki Matsuda @yosh-matsuda  |
|                                                    |
|  This software is released under the MIT License.  |
|  https://opensource.org/license/mit/               |
====================================================*/

#pragma once
#include "gpu_runtime_api.hpp"

#if defined(__HIP_DEVICE_COMPILE__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)
#define GPU_DEVICE_COMPILE
#if !defined(NDEBUG)
#define NDEBUG_DEVICE_CODE_STL
#define NDEBUG
#endif
#endif

#include <algorithm>
#include <cassert>
#include <concepts>
#include <iostream>
#include <memory>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#if defined(NDEBUG_DEVICE_CODE_STL)
#undef NDEBUG_DEVICE_CODE_STL
#undef NDEBUG
#endif

#if !defined(__NVCC__) || defined(__CUDA_ARCH__) || defined(_CLANGD)
#define OVERLOAD_DEVICE
#endif
#if !(defined(__NVCC__) && defined(__CUDA_ARCH__)) || defined(_CLANGD)
#define OVERLOAD_HOST
#endif

#if defined(GPU_DEVICE_COMPILE) || !defined(__NVCC__)
#define SIGSEGV_DEPRECATED
#else
#define SIGSEGV_DEPRECATED [[deprecated("Cannot access GPU memory directly")]]
#endif

#define CHECK_GPU_ERROR(expr) (gpu_smart_ptr::detail::check_gpu_error(expr, __FILE__, __LINE__))

namespace gpu_smart_ptr
{
    template <typename ValueType>
    requires std::is_trivially_copyable_v<ValueType>
    class array_ptr;
    template <typename ValueType>
    class unified_array_ptr;

    inline constexpr struct default_init_tag
    {
        struct tag
        {
        };
        constexpr explicit default_init_tag(tag) {}
    } default_init{default_init_tag::tag{}};

    namespace detail
    {

#ifdef NDEBUG
#define INCR_GPU_MEORY_USAGE(x) void(x)
#define DECR_GPU_MEORY_USAGE(x) void(x)
#define GPU_MEMORY_USAGE_EQ(x) (true)
#else
        inline std::size_t gpu_memory_usage = 0UL;
#define INCR_GPU_MEORY_USAGE(x) (gpu_smart_ptr::detail::gpu_memory_usage += (x))
#define DECR_GPU_MEORY_USAGE(x) (gpu_smart_ptr::detail::gpu_memory_usage -= (x))
#define GPU_MEMORY_USAGE_EQ(x) (gpu_smart_ptr::detail::gpu_memory_usage == (x))
#endif

        __host__ inline void check_gpu_error(const gpuError_t e, const char* f, decltype(__LINE__) n)
        {
            if (e != gpuSuccess)
            {
                std::stringstream s;
                s << gpuGetErrorName(e) << " (" << static_cast<unsigned>(e) << ")@" << f << "#L" << n << ": "
                  << gpuGetErrorString(e);
                throw std::runtime_error{s.str()};
            }
        }

        __host__ inline int get_device_count()
        {
            int device_count;
            CHECK_GPU_ERROR(gpuGetDeviceCount(&device_count));
            return device_count;
        }

        __host__ inline int get_device_id()
        {
            int device_id;
            CHECK_GPU_ERROR(gpuGetDevice(&device_id));
            assert(device_id >= 0 && device_id < get_device_count());
            return device_id;
        }

        template <bool Unified, typename... ValueTypes>
        requires (sizeof...(ValueTypes) > 0)
        class base
        {
        public:
#pragma region std::array like implementations
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;

            __host__ __device__ size_type size() const noexcept { return size_; }
            __host__ __device__ size_type max_size() const noexcept { return size_; }
            __host__ __device__ bool empty() const noexcept { return size_ == 0; }
#pragma endregion std::array like implementations

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
                    // do not throw in destructor
                    try
                    {
                        // std::cout << "gpuFree: " << NAMEOF_FULL_TYPE(ValueType) << ' ' << size_ << '\n';
                        if constexpr (Unified)
                        {
                            // call destructor explicitly for unified memory
                            base::tuple_for_each([this](auto* ptr) { std::destroy_n(ptr, size_); });
                        }

                        base::tuple_for_each([](auto* ptr) { CHECK_GPU_ERROR(gpuFree(ptr)); });
                        (DECR_GPU_MEORY_USAGE(sizeof(ValueTypes) * size_), ...);  // for debug
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
#ifndef _CLANGD  // clangd crashes with this code
                    delete ref_count_;
#endif
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
                if (ref_count_ != nullptr) ++*ref_count_;
#endif
                return *this;
            }
            __host__ __device__ base& operator=(base&& r) noexcept
            {
                this->operator=(r);
                r.init();
                return *this;
            }

            __host__ __device__ base() : data_(static_cast<ValueTypes*>(nullptr)...) {};
            __host__ __device__ base(const base& r) : size_(r.size_), data_(r.data_), ref_count_(r.ref_count_)
            {
#ifndef GPU_DEVICE_COMPILE
                if (ref_count_ != nullptr) ++*ref_count_;
#endif
            }
            __host__ __device__ base(base&& r) noexcept : size_(r.size_), data_(r.data_), ref_count_(r.ref_count_)
            {
                r.init();
            }
            __device__ base(ValueTypes*... ptr, size_type size)
                : size_(size), data_(static_cast<ValueTypes*>(size != 0 ? ptr : nullptr)...)
            {
                assert(size_ == 0 || ((ptr != nullptr) && ...));
            }
            __host__ explicit base(size_type size)
                : size_(size), ref_count_(size_ == 0 ? nullptr : new std::uint32_t(1))
            {
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

        public:
            template <std::size_t N>
            requires std::is_trivially_destructible_v<element_type<N>>
            __device__ element_type<N>* move_to(element_type<N>* dst,
                                                bool is_local = false)  // TODO: use __isLocal(this)
            {
                using ValueType = element_type<N>;

                // must be aligned
                assert(reinterpret_cast<std::uintptr_t>(dst) % alignof(ValueType) == 0);

                ValueType*& src = std::get<N>(data_);
                const auto block = cooperative_groups::this_thread_block();
                assert(size_ + block.size() <= std::numeric_limits<std::uint32_t>::max());
                const auto size = static_cast<std::uint32_t>(size_);

                for (auto i = block.thread_rank(); i < size; i += block.size())
                {
                    new (dst + i) ValueType(src[i]);
                }

                // NOTE: HIP does not provide __isLocal
                if (is_local)
                {
                    // reset on each thread
                    src = dst;
                }
                else
                {
                    // need sync and reset only on the first thread
                    block.sync();
                    if (block.thread_rank() == 0) src = dst;
                }

                return dst + size;
            }

            template <std::size_t N>
            __host__ std::pair<std::uintptr_t, std::uintptr_t> alignment(std::uintptr_t addr) const
            {
                using ValueType = std::tuple_element_t<N, std::tuple<ValueTypes...>>;
                const auto padding = addr % alignof(ValueType);
                if (padding != 0) addr += alignof(ValueType) - padding;
                return {addr, addr + sizeof(ValueType) * size_};
            }
        };

        template <typename T>
        concept array_convertible = std::ranges::forward_range<T> && std::ranges::sized_range<T>;

        template <bool Unified, typename... ValueType>
        constexpr void is_ptr_helper(const detail::base<Unified, ValueType...>&)
        {
        }
        template <typename... ValueType>
        constexpr void is_unified_ptr_helper(const detail::base<true, ValueType...>&)
        {
        }
        template <bool Unified, typename ValueType>
        constexpr void is_array_helper(const detail::base<Unified, ValueType>&)
        {
        }
        template <typename ValueType>
        constexpr void is_unified_array_helper(const detail::base<true, ValueType>&)
        {
        }

        template <class Derived>
        concept gpu_smart_ptr = requires(Derived d) { is_ptr_helper(d); };
        template <class Derived>
        concept gpu_unified_smart_ptr = requires(Derived d) { is_unified_ptr_helper(d); };
        template <class Derived>
        concept gpu_array_ptr = requires(Derived d) { is_array_helper(d); };
        template <class Derived>
        concept gpu_unified_array_ptr = requires(Derived d) { is_unified_array_helper(d); };

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
            using type = unified_array_ptr<unified_array_deduced_t<std::ranges::range_value_t<T>>>;
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
    using detail::gpu_smart_ptr;
    using detail::gpu_unified_array_ptr;
    using detail::gpu_unified_smart_ptr;

    template <typename ValueType>
    requires std::is_trivially_copyable_v<ValueType>
    class array_ptr : public detail::base<false, ValueType>
    {
        // TODO: range の要件を満たしたい

        using base = detail::base<false, ValueType>;

    public:
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator = value_type*;
        using const_iterator = const value_type*;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        SIGSEGV_DEPRECATED __host__ __device__ const_reference operator[](base::size_type i) const noexcept
        {
            return data()[i];
        };
        SIGSEGV_DEPRECATED __host__ __device__ reference operator[](base::size_type i) noexcept { return data()[i]; }
        SIGSEGV_DEPRECATED __host__ __device__ iterator begin() noexcept { return data(); }
        SIGSEGV_DEPRECATED __host__ __device__ iterator end() noexcept { return data() + base::size_; }
        SIGSEGV_DEPRECATED __host__ __device__ iterator rbegin() noexcept
        {
            return std::reverse_iterator<iterator>(end());
        }
        SIGSEGV_DEPRECATED __host__ __device__ iterator rend() noexcept
        {
            return std::reverse_iterator<iterator>(begin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator begin() const noexcept { return data(); }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator end() const noexcept { return data() + base::size_; }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator rbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(end());
        }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator rend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(begin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator cbegin() const noexcept { return data(); }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator cend() const noexcept
        {
            return base::data_ + base::size_;
        }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator crbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cend());
        }
        SIGSEGV_DEPRECATED __host__ __device__ const_iterator crend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cbegin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ reference front() noexcept { return *begin(); }
        SIGSEGV_DEPRECATED __host__ __device__ const_reference front() const noexcept { return *begin(); }
        SIGSEGV_DEPRECATED __host__ __device__ reference back() noexcept { return *(--end()); }
        SIGSEGV_DEPRECATED __host__ __device__ const_reference back() const noexcept { return *(--end()); }
        __host__ __device__ pointer data() noexcept { return std::get<0>(base::data_); }
        __host__ __device__ const_pointer data() const noexcept { return std::get<0>(base::data_); }

        __host__ __device__ array_ptr() = default;
        __host__ __device__ array_ptr(const array_ptr& r) : base(r) {}
        __host__ __device__ array_ptr(array_ptr&& r) noexcept : base(std::move(r)) {}

        __host__ explicit array_ptr(base::size_type size) : base(size)
        {
            if (base::size_ == 0) return;
            auto buf = std::make_unique<value_type[]>(base::size_);
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                              sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            CHECK_GPU_ERROR(
                detail::gpuMemcpy(data(), buf.get(), sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
        }

        __host__ array_ptr(base::size_type size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                              sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            if constexpr (!std::is_trivially_default_constructible_v<value_type>)
            {
                auto buf = std::make_unique<value_type[]>(base::size_);
                CHECK_GPU_ERROR(
                    detail::gpuMemcpy(data(), buf.get(), sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
            }
        }

        __host__ array_ptr(base::size_type size, const value_type& value) : base(size)
        {
            if (base::size_ == 0) return;

            auto al = std::allocator<value_type>();
            auto buf = al.allocate(base::size_);
            std::ranges::uninitialized_fill(buf, buf + base::size_, value);
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                              sizeof(value_type) * base::size_));
            assert(data() != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(data(), buf, sizeof(value_type) * base::size_, gpuMemcpyHostToDevice));
            al.deallocate(buf, base::size_);
        }

        template <detail::array_convertible T, std::same_as<ValueType> U = std::ranges::range_value_t<T>>
        requires std::ranges::contiguous_range<T>
        __host__ explicit array_ptr(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                              sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(data(), std::ranges::data(r), sizeof(ValueType) * base::size_,
                                              gpuMemcpyHostToDevice));
        }

        template <detail::array_convertible T, typename U = std::ranges::range_value_t<T>>
        requires std::is_constructible_v<ValueType, U> &&
                 (!std::same_as<ValueType, U> || !std::ranges::contiguous_range<T>)
        __host__ explicit array_ptr(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;

            auto al = std::allocator<ValueType>();
            auto buf = al.allocate(base::size_);
            for (auto i = std::size_t{0}; const auto& v : r) std::ranges::construct_at(buf + i++, v);

            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                              sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(data(), buf, sizeof(ValueType) * base::size_, gpuMemcpyHostToDevice));
            al.deallocate(buf, base::size_);
        }

        __host__ array_ptr(std::initializer_list<ValueType> r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                              sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(data(), std::ranges::data(r), sizeof(ValueType) * base::size_,
                                              gpuMemcpyHostToDevice));
        }

        __device__ array_ptr(pointer ptr, base::size_type size) : base(ptr, size) {}

        __host__ __device__ array_ptr& operator=(const array_ptr& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ array_ptr& operator=(array_ptr&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }
        __host__ array_ptr& operator=(std::initializer_list<ValueType> r) noexcept
        {
            base::operator=(array_ptr(r));
            return *this;
        }

        template <std::ranges::range T, std::same_as<ValueType> U = std::ranges::range_value_t<T>>
        requires (std::ranges::contiguous_range<T> &&
                  (std::is_constructible_v<T, typename base::size_type> || std::is_default_constructible_v<T>)) ||
                 (std::is_default_constructible_v<T> &&
                  requires { std::declval<T>().push_back(std::declval<ValueType>()); })
        __host__ T to() const
        {
            if constexpr (gpu_array_ptr<T>)
            {
                // array<ValueType> or unified_array<ValueType> like type
                return T(*this);
            }
            else if constexpr (std::ranges::contiguous_range<T>)
            {
                auto result = []([[maybe_unused]] base::size_type size) -> T {
                    if constexpr (std::is_constructible_v<T, typename base::size_type>)
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
                CHECK_GPU_ERROR(detail::gpuMemcpy(std::ranges::data(result), data(), sizeof(ValueType) * base::size_,
                                                  gpuMemcpyDeviceToHost));
                return result;
            }
            else
            {
                // std::list like type
                static_assert(requires { std::declval<T>().push_back(std::declval<ValueType>()); });
                auto buf = std::make_unique_for_overwrite<ValueType[]>(base::size_);
                CHECK_GPU_ERROR(
                    detail::gpuMemcpy(buf.get(), data(), sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));

                auto result = T();
                if constexpr (requires { std::declval<T>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }
                std::move(buf.get(), buf.get() + base::size_, std::back_inserter(result));
                return result;
            }
        }

        template <std::ranges::range T, typename U = std::ranges::range_value_t<T>>
        requires std::is_default_constructible_v<T> && requires(const ValueType& v) { static_cast<U>(v); } &&
                 (!std::same_as<ValueType, U>)
        __host__ T to() const
        {
            auto buf = std::make_unique_for_overwrite<ValueType[]>(base::size_);
            CHECK_GPU_ERROR(
                detail::gpuMemcpy(buf.get(), data(), sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));

            if constexpr (gpu_array_ptr<T>)
            {
                // array<U> or unified_array<U> like type
                return T(std::ranges::subrange(buf.get(), buf.get() + base::size_));
            }
            else if constexpr (requires { std::declval<T>().push_back(std::declval<U>()); })
            {
                // back insertable type
                auto result = T();
                if constexpr (requires { std::declval<T>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }
                std::transform(buf.get(), buf.get() + base::size_, std::back_inserter(result),
                               [](const ValueType& v) { return static_cast<U>(v); });
                return result;
            }
            else
            {
                // std::array like type
                auto result = T();
                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");
                std::transform(buf.get(), buf.get() + base::size_, std::ranges::begin(result),
                               [](const ValueType& v) { return static_cast<U>(v); });
                return result;
            }
        }

        template <template <typename...> typename T>
        __host__ auto to() const
        {
            return to<T<ValueType>>();
        }

        template <typename T>
        requires requires(const array_ptr& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

        __device__ void reset(pointer ptr, std::size_t size)
        {
            assert(size == 0 || ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            std::get<0>(base::data_) = size == 0 ? nullptr : ptr;
            base::size_ = size;
        }

        template <gpu_array_ptr T>
        __device__ void reset(const T& array)
        {
            reset(array.data(), array.size());
        }

        __host__ void reset() { base::free(); }

        __device__ pointer move_to(pointer gpu_ptr, bool is_local = false)
        {
            return base::template move_to<0>(gpu_ptr, is_local);
        }

        __host__ std::pair<std::uintptr_t, std::uintptr_t> alignment(std::uintptr_t addr) const
        {
            return base::template alignment<0>(addr);
        }
    };

    template <typename ValueType>
    class unified_array_ptr : public detail::base<true, ValueType>
    {
        using base = detail::base<true, ValueType>;

    protected:
        template <detail::array_convertible Range, typename U = std::ranges::range_value_t<Range>>
        requires std::is_constructible_v<ValueType, std::ranges::range_value_t<U>>
        __host__ explicit unified_array_ptr(const Range& nested_array, detail::join_init_tag)
            : base(std::accumulate(std::ranges::begin(nested_array), std::ranges::end(nested_array), std::size_t{0},
                                   [](auto acc, const auto& r) { return acc + std::ranges::size(r); }))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);

            for (auto i = std::size_t{0}; const auto& array : nested_array)
            {
                for (const auto& v : array) std::ranges::construct_at(data() + i++, v);
            }
        }

    public:
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator = value_type*;
        using const_iterator = const value_type*;
        using pointer = value_type*;
        using const_pointer = const value_type*;

        __host__ __device__ const_reference operator[](std::size_t i) const noexcept { return data()[i]; };
        __host__ __device__ reference operator[](std::size_t i) noexcept { return data()[i]; }
        __host__ __device__ iterator begin() noexcept { return data(); }
        __host__ __device__ iterator end() noexcept { return data() + base::size_; }
        __host__ __device__ iterator rbegin() noexcept { return std::reverse_iterator<iterator>(end()); }
        __host__ __device__ iterator rend() noexcept { return std::reverse_iterator<iterator>(begin()); }
        __host__ __device__ const_iterator begin() const noexcept { return data(); }
        __host__ __device__ const_iterator end() const noexcept { return data() + base::size_; }
        __host__ __device__ const_iterator rbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(end());
        }
        __host__ __device__ const_iterator rend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(begin());
        }
        __host__ __device__ const_iterator cbegin() const noexcept { return data(); }
        __host__ __device__ const_iterator cend() const noexcept { return data() + base::size_; }
        __host__ __device__ const_iterator crbegin() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cend());
        }
        __host__ __device__ const_iterator crend() const noexcept
        {
            return std::reverse_iterator<const_iterator>(cbegin());
        }
        __host__ __device__ reference front() noexcept { return *begin(); }
        __host__ __device__ const_reference front() const noexcept { return *begin(); }
        __host__ __device__ reference back() noexcept { return *(--end()); }
        __host__ __device__ const_reference back() const noexcept { return *(--end()); }
        __host__ __device__ pointer data() noexcept { return std::get<0>(base::data_); }
        __host__ __device__ const_pointer data() const noexcept { return std::get<0>(base::data_); }

        __host__ __device__ unified_array_ptr() = default;
        __host__ __device__ unified_array_ptr(const unified_array_ptr& r) : base(r) {}
        __host__ __device__ unified_array_ptr(unified_array_ptr&& r) noexcept : base(std::move(r)) {}

        __host__ explicit unified_array_ptr(base::size_type size) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            std::ranges::uninitialized_value_construct(*this);
        }

        __host__ explicit unified_array_ptr(base::size_type size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            std::ranges::uninitialized_default_construct(*this);
        }

        __host__ unified_array_ptr(base::size_type size, const ValueType& value) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            std::ranges::uninitialized_fill(*this, value);
        }

        template <detail::array_convertible T, typename U = std::ranges::range_value_t<T>>
        requires std::is_constructible_v<ValueType, U>
        __host__ explicit unified_array_ptr(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);

            for (auto i = std::size_t{0}; const auto& v : r)
            {
                std::ranges::construct_at(data() + i++, v);
            }
        }

        __host__ unified_array_ptr(std::initializer_list<ValueType> r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            for (auto i = std::size_t{0}; const auto& v : r) std::ranges::construct_at(data() + i++, v);
        }

        template <typename U>
        requires std::is_trivially_copyable_v<U> && std::same_as<U, ValueType>
        __host__ explicit unified_array_ptr(const array_ptr<U>& r) : base(r.size())
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            CHECK_GPU_ERROR(
                detail::gpuMemcpy(data(), r.data(), sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));
        }

        template <typename U>
        requires std::is_trivially_copyable_v<U> &&
                 (!std::same_as<U, ValueType>) && std::is_constructible_v<ValueType, U>
        __host__ explicit unified_array_ptr(const array_ptr<U>& r) : base(r.size())
        {
            if (base::size_ == 0) return;
            auto buf = std::make_unique_for_overwrite<U[]>(base::size_);
            CHECK_GPU_ERROR(detail::gpuMemcpy(buf.get(), r.data(), sizeof(U) * base::size_, gpuMemcpyDeviceToHost));
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)),
                                                     sizeof(ValueType) * base::size_));
            assert(data() != nullptr);
            for (auto i = std::size_t{0}; i < base::size_; ++i)
            {
                std::ranges::construct_at(data() + i, *(buf.get() + i));
            }
        }

        __host__ __device__ unified_array_ptr& operator=(const unified_array_ptr& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ unified_array_ptr& operator=(unified_array_ptr&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }
        __host__ unified_array_ptr& operator=(std::initializer_list<ValueType> r) noexcept
        {
            base::operator=(unified_array_ptr(r));
            return *this;
        }

        __host__ void prefetch(int device_id) const
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMemPrefetchAsync(data(), sizeof(ValueType) * base::size_, device_id));
        }
        __host__ void prefetch_to_cpu() const { prefetch(gpuCpuDeviceId); }
        __host__ void prefetch_to_gpu() const { prefetch(detail::get_device_id()); }

        template <std::ranges::range T, typename U = std::ranges::range_value_t<T>>
        requires std::is_default_constructible_v<T> && requires(const ValueType& v) { static_cast<U>(v); }
        __host__ T to() const
        {
            prefetch_to_cpu();
            if constexpr (gpu_array_ptr<T>)
            {
                // array_ptr<U> or unified_array_ptr<U>
                return T(*this);
            }
            else
            {
                auto result = T();
                if constexpr (requires { std::declval<T>().push_back(std::declval<U>()); })
                {
                    // back insertable type
                    if constexpr (requires { std::declval<T>().reserve(std::declval<std::size_t>()); })
                    {
                        result.reserve(base::size_);
                    }
                    std::ranges::transform(*this, std::back_inserter(result),
                                           [](const ValueType& v) { return static_cast<U>(v); });
                }
                else
                {
                    // std::array like type
                    if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");
                    std::ranges::transform(*this, std::ranges::begin(result),
                                           [](const ValueType& v) { return static_cast<U>(v); });
                }
                return result;
            }
        }

        template <template <typename...> typename U>
        __host__ auto to() const
        {
            return to<detail::to_range_deduced_t<unified_array_ptr, U>>();
        }

        template <typename T>
        requires requires(const unified_array_ptr& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }
        __device__ void reset(pointer ptr, std::size_t size)
        requires std::is_trivially_copyable_v<ValueType>
        {
            assert(size == 0 || ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            std::get<0>(base::data_) = size == 0 ? nullptr : ptr;
            base::size_ = size;
        }

        template <gpu_array_ptr T>
        __device__ void reset(const T& array)
        {
            reset(array.data(), array.size());
        }

        __host__ void reset() { base::free(); }

        __device__ pointer move_to(pointer gpu_ptr, bool is_local = false)
        {
            return base::template move_to<0>(gpu_ptr, is_local);
        }

        __host__ std::pair<std::uintptr_t, std::uintptr_t> alignment(std::uintptr_t addr) const
        {
            return base::template alignment<0>(addr);
        }
    };

    // deduction guide for arrays
    template <detail::array_convertible T>
    __host__ array_ptr(const T& r) -> array_ptr<std::ranges::range_value_t<T>>;
    template <detail::array_convertible T>
    __host__ unified_array_ptr(const T& r)
        -> unified_array_ptr<detail::unified_array_deduced_t<std::ranges::range_value_t<T>>>;

    template <typename ValueType>
    requires std::is_trivially_copyable_v<ValueType>
    class value : protected detail::base<false, ValueType>
    {
        using base = detail::base<false, ValueType>;
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

    public:
        using element_type = value_type;

        __host__ __device__ value(const value& r) : base(r) {};
        __host__ __device__ value(value&& r) noexcept : base(std::move(r)) {};

        __host__ value() : base(1)
        {
            auto temp = value_type();
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type)));
            assert(get() != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(get(), &temp, sizeof(value_type), gpuMemcpyHostToDevice));
        }

        __host__ explicit value(default_init_tag) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type)));
            assert(get() != nullptr);
            if constexpr (!std::is_trivially_default_constructible_v<value_type>)
            {
                auto temp = value_type();
                CHECK_GPU_ERROR(detail::gpuMemcpy(get(), &temp, sizeof(value_type), gpuMemcpyHostToDevice));
            }
        }

        __host__ explicit value(const value_type& r) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type)));
            assert(get() != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(get(), &r, sizeof(value_type), gpuMemcpyHostToDevice));
        }

        template <typename... Args>
        requires std::is_constructible_v<value_type, Args...>
        __host__ explicit value(Args&&... args) : base(1)
        {
            auto temp = value_type(std::forward<Args>(args)...);
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(value_type)));
            assert(get() != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(get(), &temp, sizeof(value_type), gpuMemcpyHostToDevice));
        }

        __device__ explicit value(pointer ptr) : base(ptr, 1) {}

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

#ifdef OVERLOAD_DEVICE
        __device__ const_reference operator*() const noexcept { return *get(); }
        __device__ reference operator*() noexcept { return *get(); }
        __device__ const_pointer operator->() const noexcept { return get(); }
        __device__ pointer operator->() noexcept { return get(); }
#endif
#ifdef OVERLOAD_HOST
        __host__ element_type operator*() const
        {
            value_type temp;
            CHECK_GPU_ERROR(detail::gpuMemcpy(&temp, get(), sizeof(value_type), gpuMemcpyDeviceToHost));
            return temp;
        }
        __host__ auto operator->() const
        {
            // proxy object
            struct
            {
                value_type t;
                inline value_type* operator->() { return &t; }
            } proxy;

            CHECK_GPU_ERROR(detail::gpuMemcpy(&proxy.t, get(), sizeof(value_type), gpuMemcpyDeviceToHost));
            return proxy;
        }
#endif
        __device__ void reset(pointer ptr)
        {
            assert(ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            std::get<0>(base::data_) = ptr;
            base::size_ = 1;
        }

        __device__ pointer move_to(pointer gpu_ptr, bool is_local = false)
        {
            // must be aligned
            assert(reinterpret_cast<std::uintptr_t>(gpu_ptr) % alignof(ValueType) == 0);

            ValueType*& src = std::get<0>(base::data_);

            const auto block = cooperative_groups::this_thread_block();

            if (block.thread_rank() == 0)
            {
                new (gpu_ptr) ValueType(*src);
                src = gpu_ptr;
            }
            else if (is_local)
            {
                src = gpu_ptr;
            }

            return gpu_ptr + 1;
        }

        __host__ std::pair<std::uintptr_t, std::uintptr_t> alignment(std::uintptr_t addr) const
        {
            return base::template alignment<0>(addr);
        }
    };

    template <typename ValueType>
    class unified_value : protected detail::base<true, ValueType>
    {
        using base = detail::base<true, ValueType>;
        using value_type = ValueType;
        using reference = value_type&;
        using const_reference = const value_type&;
        using pointer = value_type*;
        using const_pointer = const value_type*;

    public:
        using element_type = ValueType;

        __host__ __device__ unified_value(const unified_value& r) : base(r) {};
        __host__ __device__ unified_value(unified_value&& r) noexcept : base(std::move(r)) {};

        __host__ unified_value() : base(1)
        {
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::uninitialized_value_construct_n(get(), 1);
        }

        __host__ explicit unified_value(default_init_tag) : base(1)
        {
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::uninitialized_default_construct_n(get(), 1);
        }

        __host__ explicit unified_value(const ValueType& r) : base(1)
        {
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::construct_at(get(), r);
        }

        __host__ explicit unified_value(ValueType&& r) : base(1)
        {
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::construct_at(get(), std::move(r));
        }

        template <typename... Args>
        requires std::is_constructible_v<ValueType, Args...>
        __host__ explicit unified_value(Args&&... args) : base(1)
        {
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&std::get<0>(base::data_)), sizeof(ValueType)));
            assert(get() != nullptr);
            std::ranges::construct_at(get(), std::forward<Args>(args)...);
        }

        __host__ __device__ unified_value& operator=(const unified_value& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ unified_value& operator=(unified_value&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }

        __host__ __device__ const_reference operator*() const noexcept { return *get(); }
        __host__ __device__ reference operator*() noexcept { return *get(); }
        __host__ __device__ const_pointer operator->() const noexcept { return get(); }
        __host__ __device__ pointer operator->() noexcept { return get(); }

        __host__ __device__ pointer get() noexcept { return std::get<0>(base::data_); }
        __host__ __device__ const_pointer get() const noexcept { return std::get<0>(base::data_); }
        __host__ __device__ explicit operator bool() const noexcept { return std::get<0>(base::data_) != nullptr; }

        __host__ void prefetch(int device_id) const
        {
            assert(base::size_ != 0);
            CHECK_GPU_ERROR(detail::gpuMemPrefetchAsync(get(), sizeof(ValueType), device_id));
        }
        __host__ void prefetch_to_cpu() const { prefetch(gpuCpuDeviceId); }
        __host__ void prefetch_to_gpu() const { prefetch(detail::get_device_id()); }

        __device__ void reset(pointer ptr)
        requires std::is_trivially_copyable_v<ValueType>
        {
            assert(ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            std::get<0>(base::data_) = ptr;
            base::size_ = 1;
        }

        __device__ pointer move_to(pointer gpu_ptr, bool is_local = false)
        requires std::is_trivially_destructible_v<ValueType>
        {
            // must be aligned
            assert(reinterpret_cast<std::uintptr_t>(gpu_ptr) % alignof(ValueType) == 0);

            ValueType*& src = std::get<0>(base::data_);

            const auto block = cooperative_groups::this_thread_block();

            if (block.thread_rank() == 0)
            {
                new (gpu_ptr) ValueType(*src);
                src = gpu_ptr;
            }
            else if (is_local)
            {
                src = gpu_ptr;
            }

            return gpu_ptr + 1;
        }

        __host__ std::pair<std::uintptr_t, std::uintptr_t> alignment(std::uintptr_t addr) const
        {
            return base::template alignment<0>(addr);
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
    class soa_iterator
    {
        std::tuple<Ts*...> ptrs_;

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = Tuple<Ts...>;
        using iterator_concept = std::random_access_iterator_tag;

        __host__ __device__ soa_iterator() = default;
        __host__ __device__ soa_iterator(const soa_iterator&) = default;
        __host__ __device__ soa_iterator(soa_iterator&&) noexcept = default;

        __host__ __device__ soa_iterator& operator=(const soa_iterator&) = default;
        __host__ __device__ soa_iterator& operator=(soa_iterator&&) noexcept = default;

        __host__ __device__ explicit soa_iterator(std::tuple<Ts*...> ptrs) : ptrs_(ptrs) {}

        __host__ __device__ Tuple<Ts&...> operator*() const
        {
            return std::apply([](auto*... ptrs) { return Tuple<Ts&...>(*ptrs...); }, ptrs_);
        }
        __host__ __device__ Tuple<Ts&...> operator[](std::size_t n) const
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
        __host__ __device__ soa_iterator& operator++()
        {
            std::apply([](auto*&... ptrs) { (++ptrs, ...); }, ptrs_);
            return *this;
        }
        __host__ __device__ soa_iterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        __host__ __device__ soa_iterator& operator+=(difference_type n)
        {
            std::apply([n](auto*&... ptrs) { ((ptrs += n), ...); }, ptrs_);
            return *this;
        }
        __host__ __device__ soa_iterator& operator--()
        {
            std::apply([](auto*&... ptrs) { (--ptrs, ...); }, ptrs_);
            return *this;
        }
        __host__ __device__ soa_iterator operator--(int)
        {
            auto tmp = *this;
            --(*this);
            return tmp;
        }
        __host__ __device__ soa_iterator& operator-=(difference_type n)
        {
            std::apply([n](auto*&... ptrs) { ((ptrs -= n), ...); }, ptrs_);
            return *this;
        }

        __host__ __device__ friend difference_type operator-(const soa_iterator& lhs, const soa_iterator& rhs)
        {
            return std::get<0>(lhs.ptrs_) - std::get<0>(rhs.ptrs_);
        }
        __host__ __device__ friend soa_iterator operator+(const soa_iterator& lhs, difference_type n)
        {
            return soa_iterator(std::apply([n](auto*... ptrs) { return std::tuple{ptrs + n...}; }, lhs.ptrs_));
        }
        __host__ __device__ friend soa_iterator operator+(soa_iterator&& lhs, difference_type n)
        {
            lhs += n;
            return std::move(lhs);
        }
        __host__ __device__ friend soa_iterator operator+(difference_type n, const soa_iterator& rhs)
        {
            return rhs + n;
        }
        __host__ __device__ friend soa_iterator operator+(difference_type n, soa_iterator&& rhs)
        {
            return std::move(rhs) + n;
        }
        __host__ __device__ friend soa_iterator operator-(const soa_iterator& lhs, difference_type n)
        {
            return soa_iterator(std::apply([n](auto*... ptrs) { return std::tuple{ptrs - n...}; }, lhs.ptrs_));
        }
        __host__ __device__ friend soa_iterator operator-(soa_iterator&& lhs, difference_type n)
        {
            lhs -= n;
            return std::move(lhs);
        }

        __host__ __device__ friend bool operator==(const soa_iterator& lhs, const soa_iterator& rhs)
        {
            return std::get<0>(lhs.ptrs_) == std::get<0>(rhs.ptrs_);
        }
        __host__ __device__ friend std::strong_ordering operator<=>(const soa_iterator& lhs, const soa_iterator& rhs)
        {
            return std::get<0>(lhs.ptrs_) <=> std::get<0>(rhs.ptrs_);
        }
    };

    template <typename... Ts>
    class soa_ptr : public soa_ptr<std::tuple<Ts...>>
    {
        using base = soa_ptr<std::tuple<Ts...>>;
        using base::base;

    public:
        template <std::size_t N>
        using element_type = base::template element_type<N>;
        using base::operator=;
    };

    template <template <typename...> typename Tuple, typename... Ts>
    requires (sizeof...(Ts) > 0) && std::constructible_from<Tuple<Ts...>, const Ts&...> &&
             std::constructible_from<Tuple<Ts&...>, Ts&...> &&
             std::constructible_from<Tuple<const Ts&...>, const Ts&...> && (std::is_trivially_copyable_v<Ts> && ...)
    class soa_ptr<Tuple<Ts...>> : public detail::base<false, Ts...>
    {
        static constexpr auto num_arrays = sizeof...(Ts);
        using base = detail::base<false, Ts...>;

        using tuple_value_type = std::tuple<Ts...>;
        using tuple_pointer_type = std::tuple<Ts*...>;
        using tuple_const_pointer_type = std::tuple<const Ts*...>;
        using ret_tuple_value_type = Tuple<Ts...>;
        using ret_tuple_reference_type = Tuple<Ts&...>;
        using ret_tuple_const_reference_type = Tuple<const Ts&...>;
        using iterator_type = soa_iterator<Tuple, Ts...>;
        using const_iterator_type = soa_iterator<Tuple, const Ts...>;

    public:
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
        SIGSEGV_DEPRECATED __host__ __device__ auto operator[](std::size_t i) &
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_reference_type{*(ptrs + i)...}; }, base::data_);
        }
        SIGSEGV_DEPRECATED __host__ __device__ auto operator[](std::size_t i) const&
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_const_reference_type{*(ptrs + i)...}; },
                              base::data_);
        }
        SIGSEGV_DEPRECATED __host__ __device__ auto operator[](std::size_t i) &&
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

        __host__ __device__ soa_ptr() = default;
        __host__ __device__ soa_ptr(const soa_ptr& r) : base(r) {}
        __host__ __device__ soa_ptr(soa_ptr&& r) noexcept : base(std::move(r)) {}

        __host__ explicit soa_ptr(base::size_type size) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                auto buf = std::make_unique<T[]>(base::size_);
                CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                CHECK_GPU_ERROR(detail::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            });
        }

        __host__ explicit soa_ptr(base::size_type size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                if constexpr (!std::is_trivially_default_constructible_v<T>)
                {
                    auto buf = std::make_unique<T[]>(base::size_);
                    CHECK_GPU_ERROR(detail::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
                }
            });
        }

        template <std::ranges::forward_range Range, typename Element = std::ranges::range_value_t<Range>>
        requires std::ranges::sized_range<Range> && detail::assignable_to_tuple<Element, Ts...>
        __host__ explicit soa_ptr(const Range& array) : base(std::ranges::size(array))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this]<typename T>(T*& ptr, auto view_range) {
                auto buf = std::make_unique_for_overwrite<T[]>(base::size_);
                std::ranges::copy(view_range, buf.get());
                CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                CHECK_GPU_ERROR(detail::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            };

            [this, &array, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_),
                           array | std::views::transform([](const auto& v) { return (std::get<N>(v)); })),
                 ...);
            }(std::make_index_sequence<num_arrays>());
        }

        template <std::ranges::forward_range... Ranges>
        requires (sizeof...(Ranges) == num_arrays) && (std::ranges::sized_range<Ranges> && ...) &&
                 detail::assignable_to_tuple<std::tuple<std::ranges::range_value_t<Ranges>...>, Ts...>
        __host__ explicit soa_ptr(const Ranges&... arrays) : base(std::max({std::ranges::size(arrays)...}))
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
                CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                CHECK_GPU_ERROR(detail::gpuMemcpy(ptr, buf.get(), sizeof(T) * base::size_, gpuMemcpyHostToDevice));
            };

            auto arrays_tuple = std::tuple<const Ranges&...>(arrays...);
            [this, &arrays_tuple, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), std::get<N>(arrays_tuple)), ...);
            }(std::make_index_sequence<num_arrays>());
        }

        __host__ __device__ soa_ptr& operator=(const soa_ptr& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ soa_ptr& operator=(soa_ptr&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }

        template <std::ranges::range Range, typename U = std::ranges::range_value_t<Range>>
        requires std::is_default_constructible_v<Range> && std::is_constructible_v<U, Ts...>
        __host__ Range to() const
        {
            const auto copy_buffer = [this]<typename T>(const T* ptr) {
                auto buf = std::make_unique_for_overwrite<T[]>(base::size_);
                CHECK_GPU_ERROR(detail::gpuMemcpy(buf.get(), ptr, sizeof(T) * base::size_, gpuMemcpyDeviceToHost));
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
                // back insertable type
                if constexpr (requires { std::declval<Range>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }

                for (typename base::size_type i = 0; i < base::size_; ++i)
                {
                    std::apply([&result, i](const auto&... bufs) { result.push_back(U{bufs[i]...}); }, tmp_bufs);
                }

                return result;
            }
            else
            {
                // std::array like type
                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");

                for (typename base::size_type i = 0; i < base::size_; ++i)
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
        requires requires(const soa_ptr& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

        template <std::size_t N>
        __device__ void reset(std::tuple_element_t<N, tuple_pointer_type> ptr)
        {
            assert(ptr != nullptr);

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
    };

    template <typename... Ts>
    class unified_soa_ptr : public unified_soa_ptr<std::tuple<Ts...>>
    {
        using base = unified_soa_ptr<std::tuple<Ts...>>;
        using base::base;

    public:
        template <std::size_t N>
        using element_type = base::template element_type<N>;
        using base::operator=;
    };

    template <template <typename...> typename Tuple, typename... Ts>
    requires (sizeof...(Ts) > 0) && std::constructible_from<Tuple<Ts...>, const Ts&...> &&
             std::constructible_from<Tuple<Ts&...>, Ts&...> &&
             std::constructible_from<Tuple<const Ts&...>, const Ts&...>
    class unified_soa_ptr<Tuple<Ts...>> : public detail::base<true, Ts...>
    {
        static constexpr auto num_arrays = sizeof...(Ts);
        using base = detail::base<true, Ts...>;

        using tuple_value_type = std::tuple<Ts...>;
        using tuple_pointer_type = std::tuple<Ts*...>;
        using tuple_const_pointer_type = std::tuple<const Ts*...>;
        using ret_tuple_value_type = Tuple<Ts...>;
        using ret_tuple_reference_type = Tuple<Ts&...>;
        using ret_tuple_const_reference_type = Tuple<const Ts&...>;
        using iterator_type = soa_iterator<Tuple, Ts...>;
        using const_iterator_type = soa_iterator<Tuple, const Ts...>;

    protected:
        template <detail::array_convertible Range, typename U = std::ranges::range_value_t<Range>>
        requires (detail::assignable_to_tuple<std::ranges::range_value_t<U>, Ts...>)
        __host__ explicit unified_soa_ptr(const Range& nested_array, detail::join_init_tag)
            : base(std::accumulate(std::ranges::begin(nested_array), std::ranges::end(nested_array), std::size_t{0},
                                   [](auto acc, const auto& r) { return acc + std::ranges::size(r); }))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this, &nested_array]<typename T>(T*& ptr, auto f) {
                CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
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
        __host__ __device__ auto operator[](std::size_t i) &
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_reference_type{*(ptrs + i)...}; }, base::data_);
        }
        __host__ __device__ auto operator[](std::size_t i) const&
        {
            assert(i < base::size_);
            return std::apply([i](auto&... ptrs) { return ret_tuple_const_reference_type{*(ptrs + i)...}; },
                              base::data_);
        }
        __host__ __device__ auto operator[](std::size_t i) &&
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

        __host__ __device__ unified_soa_ptr() = default;
        __host__ __device__ unified_soa_ptr(const unified_soa_ptr& r) : base(r) {}
        __host__ __device__ unified_soa_ptr(unified_soa_ptr&& r) noexcept : base(std::move(r)) {}

        __host__ explicit unified_soa_ptr(base::size_type size) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                std::ranges::uninitialized_value_construct(ptr, ptr + base::size_);
            });
        }

        __host__ explicit unified_soa_ptr(base::size_type size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this]<typename T>(T*& ptr) {
                CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                std::ranges::uninitialized_default_construct(ptr, ptr + base::size_);
            });
        }

        template <detail::array_convertible Range, typename Element = std::ranges::range_value_t<Range>>
        requires detail::assignable_to_tuple<Element, Ts...>
        __host__ explicit unified_soa_ptr(const Range& array) : base(std::ranges::size(array))
        {
            if (base::size_ == 0) return;

            const auto alloc_ptr = [this, &array]<typename T>(T*& ptr, auto f) {
                CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
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

        template <detail::array_convertible... Ranges>
        requires (sizeof...(Ranges) == num_arrays) &&
                 detail::assignable_to_tuple<std::tuple<std::ranges::range_value_t<Ranges>...>, Ts...>
        __host__ explicit unified_soa_ptr(const Ranges&... arrays) : base(std::max({std::ranges::size(arrays)...}))
        {
            if (base::size_ == 0) return;

            auto sizes = std::array{std::ranges::size(arrays)...};
            if (!std::ranges::all_of(sizes, [s = base::size_](auto x) { return x == s; }))
            {
                throw std::invalid_argument("the sizes of arrays are not equal");
            }

            const auto alloc_ptr = [this]<typename T>(T*& ptr, const auto& range) {
                CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&ptr), sizeof(T) * base::size_));
                assert(ptr != nullptr);
                for (auto i = std::size_t{0}; const auto& v : range) std::ranges::construct_at(ptr + i++, v);
            };

            auto arrays_tuple = std::tuple<const Ranges&...>(arrays...);
            [this, &arrays_tuple, alloc_ptr]<std::size_t... N>(std::index_sequence<N...>) {
                (alloc_ptr(std::get<N>(base::data_), std::get<N>(arrays_tuple)), ...);
            }(std::make_index_sequence<num_arrays>());
        }

        __host__ __device__ unified_soa_ptr& operator=(const unified_soa_ptr& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ unified_soa_ptr& operator=(unified_soa_ptr&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }

        __host__ void prefetch(int device_id) const
        {
            if (base::size_ == 0) return;

            base::tuple_for_each([this, device_id]<typename T>(T* ptr) {
                CHECK_GPU_ERROR(detail::gpuMemPrefetchAsync(ptr, sizeof(T) * base::size_, device_id));
            });
        }
        __host__ void prefetch_to_cpu() const { prefetch(gpuCpuDeviceId); }
        __host__ void prefetch_to_gpu() const { prefetch(detail::get_device_id()); }

        template <std::ranges::range Range, typename U = std::ranges::range_value_t<Range>>
        requires std::is_default_constructible_v<Range> && std::is_constructible_v<U, Ts...>
        __host__ Range to() const
        {
            prefetch_to_cpu();
            auto result = Range();

            if constexpr (requires { std::declval<Range>().push_back(std::declval<U>()); })
            {
                // back insertable type
                if constexpr (requires { std::declval<Range>().reserve(std::declval<std::size_t>()); })
                {
                    result.reserve(base::size_);
                }

                for (typename base::size_type i = 0; i < base::size_; ++i)
                {
                    std::apply([&result, i](const auto&... bufs) { result.push_back(U{bufs[i]...}); }, base::data_);
                }

                return result;
            }
            else
            {
                // std::array like type
                if (std::ranges::size(result) != base::size_) throw std::runtime_error("size mismatch");

                for (typename base::size_type i = 0; i < base::size_; ++i)
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
        requires requires(const unified_soa_ptr& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

        template <std::size_t N>
        __device__ void reset(std::tuple_element_t<N, tuple_pointer_type> ptr)
        {
            assert(ptr != nullptr);

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
    };

    template <detail::array_convertible... Range>
    soa_ptr(const Range&... array) -> soa_ptr<std::ranges::range_value_t<Range>...>;
    template <detail::array_convertible Range>
    soa_ptr(const Range& array) -> soa_ptr<std::ranges::range_value_t<Range>>;
    template <detail::array_convertible... Range>
    unified_soa_ptr(const Range&... array) -> unified_soa_ptr<std::ranges::range_value_t<Range>...>;
    template <detail::array_convertible Range>
    unified_soa_ptr(const Range& array) -> unified_soa_ptr<std::ranges::range_value_t<Range>>;

    template <gpu_unified_smart_ptr ArrayType>
    class jagged_array : public ArrayType
    {
        using base = ArrayType;
        using offsets_type = unified_array_ptr<std::size_t>;
        using iterator_type = std::ranges::iterator_t<ArrayType>;
        using const_iterator_type = decltype(std::ranges::cbegin(std::declval<ArrayType&>()));

        offsets_type offsets_;
        static constexpr auto has_data = requires(const ArrayType& a) { a.data(); };
        static constexpr auto has_prefetch = requires(const ArrayType& a) { a.prefetch(); };

    public:
        __host__ __device__ jagged_array() = default;
        __host__ __device__ jagged_array(const jagged_array&) = default;
        __host__ __device__ jagged_array(jagged_array&&) noexcept = default;

        template <std::ranges::forward_range Range>
        requires std::ranges::sized_range<Range> &&
                     std::constructible_from<std::size_t, std::ranges::range_value_t<Range>>
        __host__ explicit jagged_array(const Range& sizes)
            : base(std::accumulate(std::ranges::begin(sizes), std::ranges::end(sizes), std::size_t{0})),
              offsets_(std::ranges::size(sizes) + 1, default_init)
        {
            prefetch_to_cpu();

            offsets_[0] = 0;
            for (std::size_t i = 0; const auto& s : sizes)
            {
                offsets_[i + 1] = offsets_[i] + static_cast<std::size_t>(s);
                ++i;
            }
        }

        template <detail::array_convertible Range, detail::array_convertible Inner = std::ranges::range_value_t<Range>,
                  typename Element = std::ranges::range_value_t<Inner>>
        __host__ explicit jagged_array(const Range& nested_array)
            : base(nested_array, detail::join_init), offsets_(std::ranges::size(nested_array) + 1, default_init)
        {
            prefetch_to_cpu();

            offsets_[0] = 0;
            for (std::size_t i = 0; const auto& a : nested_array)
            {
                offsets_[i + 1] = offsets_[i] + static_cast<std::size_t>(std::ranges::size(a));
                ++i;
            }
        }

        __host__ __device__ jagged_array& operator=(const jagged_array&) = default;
        __host__ __device__ jagged_array& operator=(jagged_array&&) noexcept = default;

        using base::begin;
        using base::empty;
        using base::end;
        using base::size;
        using base::operator[];

        __host__ __device__ auto begin(std::size_t i) noexcept { return base::begin() + offsets_[i]; }
        __host__ __device__ auto end(std::size_t i) noexcept { return base::begin() + offsets_[i + 1]; }
        __host__ __device__ auto begin(std::size_t i) const noexcept { return base::begin() + offsets_[i]; }
        __host__ __device__ auto end(std::size_t i) const noexcept { return base::begin() + offsets_[i + 1]; }

        __host__ __device__ auto size(std::size_t i) const noexcept
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
        __host__ __device__ auto data(std::size_t i) noexcept
        requires has_data
        {
            assert(i < num_rows());
            return base::data() + offsets_[i];
        }
        __host__ __device__ auto data(std::size_t i) const noexcept
        requires has_data
        {
            assert(i < num_rows());
            return base::data() + offsets_[i];
        }

        __host__ __device__ decltype(auto) operator[](std::array<std::size_t, 2> idx) &
        {
            assert(idx[0] < num_rows());
            assert(idx[1] < size(idx[0]));
            return (*this)[offsets_[idx[0]] + idx[1]];
        }
        __host__ __device__ decltype(auto) operator[](std::array<std::size_t, 2> idx) const&
        {
            assert(idx[0] < num_rows());
            assert(idx[1] < size(idx[0]));
            return (*this)[offsets_[idx[0]] + idx[1]];
        }
        __host__ __device__ decltype(auto) operator[](std::array<std::size_t, 2> idx) &&
        {
            assert(idx[0] < num_rows());
            assert(idx[1] < size(idx[0]));
            return (*this)[offsets_[idx[0]] + idx[1]];
        }

        __host__ __device__ auto row(std::size_t i) noexcept
        {
            assert(i < num_rows());
            return std::ranges::subrange<iterator_type, iterator_type, std::ranges::subrange_kind::sized>(
                base::begin() + offsets_[i], base::begin() + offsets_[i + 1], offsets_[i + 1] - offsets_[i]);
        }
        __host__ __device__ auto row(std::size_t i) const noexcept
        {
            assert(i < num_rows());
            return std::ranges::subrange<const_iterator_type, const_iterator_type, std::ranges::subrange_kind::sized>(
                base::begin() + offsets_[i], base::begin() + offsets_[i + 1], offsets_[i + 1] - offsets_[i]);
        }

        __host__ __device__ auto num_rows() const noexcept { return offsets_.size() - 1; }

        __host__ void prefetch(int device_id) const
        {
            if constexpr (has_prefetch) base::prefetch(device_id);
            offsets_.prefetch(device_id);
        }
        __host__ void prefetch_to_cpu() const
        {
            if constexpr (has_prefetch) base::prefetch_to_cpu();
            offsets_.prefetch_to_cpu();
        }
        __host__ void prefetch_to_gpu() const
        {
            if constexpr (has_prefetch) base::prefetch_to_gpu();
            offsets_.prefetch_to_gpu();
        }

        [[deprecated("for debug")]] [[nodiscard]] const auto& get_offsets() const noexcept { return offsets_; }
        [[deprecated("for debug")]] [[nodiscard]] auto get_sizes() const noexcept
        {
            auto result = std::vector<std::size_t>();
            result.reserve(num_rows());
            for (std::size_t i = 0; i < num_rows(); ++i) result.emplace_back(offsets_[i + 1] - offsets_[i]);
            return result;
        }
    };

}  // namespace gpu_smart_ptr

#undef GPU_DEVICE_COMPILE
#undef OVERLOAD_DEVICE
#undef OVERLOAD_HOST
#undef SIGSEGV_DEPRECATED
#undef INCR_GPU_MEORY_USAGE
#undef DECR_GPU_MEORY_USAGE
#undef GPU_MEMORY_USAGE_EQ
#undef CHECK_GPU_ERROR
