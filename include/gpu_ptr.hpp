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
#include "gpu_runtime_api.hpp"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <iostream>
#include <memory>
#include <vector>

#if defined(__HIP_DEVICE_COMPILE__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)
#define GPU_DEVICE_COMPILE
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

namespace gpu_ptr
{

    template <typename ValueType>
    requires std::is_trivially_copyable_v<ValueType>
    class array;
    template <typename ValueType>
    class unified_array;

    inline constexpr struct default_init_tag
    {
        struct tag
        {
        };
        constexpr explicit default_init_tag(tag) {}
    } default_init{default_init_tag::tag{}};

    namespace detail
    {
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

        inline auto gpu_memory_usage =
            std::vector<std::pair<std::size_t, std::size_t>>(static_cast<std::size_t>(get_device_count()), {0, 0});

        template <typename ValueType, bool Unified = false>
        class array_base
        {
        public:
#pragma region std::array like implementations
            using reference = ValueType&;
            using const_reference = const ValueType&;
            using iterator = ValueType*;
            using const_iterator = const ValueType*;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;
            using pointer = ValueType*;
            using const_pointer = const ValueType*;
            using value_type = ValueType;

            __host__ __device__ size_type size() const noexcept { return size_; }
            __host__ __device__ size_type max_size() const noexcept { return size_; }
            __host__ __device__ bool empty() const noexcept { return size_ == 0; }
#pragma endregion std::array like implementations

        protected:
            std::size_t size_ = 0;
            ValueType* data_ = nullptr;
            std::uint32_t* ref_count_ = nullptr;  // reference counter, not used on GPU
#ifdef GPU_DEVICE_COMPILE
            int device_id_ = gpuInvalidDeviceId;
#else
            int device_id_ = get_device_id();
#endif

            __host__ __device__ void init()
            {
                size_ = 0;
                data_ = nullptr;
                ref_count_ = nullptr;
#ifdef GPU_DEVICE_COMPILE
                device_id_ = gpuInvalidDeviceId;
#else
                device_id_ = get_device_id();
#endif
            }
            __host__ void free()
            {
#ifndef GPU_DEVICE_COMPILE
                assert((data_ == nullptr) == (size_ == 0));
                assert((size_ == 0) == (ref_count_ == nullptr));
                if (ref_count_ == nullptr) return;

                // delete objects
                if (--*ref_count_ == 0)
                {
                    // do not throw in destructor
                    try
                    {
                        assert(data_ != nullptr);
                        // std::cout << "gpuFree: " << NAMEOF_FULL_TYPE(ValueType) << ' ' << size_ << '\n';
                        if constexpr (Unified)
                        {
                            // call destructor explicitly for unified memory
                            std::destroy_n(data_, size_);
                            gpu_memory_usage[device_id_].second -= sizeof(ValueType) * size_;
                        }
                        else
                        {
                            gpu_memory_usage[device_id_].first -= sizeof(ValueType) * size_;
                        }
                        CHECK_GPU_ERROR(gpuFree(data_));
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
#ifndef GPU_DEVICE_COMPILE
            __host__ ~array_base() { free(); }
#endif
            __host__ __device__ array_base& operator=(const array_base& r)
            {
#ifndef GPU_DEVICE_COMPILE
                free();
#endif

                size_ = r.size_;
                data_ = r.data_;
                ref_count_ = r.ref_count_;
                device_id_ = r.device_id_;
                if (ref_count_ != nullptr) ++*ref_count_;

                return *this;
            }
            __host__ __device__ array_base& operator=(array_base&& r) noexcept
            {
#ifndef GPU_DEVICE_COMPILE
                free();
#endif

                size_ = r.size_;
                data_ = r.data_;
                ref_count_ = r.ref_count_;
                device_id_ = r.device_id_;
                r.init();

                return *this;
            }

            __host__ __device__ array_base() = default;
            __host__ explicit array_base(std::size_t size)
                : size_(size), ref_count_(size_ == 0 ? nullptr : new std::uint32_t(1))
            {
                // increment GPU memory usage
                if constexpr (Unified)
                {
                    gpu_memory_usage[device_id_].second += sizeof(ValueType) * size_;
                }
                else
                {
                    gpu_memory_usage[device_id_].first += sizeof(ValueType) * size_;
                }
            };
            __device__ array_base(pointer ptr, std::size_t size) : size_(size), data_(size != 0 ? ptr : nullptr)
            {
                assert(size_ == 0 || data_ != nullptr);
            }
            __host__ __device__ array_base(const array_base& r)
                : size_(r.size_), data_(r.data_), ref_count_(r.ref_count_)
            {
#ifndef GPU_DEVICE_COMPILE
                if (ref_count_ != nullptr) ++*ref_count_;
#endif
            }
            __host__ __device__ array_base(array_base&& r) noexcept
                : size_(r.size_), data_(r.data_), ref_count_(r.ref_count_)
            {
                r.init();
            }
        };

        template <typename T>
        concept array_convertible = std::ranges::input_range<T> && std::ranges::sized_range<T>;

        template <typename ValueType, bool Unified>
        constexpr void is_array_helper(const detail::array_base<ValueType, Unified>&)
        {
        }
        template <typename ValueType>
        constexpr void is_unified_array_helper(const detail::array_base<ValueType, true>&)
        {
        }

        template <class Derived>
        concept gpu_array = requires(Derived d) { is_array_helper(d); };
        template <class Derived>
        concept gpu_unified_array = requires(Derived d) { is_unified_array_helper(d); };

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
            using type = unified_array<unified_array_deduced_t<std::ranges::range_value_t<T>>>;
        };

        template <gpu_array T, template <typename...> typename U>
        struct to_range_deduced
        {
            using type = U<std::ranges::range_value_t<T>>;
        };
        template <typename T, template <typename...> typename U>
        using to_range_deduced_t = typename to_range_deduced<T, U>::type;

        template <gpu_array Array, template <typename...> typename U>
        requires gpu_array<std::ranges::range_value_t<Array>>
        struct to_range_deduced<Array, U>
        {
            using type = U<to_range_deduced_t<std::ranges::range_value_t<Array>, U>>;
        };

    }  // namespace detail

    using detail::gpu_array;
    using detail::gpu_unified_array;

    __host__ inline std::size_t gpu_memory_usage(int device_id)
    {
        if (device_id < 0 || device_id >= detail::get_device_count())
        {
            throw std::runtime_error("Invalid device_id");
        }
        return detail::gpu_memory_usage[static_cast<std::size_t>(device_id)].first +
               detail::gpu_memory_usage[static_cast<std::size_t>(device_id)].second;
    }

    __host__ inline std::size_t gpu_memory_usage() { return gpu_memory_usage(detail::get_device_id()); }

    template <typename ValueType>
    requires std::is_trivially_copyable_v<ValueType>
    class array : public detail::array_base<ValueType>
    {
        using base = detail::array_base<ValueType>;

    public:
        SIGSEGV_DEPRECATED __host__ __device__ base::const_reference operator[](std::size_t i) const noexcept
        {
            return base::data_[i];
        };
        SIGSEGV_DEPRECATED __host__ __device__ base::reference operator[](std::size_t i) noexcept
        {
            return base::data_[i];
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::iterator begin() noexcept { return base::data_; }
        SIGSEGV_DEPRECATED __host__ __device__ base::iterator end() noexcept { return base::data_ + base::size_; }
        SIGSEGV_DEPRECATED __host__ __device__ base::iterator rbegin() noexcept
        {
            return std::reverse_iterator<typename base::iterator>(end());
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::iterator rend() noexcept
        {
            return std::reverse_iterator<typename base::iterator>(begin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator begin() const noexcept { return base::data_; }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator end() const noexcept
        {
            return base::data_ + base::size_;
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator rbegin() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(end());
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator rend() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(begin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator cbegin() const noexcept { return base::data_; }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator cend() const noexcept
        {
            return base::data_ + base::size_;
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator crbegin() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(cend());
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_iterator crend() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(cbegin());
        }
        SIGSEGV_DEPRECATED __host__ __device__ base::reference front() noexcept { return *begin(); }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_reference front() const noexcept { return *begin(); }
        SIGSEGV_DEPRECATED __host__ __device__ base::reference back() noexcept { return *(--end()); }
        SIGSEGV_DEPRECATED __host__ __device__ base::const_reference back() const noexcept { return *(--end()); }
        __host__ __device__ base::pointer data() noexcept { return base::data_; }
        __host__ __device__ base::const_pointer data() const noexcept { return base::data_; }

        __host__ __device__ array() = default;
        __host__ __device__ array(const array& r) : base(r) {}
        __host__ __device__ array(array&& r) noexcept : base(std::move(r)) {}

        __host__ explicit array(std::size_t size) : base(size)
        {
            if (base::size_ == 0) return;
            auto buf = std::make_unique<ValueType[]>(base::size_);
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(
                detail::gpuMemcpy(base::data_, buf.get(), sizeof(ValueType) * base::size_, gpuMemcpyHostToDevice));
        }

        __host__ array(std::size_t size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            if constexpr (!std::is_trivially_default_constructible_v<ValueType>)
            {
                auto buf = std::make_unique<ValueType[]>(base::size_);
                CHECK_GPU_ERROR(
                    detail::gpuMemcpy(base::data_, buf.get(), sizeof(ValueType) * base::size_, gpuMemcpyHostToDevice));
            }
        }

        __host__ array(std::size_t size, const ValueType& value) : base(size)
        {
            if (base::size_ == 0) return;

            auto al = std::allocator<ValueType>();
            auto buf = al.allocate(base::size_);
            std::ranges::uninitialized_fill(buf, buf + base::size_, value);
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(
                detail::gpuMemcpy(base::data_, buf, sizeof(ValueType) * base::size_, gpuMemcpyHostToDevice));
            al.deallocate(buf, base::size_);
        }

        template <detail::array_convertible T, std::same_as<ValueType> U = std::ranges::range_value_t<T>>
        requires std::ranges::contiguous_range<T>
        __host__ explicit array(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(base::data_, std::ranges::data(r), sizeof(ValueType) * base::size_,
                                              gpuMemcpyHostToDevice));
        }

        template <detail::array_convertible T, typename U = std::ranges::range_value_t<T>>
        requires std::is_constructible_v<ValueType, U> &&
                 (!std::same_as<ValueType, U> || !std::ranges::contiguous_range<T>)
        __host__ explicit array(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;

            auto al = std::allocator<ValueType>();
            auto buf = al.allocate(base::size_);
            for (auto i = std::size_t{0}; const auto& v : r) std::ranges::construct_at(buf + i++, v);

            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(
                detail::gpuMemcpy(base::data_, buf, sizeof(ValueType) * base::size_, gpuMemcpyHostToDevice));
            al.deallocate(buf, base::size_);
        }

        __host__ array(std::initializer_list<ValueType> r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(base::data_, std::ranges::data(r), sizeof(ValueType) * base::size_,
                                              gpuMemcpyHostToDevice));
        }

        __device__ array(base::pointer ptr, std::size_t size) : base(ptr, size) {}

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

        template <std::ranges::range T, std::same_as<ValueType> U = std::ranges::range_value_t<T>>
        requires (std::ranges::contiguous_range<T> &&
                  (std::is_constructible_v<T, std::size_t> || std::is_default_constructible_v<T>)) ||
                 (std::is_default_constructible_v<T> &&
                  requires { std::declval<T>().push_back(std::declval<ValueType>()); })
        __host__ T to() const
        {
            if constexpr (gpu_array<T>)
            {
                // array<ValueType> or unified_array<ValueType> like type
                return T(*this);
            }
            else if constexpr (std::ranges::contiguous_range<T>)
            {
                auto result = []([[maybe_unused]] std::size_t size) -> T {
                    if constexpr (std::is_constructible_v<T, std::size_t>)
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
                CHECK_GPU_ERROR(detail::gpuMemcpy(std::ranges::data(result), base::data_,
                                                  sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));
                return result;
            }
            else
            {
                // std::list like type
                static_assert(requires { std::declval<T>().push_back(std::declval<ValueType>()); });
                auto buf = std::make_unique_for_overwrite<ValueType[]>(base::size_);
                CHECK_GPU_ERROR(
                    detail::gpuMemcpy(buf.get(), base::data_, sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));

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
                detail::gpuMemcpy(buf.get(), base::data_, sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));

            if constexpr (gpu_array<T>)
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
        requires requires(const array& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

        __device__ void reset([[maybe_unused]] base::pointer ptr, [[maybe_unused]] std::size_t size)
        {
#ifdef GPU_DEVICE_COMPILE
            assert(size == 0 || ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            base::data_ = size == 0 ? nullptr : ptr;
            base::size_ = size;
#endif
        }

        __host__ void free()
        {
#ifndef GPU_DEVICE_COMPILE
            base::free();
#endif
        }
    };

    template <typename ValueType>
    class unified_array : public detail::array_base<ValueType, true>
    {
        using base = detail::array_base<ValueType, true>;

    public:
        __host__ __device__ base::const_reference operator[](std::size_t i) const noexcept { return base::data_[i]; };
        __host__ __device__ base::reference operator[](std::size_t i) noexcept { return base::data_[i]; }
        __host__ __device__ base::iterator begin() noexcept { return base::data_; }
        __host__ __device__ base::iterator end() noexcept { return base::data_ + base::size_; }
        __host__ __device__ base::iterator rbegin() noexcept
        {
            return std::reverse_iterator<typename base::iterator>(end());
        }
        __host__ __device__ base::iterator rend() noexcept
        {
            return std::reverse_iterator<typename base::iterator>(begin());
        }
        __host__ __device__ base::const_iterator begin() const noexcept { return base::data_; }
        __host__ __device__ base::const_iterator end() const noexcept { return base::data_ + base::size_; }
        __host__ __device__ base::const_iterator rbegin() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(end());
        }
        __host__ __device__ base::const_iterator rend() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(begin());
        }
        __host__ __device__ base::const_iterator cbegin() const noexcept { return base::data_; }
        __host__ __device__ base::const_iterator cend() const noexcept { return base::data_ + base::size_; }
        __host__ __device__ base::const_iterator crbegin() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(cend());
        }
        __host__ __device__ base::const_iterator crend() const noexcept
        {
            return std::reverse_iterator<typename base::const_iterator>(cbegin());
        }
        __host__ __device__ base::reference front() noexcept { return *begin(); }
        __host__ __device__ base::const_reference front() const noexcept { return *begin(); }
        __host__ __device__ base::reference back() noexcept { return *(--end()); }
        __host__ __device__ base::const_reference back() const noexcept { return *(--end()); }
        __host__ __device__ base::pointer data() noexcept { return base::data_; }
        __host__ __device__ base::const_pointer data() const noexcept { return base::data_; }

        __host__ __device__ unified_array() = default;
        __host__ __device__ unified_array(const unified_array& r) : base(r) {}
        __host__ __device__ unified_array(unified_array&& r) noexcept : base(std::move(r)) {}

        __host__ explicit unified_array(std::size_t size) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            std::ranges::uninitialized_value_construct(*this);
        }

        __host__ explicit unified_array(std::size_t size, default_init_tag) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            std::ranges::uninitialized_default_construct(*this);
        }

        __host__ unified_array(std::size_t size, const ValueType& value) : base(size)
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            std::ranges::uninitialized_fill(*this, value);
        }

        template <detail::array_convertible T, typename U = std::ranges::range_value_t<T>>
        requires std::is_constructible_v<ValueType, U>
        __host__ explicit unified_array(const T& r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            for (auto i = std::size_t{0}; const auto& v : r) std::ranges::construct_at(base::data_ + i++, v);
        }

        __host__ unified_array(std::initializer_list<ValueType> r) : base(std::ranges::size(r))
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            for (auto i = std::size_t{0}; const auto& v : r) std::ranges::construct_at(base::data_ + i++, v);
        }

        template <typename U>
        requires std::is_trivially_copyable_v<U> && std::same_as<U, ValueType>
        __host__ explicit unified_array(const array<U>& r) : base(r.size())
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(
                detail::gpuMemcpy(base::data_, r.data(), sizeof(ValueType) * base::size_, gpuMemcpyDeviceToHost));
        }

        template <typename U>
        requires std::is_trivially_copyable_v<U> &&
                 (!std::same_as<U, ValueType>) && std::is_constructible_v<ValueType, U>
        __host__ explicit unified_array(const array<U>& r) : base(r.size())
        {
            if (base::size_ == 0) return;
            auto buf = std::make_unique_for_overwrite<U[]>(base::size_);
            CHECK_GPU_ERROR(detail::gpuMemcpy(buf.get(), r.data(), sizeof(U) * base::size_, gpuMemcpyDeviceToHost));
            CHECK_GPU_ERROR(
                detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType) * base::size_));
            assert(base::data_ != nullptr);
            for (auto i = std::size_t{0}; i < base::size_; ++i)
            {
                std::ranges::construct_at(base::data_ + i, *(buf.get() + i));
            }
        }

        __host__ __device__ unified_array& operator=(const unified_array& r)
        {
            base::operator=(r);
            return *this;
        }
        __host__ __device__ unified_array& operator=(unified_array&& r) noexcept
        {
            base::operator=(std::move(r));
            return *this;
        }
        __host__ unified_array& operator=(std::initializer_list<ValueType> r) noexcept
        {
            base::operator=(unified_array(r));
            return *this;
        }

        __host__ void prefetch(int device_id) const
        {
            if (base::size_ == 0) return;
            CHECK_GPU_ERROR(detail::gpuMemPrefetchAsync(base::data_, sizeof(ValueType) * base::size_, device_id));
        }
        __host__ void prefetch_to_cpu() const { prefetch(gpuCpuDeviceId); }
        __host__ void prefetch_to_gpu() const { prefetch(base::device_id_); }

        template <std::ranges::range T, typename U = std::ranges::range_value_t<T>>
        requires std::is_default_constructible_v<T> && requires(const ValueType& v) { static_cast<U>(v); }
        __host__ T to() const
        {
            prefetch_to_cpu();
            if constexpr (gpu_array<T>)
            {
                // array<U> or unified_array<U>
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
            return to<detail::to_range_deduced_t<unified_array, U>>();
        }

        template <typename T>
        requires requires(const unified_array& self) { self.to<T>(); }
        __host__ explicit operator T() const
        {
            return to<T>();
        }

        __device__ void reset([[maybe_unused]] base::pointer ptr, [[maybe_unused]] std::size_t size)
        requires std::is_trivially_copyable_v<ValueType>
        {
#ifdef GPU_DEVICE_COMPILE
            assert(size == 0 || ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            base::data_ = size == 0 ? nullptr : ptr;
            base::size_ = size;
#endif
        }

        __host__ void free()
        {
#ifndef GPU_DEVICE_COMPILE
            base::free();
#endif
        }
    };

    // deduction guide for arrays
    template <detail::array_convertible T>
    __host__ array(const T& r) -> array<std::ranges::range_value_t<T>>;
    template <detail::array_convertible T>
    __host__ unified_array(const T& r) -> unified_array<detail::unified_array_deduced_t<std::ranges::range_value_t<T>>>;

    template <typename ValueType>
    requires std::is_trivially_copyable_v<ValueType>
    class value : protected detail::array_base<ValueType>
    {
        using base = detail::array_base<ValueType>;

    public:
        using element_type = ValueType;

        __host__ __device__ value(const value& r) : base(r) {};
        __host__ __device__ value(value&& r) noexcept : base(std::move(r)) {};

        __host__ value() : base(1)
        {
            auto temp = ValueType();
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(base::data_, &temp, sizeof(ValueType), gpuMemcpyHostToDevice));
        }

        __host__ explicit value(default_init_tag) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            if constexpr (!std::is_trivially_default_constructible_v<ValueType>)
            {
                auto temp = ValueType();
                CHECK_GPU_ERROR(detail::gpuMemcpy(base::data_, &temp, sizeof(ValueType), gpuMemcpyHostToDevice));
            }
        }

        __host__ explicit value(const ValueType& r) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(base::data_, &r, sizeof(ValueType), gpuMemcpyHostToDevice));
        }

        template <typename... Args>
        requires std::is_constructible_v<ValueType, Args...>
        __host__ explicit value(Args&&... args) : base(1)
        {
            auto temp = ValueType(std::forward<Args>(args)...);
            CHECK_GPU_ERROR(detail::gpuMalloc(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            CHECK_GPU_ERROR(detail::gpuMemcpy(base::data_, &temp, sizeof(ValueType), gpuMemcpyHostToDevice));
        }

        __device__ explicit value(base::pointer ptr) : base(ptr, 1) {}

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

        __host__ __device__ base::pointer get() noexcept { return base::data; }
        __host__ __device__ base::const_pointer get() const noexcept { return base::data; }
        __host__ __device__ explicit operator bool() const noexcept { return base::data != nullptr; }

#ifdef OVERLOAD_DEVICE
        __device__ base::const_reference operator*() const noexcept { return *base::data_; }
        __device__ base::reference operator*() noexcept { return *base::data_; }
        __device__ base::const_pointer operator->() const noexcept { return base::data_; }
        __device__ base::pointer operator->() noexcept { return base::data_; }
#endif
#ifdef OVERLOAD_HOST
        __host__ element_type operator*() const
        {
            ValueType temp;
            CHECK_GPU_ERROR(detail::gpuMemcpy(&temp, base::data_, sizeof(ValueType), gpuMemcpyDeviceToHost));
            return temp;
        }
        __host__ auto operator->() const
        {
            // proxy object
            struct
            {
                ValueType t;
                inline ValueType* operator->() { return &t; }
            } proxy;

            CHECK_GPU_ERROR(detail::gpuMemcpy(&proxy.t, base::data_, sizeof(ValueType), gpuMemcpyDeviceToHost));
            return proxy;
        }
#endif
        __device__ void reset([[maybe_unused]] base::pointer ptr)
        {
#ifdef GPU_DEVICE_COMPILE
            assert(ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            base::data_ = ptr;
            base::size_ = 1;
#endif
        }
    };

    template <typename ValueType>
    class unified_value : protected detail::array_base<ValueType, true>
    {
        using base = detail::array_base<ValueType, true>;

    public:
        using element_type = ValueType;

        __host__ __device__ unified_value(const unified_value& r) : base(r) {};
        __host__ __device__ unified_value(unified_value&& r) noexcept : base(std::move(r)) {};

        __host__ unified_value() : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            std::ranges::uninitialized_value_construct_n(base::data_, 1);
        }

        __host__ explicit unified_value(default_init_tag) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            std::ranges::uninitialized_default_construct_n(base::data_, 1);
        }

        __host__ explicit unified_value(const ValueType& r) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            std::ranges::construct_at(base::data_, r);
        }

        __host__ explicit unified_value(ValueType&& r) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            std::ranges::construct_at(base::data_, std::move(r));
        }

        template <typename... Args>
        requires std::is_constructible_v<ValueType, Args...>
        __host__ explicit unified_value(Args&&... args) : base(1)
        {
            CHECK_GPU_ERROR(detail::gpuMallocManaged(reinterpret_cast<void**>(&this->data_), sizeof(ValueType)));
            assert(base::data_ != nullptr);
            std::ranges::construct_at(base::data_, std::forward<Args>(args)...);
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

        __host__ __device__ base::const_reference operator*() const noexcept { return *base::data_; }
        __host__ __device__ base::reference operator*() noexcept { return *base::data_; }
        __host__ __device__ base::const_pointer operator->() const noexcept { return base::data_; }
        __host__ __device__ base::pointer operator->() noexcept { return base::data_; }

        __host__ __device__ base::pointer get() noexcept { return base::data; }
        __host__ __device__ base::const_pointer get() const noexcept { return base::data; }
        __host__ __device__ explicit operator bool() const noexcept { return base::data != nullptr; }

        __host__ void prefetch(int device_id) const
        {
            assert(base::size_ != 0);
            CHECK_GPU_ERROR(detail::gpuMemPrefetchAsync(base::data_, sizeof(ValueType), device_id));
        }
        __host__ void prefetch_to_cpu() const { prefetch(gpuCpuDeviceId); }
        __host__ void prefetch_to_gpu() const { prefetch(base::device_id_); }

        __device__ void reset([[maybe_unused]] base::pointer ptr)
        requires std::is_trivially_copyable_v<ValueType>
        {
#ifdef GPU_DEVICE_COMPILE
            assert(ptr != nullptr);

            // no need to be freed on GPU
            base::init();

            // ref_count_ does not need to be set because it is not used on GPU
            base::data_ = ptr;
            base::size_ = 1;
#endif
        }
    };

}  // namespace gpu_ptr

#undef GPU_DEVICE_COMPILE
#undef OVERLOAD_DEVICE
#undef OVERLOAD_HOST
#undef SIGSEGV_DEPRECATED
