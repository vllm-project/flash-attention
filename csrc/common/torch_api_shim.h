#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef TORCH_TARGET_VERSION

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream);

// Keep the canonical FA3 implementation's tensor and dtype spelling while
// routing both types through PyTorch's stable API.
namespace at {
using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
}  // namespace at

namespace c10 {
using DeviceIndex = torch::stable::accelerator::DeviceIndex;
}  // namespace c10

namespace flash::torch_api {

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using DeviceGuard = torch::stable::accelerator::DeviceGuard;

inline DeviceGuard device_guard(const Tensor& tensor) {
    return DeviceGuard(static_cast<torch::stable::accelerator::DeviceIndex>(
        tensor.get_device()));
}

inline cudaDeviceProp* current_device_properties() {
    static std::deque<std::once_flag> device_flags;
    static std::vector<cudaDeviceProp> device_properties;
    static const bool initialized [[maybe_unused]] = [] {
        int device_count;
        const cudaError_t error = cudaGetDeviceCount(&device_count);
        STD_TORCH_CHECK(
            error == cudaSuccess,
            "cudaGetDeviceCount failed: " +
                std::string(cudaGetErrorString(error)));
        device_flags.resize(device_count);
        device_properties.resize(device_count);
        return true;
    }();

    int device_index;
    const cudaError_t error = cudaGetDevice(&device_index);
    STD_TORCH_CHECK(
        error == cudaSuccess,
        "cudaGetDevice failed: " + std::string(cudaGetErrorString(error)));
    std::call_once(device_flags[device_index], [device_index] {
        cudaDeviceProp properties{};
        const cudaError_t property_error =
            cudaGetDeviceProperties(&properties, device_index);
        STD_TORCH_CHECK(
            property_error == cudaSuccess,
            "cudaGetDeviceProperties failed: " +
                std::string(cudaGetErrorString(property_error)));
        device_properties[device_index] = properties;
    });
    return &device_properties[device_index];
}

inline cudaStream_t current_stream() {
    const auto device_index =
        torch::stable::accelerator::getCurrentDeviceIndex();
    void* stream = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream));
    return static_cast<cudaStream_t>(stream);
}

template <typename T>
inline T* data_ptr(const Tensor& tensor) {
    return static_cast<T*>(tensor.data_ptr());
}

inline std::optional<const Tensor>& as_optional_const(
    std::optional<Tensor>& tensor) {
    return reinterpret_cast<std::optional<const Tensor>&>(tensor);
}

namespace detail {

template <typename T>
struct StableBoxedArgument {
    using argument_type = T;
    using value_type = std::remove_cv_t<std::remove_reference_t<T>>;
    using storage_type = value_type;

    static decltype(auto) convert(storage_type& value) {
        if constexpr (std::is_lvalue_reference_v<argument_type>) {
            return static_cast<argument_type>(value);
        } else {
            return value_type(value);
        }
    }
};

template <>
struct StableBoxedArgument<int> {
    using storage_type = int64_t;

    static int convert(storage_type value) {
        STD_TORCH_CHECK(
            value <= std::numeric_limits<int>::max() &&
                value >= std::numeric_limits<int>::min(),
            "int64_t value is out of range for int");
        return static_cast<int>(value);
    }
};

template <>
struct StableBoxedArgument<float> {
    using storage_type = double;

    static float convert(storage_type value) {
        STD_TORCH_CHECK(
            value <= std::numeric_limits<float>::max() &&
                value >= -std::numeric_limits<float>::max(),
            "double value is out of range for float");
        return static_cast<float>(value);
    }
};

template <>
struct StableBoxedArgument<std::optional<int>> {
    using storage_type = std::optional<int64_t>;

    static std::optional<int> convert(const storage_type& value) {
        if (!value.has_value()) {
            return std::nullopt;
        }
        return StableBoxedArgument<int>::convert(*value);
    }
};

template <>
struct StableBoxedArgument<std::optional<const Tensor>&> {
    using storage_type = std::optional<Tensor>;

    static std::optional<const Tensor>& convert(storage_type& value) {
        return as_optional_const(value);
    }
};

template <auto Function>
struct StableBoxer;

template <typename Return, typename... Args, Return (*Function)(Args...)>
struct StableBoxer<Function> {
    using Storage =
        std::tuple<typename StableBoxedArgument<Args>::storage_type...>;

    template <size_t... Indices>
    static Storage unbox(StableIValue* stack,
                         std::index_sequence<Indices...>) {
        return Storage{
            torch::stable::detail::to<
                typename StableBoxedArgument<Args>::storage_type>(
                stack[Indices])...};
    }

    template <size_t... Indices>
    static Return invoke(Storage& args, std::index_sequence<Indices...>) {
        return Function(
            StableBoxedArgument<Args>::convert(std::get<Indices>(args))...);
    }

    static void boxed(StableIValue* stack,
                      uint64_t num_args,
                      uint64_t num_outputs) {
        STD_TORCH_CHECK(
            num_args == sizeof...(Args),
            "Registered schema has ", num_args,
            " arguments, but the kernel has ", sizeof...(Args));
        STD_TORCH_CHECK(
            num_outputs == 1,
            "Registered schema has ", num_outputs,
            " outputs, but the kernel has 1");
        auto args = unbox(stack, std::index_sequence_for<Args...>{});
        stack[0] = torch::stable::detail::from(
            invoke(args, std::index_sequence_for<Args...>{}));
    }
};

}  // namespace detail

inline bool has_shape(const Tensor& tensor,
                      std::initializer_list<int64_t> expected) {
    if (tensor.dim() != static_cast<int64_t>(expected.size())) {
        return false;
    }
    size_t dimension = 0;
    for (const int64_t size : expected) {
        if (tensor.size(dimension++) != size) {
            return false;
        }
    }
    return true;
}

inline Tensor empty_like(const Tensor& tensor) {
    return torch::stable::empty_like(tensor);
}

inline Tensor empty(const Tensor& like,
                    std::initializer_list<int64_t> sizes,
                    ScalarType dtype) {
    return torch::stable::new_empty(
        like, std::vector<int64_t>(sizes), std::make_optional(dtype));
}

inline Tensor zeros(const Tensor& like,
                    std::initializer_list<int64_t> sizes,
                    ScalarType dtype) {
    return torch::stable::new_zeros(
        like, std::vector<int64_t>(sizes), std::make_optional(dtype));
}

inline void zero_(Tensor& tensor) {
    torch::stable::zero_(tensor);
}

inline void fill_(Tensor& tensor, double value) {
    torch::stable::fill_(tensor, value);
}

inline Tensor pad(const Tensor& tensor, std::initializer_list<int64_t> padding) {
    return torch::stable::pad(tensor, std::vector<int64_t>(padding));
}

inline Tensor narrow(Tensor tensor,
                     int64_t dimension,
                     int64_t start,
                     int64_t length) {
    return torch::stable::narrow(tensor, dimension, start, length);
}

inline Tensor transpose(const Tensor& tensor, int64_t dim0, int64_t dim1) {
    return torch::stable::transpose(tensor, dim0, dim1);
}

inline void zero_slice_(Tensor& tensor, int64_t start, int64_t length) {
    Tensor slice = flash::torch_api::narrow(tensor, 0, start, length);
    flash::torch_api::zero_(slice);
}

}  // namespace flash::torch_api

namespace at::cuda {

using CUDAGuard = flash::torch_api::DeviceGuard;

inline cudaDeviceProp* getCurrentDeviceProperties() {
    return flash::torch_api::current_device_properties();
}

class CUDAStream {
  public:
    explicit CUDAStream(cudaStream_t stream) : stream_(stream) {}

    cudaStream_t stream() const {
        return stream_;
    }

  private:
    cudaStream_t stream_;
};

inline CUDAStream getCurrentCUDAStream() {
    return CUDAStream(flash::torch_api::current_stream());
}

}  // namespace at::cuda

#ifndef TORCH_CHECK
#define TORCH_CHECK STD_TORCH_CHECK
#endif

#define FLASH_STABLE_TORCH_BOX(function) \
    flash::torch_api::detail::StableBoxer<&function>::boxed

#else

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/version.h>

namespace flash::torch_api {

using Tensor = at::Tensor;
using ScalarType = at::ScalarType;
using DeviceGuard = at::cuda::CUDAGuard;

inline DeviceGuard device_guard(const Tensor& tensor) {
    return DeviceGuard(static_cast<c10::DeviceIndex>(tensor.get_device()));
}

inline cudaDeviceProp* current_device_properties() {
    return at::cuda::getCurrentDeviceProperties();
}

inline cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

template <typename T>
inline T* data_ptr(const Tensor& tensor) {
    return tensor.data_ptr<T>();
}

inline bool has_shape(const Tensor& tensor,
                      std::initializer_list<int64_t> expected) {
    return tensor.sizes() == at::IntArrayRef(expected);
}

inline Tensor empty_like(const Tensor& tensor) {
    return torch::empty_like(tensor);
}

inline Tensor empty(const Tensor& like,
                    std::initializer_list<int64_t> sizes,
                    ScalarType dtype) {
    return torch::empty(sizes, like.options().dtype(dtype));
}

inline Tensor zeros(const Tensor& like,
                    std::initializer_list<int64_t> sizes,
                    ScalarType dtype) {
    return torch::zeros(sizes, like.options().dtype(dtype));
}

inline void zero_(Tensor& tensor) {
    tensor.zero_();
}

inline void fill_(Tensor& tensor, double value) {
    tensor.fill_(value);
}

inline Tensor pad(const Tensor& tensor, std::initializer_list<int64_t> padding) {
    return torch::nn::functional::pad(
        tensor, torch::nn::functional::PadFuncOptions(padding));
}

inline Tensor narrow(Tensor tensor,
                     int64_t dimension,
                     int64_t start,
                     int64_t length) {
    return tensor.narrow(dimension, start, length);
}

inline Tensor transpose(const Tensor& tensor, int64_t dim0, int64_t dim1) {
    return tensor.transpose(dim0, dim1);
}

inline void zero_slice_(Tensor& tensor, int64_t start, int64_t length) {
    tensor.narrow(0, start, length).zero_();
}

}  // namespace flash::torch_api

#endif
