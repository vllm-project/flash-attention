#pragma once

#include <cuda_runtime.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <deque>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

extern "C" AOTITorchError aoti_torch_get_current_cuda_stream(
    int32_t device_index, void** ret_stream);

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

#define FLASH_TORCH_CHECK STD_TORCH_CHECK
