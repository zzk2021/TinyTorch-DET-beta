
#pragma once

#include <cudnn.h>
#include <unordered_map>
#include <mutex>

#include "TensorImpl.h"
namespace TinyTorch {

class TensorOpsCudnn : public TensorOperations {
 public:
  explicit TensorOpsCudnn(int32_t device = 0);
  ~TensorOpsCudnn() override;

  cudnnHandle_t getCudnnHandle() { return cudnn_handle_; }

  TensorImpl conv2d(const TensorImpl& input, const TensorImpl& weight,
                    const TensorImpl& bias, const Padding2D& padding,
                    const Stride2D& stride) override;

  TENSOR_OPS_DECLARE(, override)

 private:
  int32_t device_;
  cudnnHandle_t cudnn_handle_;
  size_t workspace_size_ = 0;
  void* workspace_ = nullptr;

  struct DescriptorCache {
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnTensorDescriptor_t bias_desc = nullptr;
  };

  std::unordered_map<size_t, DescriptorCache> descriptor_cache_;
  std::mutex cache_mutex_;

  DescriptorCache& getDescriptors(const TensorImpl& input,
                                  const TensorImpl& weight,
                                  const Padding2D& padding,
                                  const Stride2D& stride);

  size_t makeCacheKey(const TensorImpl& input, const TensorImpl& weight,
                      const Padding2D& padding, const Stride2D& stride) const;
};

TensorOpsCudnn::TensorOpsCudnn(int32_t device) : device_(device) {
  cudaSetDevice(device_);

  cudnnStatus_t status = cudnnCreate(&cudnn_handle_);
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error("cuDNN create handle failed");
  }

  workspace_size_ = 32 * 1024 * 1024;
  cudaMalloc(&workspace_, workspace_size_);
}

std::unique_ptr<TensorOpsCudnn::Descriptors> TensorOpsCudnn::createDescriptors(
    const TensorImpl& input, const TensorImpl& weight, const Padding2D& padding,
    const Stride2D& stride) {
  auto desc = std::make_unique<Descriptors>();
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->input_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc->filter_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->output_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&desc->conv_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc->bias_desc));

  const auto& in_shape = input.shape();
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      desc->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_shape[0],
      in_shape[1], in_shape[2], in_shape[3]));

  const auto& w_shape = weight.shape();
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(desc->filter_desc, CUDNN_DATA_FLOAT,
                                         CUDNN_TENSOR_NCHW, w_shape[0],
                                         w_shape[1], w_shape[2], w_shape[3]));

  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      desc->conv_desc, padding.h, padding.w,  // padding
      stride.h, stride.w,                     // stride
      1, 1,                                   // dilation (1x1)
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  int outN, outC, outH, outW;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
      desc->conv_desc, desc->input_desc, desc->filter_desc, &outN, &outC, &outH,
      &outW));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc->output_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, outN, outC, outH,
                                         outW));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc->bias_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, 1, outC, 1, 1));

  return desc;
}
}