
#pragma once

#include <cudnn.h>
#include <unordered_map>
#include <mutex>

#include "TensorImpl.h"
namespace TinyTorch {
struct Padding2D{
  int h;
  int w;
};

struct Stride2D{
  int h;
  int w;
};

class TensorOpsCudnn : public TensorOperations {
 public:
  explicit TensorOpsCudnn(int32_t device = 0);
  ~TensorOpsCudnn() override;
  cudnnHandle_t getCudnnHandle();
  TensorImpl conv2d(const TensorImpl& input, const TensorImpl& weight,
                    const TensorImpl& bias, const Padding2D& padding,
                    const Stride2D& stride) override;


  TENSOR_OPS_DECLARE(, override)

 private:
  int32_t cudaDeviceIdx_;
  cudnnHandle_t cudnnHandle_ = nullptr;
  std::shared_ptr<Allocator> allocator_;
  void* workspace_ = nullptr;
  size_t workspaceSize_ = 0;
  struct Descriptors {
    cudnnTensorDescriptor_t input_desc = nullptr;
    cudnnFilterDescriptor_t filter_desc = nullptr;
    cudnnTensorDescriptor_t output_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnTensorDescriptor_t bias_desc = nullptr;
    ~Descriptors() {
      if (input_desc) cudnnDestroyTensorDescriptor(input_desc);
      if (filter_desc) cudnnDestroyFilterDescriptor(filter_desc);
      if (output_desc) cudnnDestroyTensorDescriptor(output_desc);
      if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
      if (bias_desc) cudnnDestroyTensorDescriptor(bias_desc);
    }
  };
  std::unique_ptr<Descriptors> createDescriptors(
      const TensorImpl& input,
      const TensorImpl& weight,
      const Padding2D& padding,
      const Stride2D& stride);
  };

}