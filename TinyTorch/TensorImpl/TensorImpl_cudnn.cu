#include <cudnn.h>

#include "TensorImpl_cuda.cuh"
#include "TensorImpl_cudnn.cuh"
namespace TinyTorch {

#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        const char* errstr = cudnnGetErrorString(status); \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, errstr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line "         \
                << __LINE__ << ": " << cudaGetErrorString(err) << " (" << err \
                << ")" << std::endl;                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

cudnnHandle_t TensorOpsCudnn::getCudnnHandle() {
  if (!cudnnHandle_) {
    CUDNN_CHECK(cudnnCreate(&cudnnHandle_));
  }
  return cudnnHandle_;
}

TensorOpsCudnn::TensorOpsCudnn(int32_t device)
    : cudaDeviceIdx_(device), allocator_(std::make_shared<AllocatorCUDA>()) {
  CUDA_CHECK(cudaSetDevice(0));
  workspaceSize_ = 32 * 1024 * 1024;
  allocator_->allocate(&workspace_, workspaceSize_);
}

TensorOpsCudnn::~TensorOpsCudnn() {
  if (cudnnHandle_) {
    cudnnDestroy(cudnnHandle_);
  }
  if (workspace_) {
    allocator_->deallocate(workspace_);
  }
}
TensorImpl TensorOpsCudnn::conv2d(const TensorImpl& input,
                                  const TensorImpl& weight,
                                  const TensorImpl& bias,
                                  const Padding2D& padding,
                                  const Stride2D& stride) {
  auto desc = createDescriptors(input, weight, padding, stride);
  int outN, outC, outH, outW;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
      desc->conv_desc, desc->input_desc, desc->filter_desc,
      &outN, &outC, &outH, &outW));
  Shape ret_shape = {static_cast<int32_t>(outN),static_cast<int32_t>(outC),
                     static_cast<int32_t>(outH), static_cast<int32_t>(outW)};
  TensorImpl output = TensorImpl::shape(ret_shape, input.device(), input.type());
  cudnnHandle_t handle = getCudnnHandle();
  cudnnConvolutionFwdAlgoPerf_t perf_results[10];
  int num_algos;
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
      handle,
      desc->input_desc, input.data(),
      desc->filter_desc, weight.data(),
      desc->conv_desc,
      desc->output_desc, output.data(),
      10, &num_algos, perf_results,
      workspace_, workspaceSize_));

  const auto best_algo = perf_results[0].algo;
  float alpha = 1.0f, beta = 0.0f;
  CUDNN_CHECK(cudnnConvolutionForward(
      handle, &alpha,
      desc->input_desc, input.data(),
      desc->filter_desc, weight.data(),
      desc->conv_desc, best_algo,
      workspace_, workspaceSize_,
      &beta, desc->output_desc, output.data()));

  if (!bias.empty()) {
    CUDNN_CHECK(cudnnAddTensor(
        handle, &alpha,
        desc->bias_desc, bias.data(),
        &alpha,
        desc->output_desc, output.data()));
  }

  return output;
}
}