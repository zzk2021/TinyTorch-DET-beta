#include <cudnn.h>
#include "Tensorlmpl_cudnn.cuh"
#define CUDNN_CHECK(call) do { \
    cudnnStatus_t status = (call); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        const char* errstr = cudnnGetErrorString(status); \
        fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, errstr); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


TensorOpsCudnn::TensorOpsCudnn(int32_t device)
    : device_(device), allocator_(std::make_shared<AllocatorCUDA>()) {

  // 设置当前设备
  CUDA_CHECK(cudaSetDevice(device_));

  // 初始化工作空间 (32MB)
  workspaceSize_ = 32 * 1024 * 1024;
  allocator_->allocate(&workspace_, workspaceSize_);
}

TensorOpsCudnn::~TensorOpsCudnn() {
  // 销毁 cuDNN 句柄
  if (cudnnHandle_) {
    cudnnDestroy(cudnnHandle_);
  }

  // 释放工作空间
  if (workspace_) {
    allocator_->deallocate(workspace_);
  }
}