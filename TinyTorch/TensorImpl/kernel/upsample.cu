#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include "../TensorImpl_cuda.cuh"
#include "upsample.inc"

#define kBlockSize 256

// this is an example of UpSample 2D with scala_factor = 2
// make sure all data on CUDA
namespace TinyTorch {
    TensorImpl TensorOpsCUDA::upsample_forward(const TensorImpl& a){
        int32_t N = a.numel();
        dim3 grid(N / kBlockSize, 1);
        dim3 block(kBlockSize, 1);
        auto O = TensorImpl::zerosLike(a, Device::CUDA);
        UpsampleNearest2D2XForward<<<grid, block>>>(N, a.data(), 1024, 1024, O.data());
        return O;
    }

    TensorImpl TensorOpsCUDA::upsample_backward(const TensorImpl& a){
        int32_t N = a.numel();
        dim3 grid(N / kBlockSize, 1);
        dim3 block(kBlockSize, 1);
        auto O = TensorImpl::zerosLike(a, Device::CUDA);
        UpsampleNearest2D2XBackward<<<grid, block>>>(N, a.data(), 1024, 1024, O.data());
        return O;
    }
}