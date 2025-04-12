#include <cuda.h>
#include <cuda_runtime.h>
#include "TensorImpl_cpu.h"
#include "TensorImpl_cuda.cuh"
#include "flash_attention.inc"
#include <cmath>
namespace TinyTorch {

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

#define CURAND_CHECK(call)                                               \
  do {                                                                   \
    curandStatus_t err = call;                                           \
    if (err != CURAND_STATUS_SUCCESS) {                                  \
      std::cerr << "CURAND error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << curandGetErrorString(err) << " (" \
                << err << ")" << std::endl;                              \
      abort();                                                           \
    }                                                                    \
  } while (0)

#define CUBLAS_CHECK(call)                                               \
  do {                                                                   \
    cublasStatus_t err = call;                                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "CUBLAS error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << cublasGetErrorString(err) << " (" \
                << err << ")" << std::endl;                              \
      abort();                                                           \
    }                                                                    \
  } while (0)

#define CUDA_KERNEL_CHECK()                                                   \
  do {                                                                        \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA kernel error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << cudaGetErrorString(err) << " (" << err \
                << ")" << std::endl;                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

TensorImpl TensorOpsCUDA::flash_attention_(const TensorImpl& Q, const TensorImpl& K, const TensorImpl& V){

    const int Bc = 32; const int Br = 32;
    // 获取输入张量的维度
    const int B = Q.shape()[2]; const int nh = Q.shape()[1];
    const int N = Q.shape()[2]; const int d = Q.shape()[3];
    auto O = TensorImpl::zerosLike(Q);
    auto l = TensorImpl::zeros({B, nh, N});
    auto m = TensorImpl::zeros({B, nh, N});
    TensorImpl::fill(m, -INFINITY);
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / std::sqrt(d);
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
      int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    // 设置网格和块的维度
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // 每个块Bc个线程

    // 启动kernel
    flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_, K.data_, V.data_,
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_, m.data_, O.data_
    );
   return O;
}

}
