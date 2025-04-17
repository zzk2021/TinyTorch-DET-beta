#include "TensorImpl_cuda.cuh"
#include "flash_attention.inc"

namespace TinyTorch {
TensorImpl TensorOpsCUDA::flash_attention_(const TensorImpl& Q, const TensorImpl& K, const TensorImpl& V, int32_t head) {

  const int Bc = 32; const int Br = 32;
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

  dim3 grid_dim(B, nh);  // batch_size x num_heads
  dim3 block_dim(Bc);

  flash_attention_kernel<<<grid_dim, block_dim, sram_size>>>(
      Q.data_, K.data_, V.data_,
      N, d, Tc, Tr, Bc, Br, softmax_scale,
      l.data_, m.data_, O.data_
  );
  return O;
}
}
