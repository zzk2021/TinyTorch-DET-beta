

#include <Torch.h>

#include "test.h"

using namespace TinyTorch;

TEST(TEST_cuda_speed, fp32_gemm) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({16384,16384},Device::CUDA),true);
  Tensor b = Tensor(TensorImpl::randn({16384,16384},Device::CUDA),true);
  Tensor c = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);

  a.data().ops()->gemm(c.data().data(), a.data().data(), b.data().data(), 16384,16384,16384,false, true);

}

TEST(TEST_cuda_speed, fp16_gemm) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({16384,16384},Device::CUDA),true);
  Tensor b = Tensor(TensorImpl::randn({16384,16384},Device::CUDA),true);
  Tensor c = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  a.to(Dtype::float16);
  b.to(Dtype::float16);
  c.to(Dtype::float32);
  a.data().ops()->gemm(c.data().data(), a.data().data(), b.data().data(), 16384,16384,16384,false, true, a.data().type(), c.data().type());

}

TEST(TEST_cuda_speed, bf16_gemm) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({16384,16384},Device::CUDA),true);
  Tensor b = Tensor(TensorImpl::randn({16384,16384},Device::CUDA),true);
  Tensor c = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  a.to(Dtype::bfloat16);
  b.to(Dtype::bfloat16);
  c.to(Dtype::float32);
  a.data().ops()->gemm(c.data().data(), a.data().data(), b.data().data(), 16384,16384,16384,false, true, a.data().type(), c.data().type());
}
