#include <Torch.h>

#include "test.h"

using namespace TinyTorch;

TEST(TEST_cuda_benchmark, fp32_gemm) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({8192,8192},Device::CUDA),true);
  Tensor b = Tensor(TensorImpl::randn({8192,8192},Device::CUDA),true);
  Tensor c = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  a.data().ops()->gemm(c.data().data(), a.data().data(), b.data().data(), 8192,8192,8192,false, true);
}

TEST(TEST_cuda_benchmark, fp16_gemm) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({8192,8192},Device::CUDA),true);
  Tensor b = Tensor(TensorImpl::randn({8192,8192},Device::CUDA),true);
  Tensor c = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  a.to(Dtype::float16);
  b.to(Dtype::float16);
  c.to(Dtype::float32);
  a.data().ops()->gemm(c.data().data(), a.data().data(), b.data().data(), 8192,8192,8192,false, true, a.data().type(), c.data().type());

}

TEST(TEST_cuda_benchmark, bf16_gemm) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({8192,8192},Device::CUDA),true);
  Tensor b = Tensor(TensorImpl::randn({8192,8192},Device::CUDA),true);
  Tensor c = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  a.to(Dtype::bfloat16);
  b.to(Dtype::bfloat16);
  c.to(Dtype::float32);
  a.data().ops()->gemm(c.data().data(), a.data().data(), b.data().data(), 8192,8192,8192,false, true, a.data().type(), c.data().type());
}

TEST(TEST_cuda_benchmark, leakyrelu_cuda) {
  Tensor b = Tensor(TensorImpl::ones({8,32,416,416},Device::CUDA),true);
  auto y_cuda = Function::leakyrelu(b, 0.1);
  y_cuda.sum().backward();
}
