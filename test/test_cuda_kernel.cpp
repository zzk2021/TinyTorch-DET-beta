/*
 * TinyTorch
 * @author 	: zekunzhang46@gmail.com
 *
 */

#include <Torch.h>

#include "test.h"

using namespace TinyTorch;

TEST(TEST_cuda_kernel, func_upsample_cuda) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({1,3,16,16},Device::CUDA),true);
  auto y = Function::upsample(a, 2);
  y.backward(Tensor::onesLike(y));

  Tensor b = a.to(Device::CPU);
  auto y_cpu = Function::upsample(b, 2);
  y_cpu.backward(Tensor::onesLike(y_cpu));
  EXPECT_THAT(y.data().toList(), y.data().toList());
  EXPECT_THAT(a.getGrad().data().toList(), b.getGrad().data().toList());
}

// fp32 to fp16
TEST(TEST_cuda_kernel, fp32_to_fp16_cuda) {
  // test scale_factor = 2    device = CUDA
  Tensor a1 = Tensor(TensorImpl::randn({1,3,16,16},Device::CUDA),true);

  auto p = a1.data().toList();
  a1.to(Dtype::float16);
  a1.to(Dtype::float32);
  a1.to(Device::CPU);

  auto p1 = a1.data().toList();
  for (size_t i = 0; i < p.size(); ++i) {
        ASSERT_NEAR(p[i], p1[i], 1e-3);
  }
}

// fp32 to bf16
TEST(TEST_cuda_kernel, fp32_to_bf16_cuda) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({1,3,16,16},Device::CUDA),true);

  auto p = a.data().toList();
  a.to(Dtype::bfloat16);
  a.to(Device::CPU);

  auto p1 = a.data().toList();
  for (size_t i = 0; i < p.size(); ++i) {
        ASSERT_NEAR(p[i], p1[i], 1e-2);
  }
}

TEST(TEST_cuda_kernel, bf16gemm) {
  // test scale_factor = 2    device = CUDA
  Tensor a = Tensor(TensorImpl::randn({16,16},Device::CUDA),true);
  Tensor b = Tensor(TensorImpl::randn({16,16},Device::CUDA),true);
  Tensor c = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  Tensor c1 = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  Tensor c2 = Tensor(TensorImpl::zerosLike(a.data(), Device::CUDA),true);
  // void TensorOpsCUDA::gemm(float* c, const float* a, const float* b, int32_t m,
   //                      int32_t k, int32_t n, bool transA, bool transB) {
  a.data().ops()->gemm(c.data().data(), a.data().data(), b.data().data(), 16,16,16,false, true);
  c.to(Device::CPU);
  auto p = c.data().toList();
  a.to(Dtype::float16);
  b.to(Dtype::float16);
  c1.to(Dtype::float32);
  a.data().ops()->gemm(c1.data().data(), a.data().data(), b.data().data(), 16,16,16,false, true, a.data().type(), c1.data().type());
  c1.to(Device::CPU);
  auto p1 = c1.data().toList();
  for (size_t i = 0; i < p.size(); ++i) {
        printf("%f %f\n",p[i], p1[i]);
        ASSERT_NEAR(p[i], p1[i], 1e-2);
  }

}