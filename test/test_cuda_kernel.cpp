/*
* TinyTorch
* @author 	: zekunzhang46@gmail.com
*
*/

#include <Torch.h>

#include "test.h"
#include <chrono>

using namespace TinyTorch;

TEST(TEST_cuda_kernel, broadcast_mask_test) {
 Array3d a_data = {{
                       {1.0, -2.0, 3.0, -4.0},
                       {-5.0, 6.0, -7.0, 8.0},
                       {9.0, -10.0, 11.0, -12.0}
                   }, {
                       {-13.0, 14.0, -15.0, 16.0},
                       {17.0, -18.0, 19.0, -20.0},
                       {-21.0, 22.0, -23.0, 24.0}
                   }};
 Tensor a = Tensor(a_data, true).to(Device::CUDA);
 Array3d mask_raw = {
     {{1.0f}, {0.0f}, {1.0f}}
 };
 Tensor mask = Tensor(mask_raw, false).to(Device::CUDA);
 auto b_gpu = a[mask];

 auto expected_values = std::vector<float>({
     1.0, -2.0, 3.0, -4.0,
     9.0, -10.0, 11.0, -12.0,

     -13.0, 14.0, -15.0, 16.0,
     -21.0, 22.0, -23.0, 24.0
 });
 EXPECT_THAT(b_gpu.data().toList(), ElementsAreArray(expected_values));

 Tensor loss = b_gpu.sum();
 loss.backward();
 std::vector<float> expected_grad = {
     // batch 0
     1.0, 1.0, 1.0, 1.0,  // row 0
     0.0, 0.0, 0.0, 0.0,  // row 1
     1.0, 1.0, 1.0, 1.0,  // row 2

     // batch 1
     1.0, 1.0, 1.0, 1.0,  // row 0
     0.0, 0.0, 0.0, 0.0,  // row 1
     1.0, 1.0, 1.0, 1.0   // row 2
 };
 auto grad_list = a.getGrad().toList();
 EXPECT_THAT(grad_list, ElementsAreArray(expected_grad));
}

TEST(TEST_cuda_kernel, func_from_mask) {
 // test scale_factor = 2    device = CUDA
 Array4d ary = {{
                    {{1.,2.},{3.,4.}},
                    {{5.,-6.},{7.,8.}},
                    {{9.,10.},{-11.,12.}},
                    {{13.,14.},{15.,16.}}
                }, {
                    {{1.,-2.},{3.,4.}},
                    {{5.,6.},{-7.,8.}},
                    {{9.,10.},{11.,12.}},
                    {{-13.,14.},{-15.,16.}}
                }};
 Tensor a = Tensor(ary, true);
 Tensor mask = Tensor(a.data() > 0, false);
 auto b = a[mask];

 Tensor loss1 = b.sum();
 loss1.backward();
 EXPECT_THAT(b.data().toList(), ElementsAre(1,2,3,4,5,7,8,9,10,12,13,14,15,16,1,3,4,5,6,8,9,10,11,12,14,16));
 EXPECT_THAT(a.getGrad().toList(), ElementsAre(
                                       1, 1, 1, 1,
                                       1, 0, 1, 1,
                                       1, 1, 0, 1,
                                       1, 1, 1, 1,

                                       1, 0, 1, 1,
                                       1, 1, 0, 1,
                                       1, 1, 1, 1,
                                       0, 1, 0, 1));
 Tensor a_gpu = a.to(Device::CUDA);
 Tensor mask_gpu = mask.to(Device::CUDA);
 auto b_gpu = a_gpu[mask_gpu];

 Tensor loss = b_gpu.sum();
 loss.backward();

 EXPECT_THAT(b_gpu.data().toList(), ElementsAre(1,2,3,4,5,7,8,9,10,12,13,14,15,16,1,3,4,5,6,8,9,10,11,12,14,16));
 EXPECT_THAT(a_gpu.getGrad().toList(), ElementsAre(
                                           1, 1, 1, 1,
                                           1, 0, 1, 1,
                                           1, 1, 0, 1,
                                           1, 1, 1, 1,

                                           1, 0, 1, 1,
                                           1, 1, 0, 1,
                                           1, 1, 1, 1,
                                           0, 1, 0, 1));
}

TEST(TEST_cuda_kernel, func_from_slice) {
 // test scale_factor = 2    device = CUDA
 Array4d ary = {{
                    {{1.,2.},{3.,4.}},
                    {{5.,6.},{7.,8.}},
                    {{9.,10.},{11.,12.}},
                    {{13.,14.},{15.,16.}}
                }, {
                    {{1.,2.},{3.,4.}},
                    {{5.,6.},{7.,8.}},
                    {{9.,10.},{11.,12.}},
                    {{13.,14.},{15.,16.}}
                }};
 Tensor a = Tensor(ary, true);
 auto b = a[{{},{0, 2},{0, 2},{0, 1}}];
 EXPECT_THAT(b.data().toList(), ElementsAre(1., 3.,5.,7.,1.,3.,5.,7.));
 Tensor a_gpu = a.to(Device::CUDA);
 auto b_gpu = a_gpu[{{},{0, 2},{0, 2},{0, 1}}];
 auto c_gpu = b_gpu[{{},{},{0, 1},{}}];
 auto d_gpu = b_gpu[{{},{},{1, 2},{}}];
 Tensor loss = c_gpu.sum() + d_gpu.sum();
 loss.backward();

 EXPECT_THAT(c_gpu.data().toList(), ElementsAre(1., 5., 1., 5.));
 EXPECT_THAT(d_gpu.data().toList(), ElementsAre(3., 7., 3., 7.));
 EXPECT_THAT(a_gpu.getGrad().toList(),
             ElementsAre(1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

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
   ASSERT_NEAR(p[i], p1[i], 1e-2);
 }
}

TEST(TEST_cuda_kernel, conv_fp16) {
 Tensor a = Tensor(TensorImpl::randn({1,3,16,16},Device::CUDA),true);
 auto weight = Tensor(TensorImpl::randn({3, 3, 3, 3},Device::CUDA), true);
 auto bias = Tensor(TensorImpl::randn({3},Device::CUDA), true);
 auto output1 = Function::conv2d(a, weight, bias);

 auto p = output1.data().toList();
 auto output = Function::conv2d(a.to(Dtype::float16), weight.to(Dtype::float16), bias.to(Dtype::float16));
 output.to(Dtype::float32);
 auto p1 = output.data().toList();

 for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i], 1e-1);
 }
}

TEST(TEST_cuda_kernel, linear_fp16) {
 Tensor a = Tensor(TensorImpl::randn({1,3,16,16},Device::CUDA),true);
 auto weight = Tensor(TensorImpl::randn({16,16},Device::CUDA), true);
 auto bias = Tensor(TensorImpl::randn({16},Device::CUDA), true);
 auto output1 = Function::linear(a, weight, bias);
 //  output1.backward(Tensor::shape(output1.shape()).to(Device::CUDA));
 auto p = output1.data().toList();
 auto output = Function::linear(a.to(Dtype::float16), weight.to(Dtype::float16), bias.to(Dtype::float16));
 //output.backward(Tensor::onesLike(output).to(Device::CUDA).to(Dtype::float16));
 output.to(Dtype::float32);
 auto p1 = output.data().toList();
 auto bp1 = output1.getGrad().data().toList();
 auto bp2 = output.getGrad().data().toList();
 for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i], 1e-1);
 }
 for (size_t i = 0; i < bp1.size(); ++i) {
   ASSERT_NEAR(bp1[i], bp2[i], 1e-1);
 }
}

TEST(TEST_cuda_kernel, max_pool2d_fp16) {
 Tensor a = Tensor(TensorImpl::randn({1,3,16,16},Device::CUDA),true);
 auto output1 = Function::maxPool2d(a, 2, 2);

 auto p = output1.data().toList();
 auto output = Function::maxPool2d(a.to(Dtype::float16), 2, 2);
 output.to(Dtype::float32);
 auto p1 = output.data().toList();
 for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i], 1e-2);
 }
}

TEST(TEST_cuda_kernel, relu_fp16) {
 Tensor a = Tensor(TensorImpl::randn({1,3,16,16},Device::CUDA),true);
 auto output1 = Function::relu(a);
 auto p = output1.data().toList();
 auto output = Function::relu(a.to(Dtype::float16));
 output.to(Dtype::float32);
 auto p1 = output.data().toList();
 for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i], 1e-2);
 }
}

TEST(TEST_cuda_kernel, logSoftmax_fp16) {
 auto o = TensorImpl::ones({1,3,16,16});
 Tensor a = Tensor(TensorImpl::ones({1,3,16,16},Device::CUDA),true);
 auto output1 = Function::logSoftmax(a, -1);
 auto p = output1.data().toList();
 auto l =  Tensor(TensorImpl::ones({1,3,16,16},Device::CUDA),true).to(Dtype::float16);
 auto output = Function::logSoftmax(l, -1);
 output.to(Dtype::float32);
 auto p1 = output.data().toList();
 for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i], 1e-3);
 }
}

TEST(TEST_cuda_kernel, concat_cuda) {
 Tensor x({
              {1.0f, 2.0f, 3.0f},
              {2.0f,2.5f,1.0f},
              {2.5f,3.0f,1.0f}}, true);

 Tensor a({
              {1.0f, 2.0f, 3.0f},
              {2.0f,2.5f,1.0f},
              {2.5f,3.0f,1.0f}}, true);

 a.to(Device::CUDA);
 x.to(Device::CUDA);
 Tensor grad({
                 {1.0f, 2.0f, 3.0f, 2.0f, 3.0f, 4.0f},
                 {2.0f,2.5f,1.0f,3.0f,3.5f,2.0f},
                 {2.5f,3.0f,1.0f,3.0f,3.5f,2.0f}}, true);
 auto y = Function::concat(x, a, 1);
 EXPECT_THAT(y.data().toList(), ElementsAre(1.0f, 2.0f, 3.0f,1.0f, 2.0f, 3.0f, 2.0f, 2.5f, 1.0f, 2.0f,2.5f,1.0f,2.5f,3.0f,1.0f,2.5f,3.0f,1.0f));
 y.backward(grad.to(Device::CUDA));
 EXPECT_THAT(x.getGrad().data().toList(), ElementsAre(1.0f, 2.0f, 3.0f, 2.0f, 2.5f, 1.0f, 2.5f, 3.0f, 1.0f));
 EXPECT_THAT(a.getGrad().data().toList(), ElementsAre(2.0f, 3.0f, 4.0f, 3.0f, 3.5f, 2.0f, 3.0f, 3.5f, 2.0f));

 Tensor x1(TensorImpl::randn({32,3,256,256}), true);
 Tensor a1(TensorImpl::randn({32,3,256,256}), true);
 Tensor grad1(TensorImpl::randn({32,6,256,256}), true);

 auto start = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 auto y1 = Function::concat(x1, a1, 1);
 auto end = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 auto startb = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 y1.backward(grad1);
 auto endb = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 Tensor a2 = a1.to(Device::CUDA);
 Tensor x2 = x1.to(Device::CUDA);

 Tensor grad2 = grad1.to(Device::CUDA);

 auto start1 = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 auto y2 = Function::concat(x2, a2, 1);
 auto end1 = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
 std::cout << "Concat operation forward cpu time (us): " << duration << std::endl;
 auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();
 std::cout << "Concat operation  forward cuda  time (us): " << duration1 << std::endl;

 auto startb1 = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 y2.backward(grad2);
 auto endb1 = std::chrono::high_resolution_clock::now(); // 开始记录 concat 操作时间
 auto durationd = std::chrono::duration_cast<std::chrono::microseconds>(endb - startb).count();
 auto durationd1 = std::chrono::duration_cast<std::chrono::microseconds>(endb1 - startb1).count();
 std::cout << "Concat operation backward cpu time (us): " << durationd << std::endl;
 std::cout << "Concat operation backward cuda  time (us): " << durationd1 << std::endl;

 auto p = y1.toList();
 auto p1 = y2.toList();

 auto pga1 = y1.getGrad().data().toList();

 auto pga2 = y2.getGrad().data().toList();

 for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i], 1e-3);
 }
 for (size_t i = 0; i < pga1.size(); ++i) {
   ASSERT_NEAR(pga1[i], pga2[i], 1e-3);
 }

}

TEST(TEST_Function, leakyrelu_cuda) {
 Tensor a = Tensor(TensorImpl::ones({32,3,256,256},Device::CPU),true);
 Tensor grad = Tensor(TensorImpl::randn({32,3,256,256},Device::CPU),true);

 auto y_cpu = Function::leakyrelu(a, 0.1);

 y_cpu.backward(grad);
 auto p = y_cpu.data().toList();
 auto gp = y_cpu.getGrad().data().toList();
 grad.to(Device::CUDA);

 Tensor b = Tensor(TensorImpl::ones({32,3,256,256},Device::CUDA),true);
 auto y_cuda = Function::leakyrelu(b, 0.1);
 y_cuda.backward(grad);
 auto p1 = y_cuda.data().toList();
 auto gp1 = y_cuda.getGrad().data().toList();

 for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i], 1e-3);
 }
 for (size_t i = 0; i < gp.size(); ++i) {
   ASSERT_NEAR(gp[i], gp1[i], 1e-3);
 }
}

TEST(TEST_cuda_kernel, func_basic_im2col_col2im_1d) {
 {
   auto input = TensorImpl({1, 2, 3, 4},Device::CUDA);
   input.reshape_({1, 1, 4}); // [N=1, C=1, L=4]

   auto col = input.im2col1D(Size1D{2}, Size1D{2}, Size1D{0});

   EXPECT_THAT(col.shape(), ElementsAre(2, 2));
   EXPECT_THAT(col.toList(), ElementsAre(1, 2, 3, 4));

   auto r = col.col2im1D(input.shape(), Size1D{2}, Size1D{2}, Size1D{0});

   EXPECT_EQ(r.shape(), input.shape());
   EXPECT_EQ(r.toList(), input.toList());
 }
 {
   auto input = TensorImpl({1, 2, 3, 4},Device::CUDA);
   input.reshape_({1, 1, 4});
   auto col = input.im2col1D(Size1D{2}, Size1D{3}, Size1D{0});
   EXPECT_THAT(col.shape(), ElementsAre(1, 2));
   EXPECT_THAT(col.toList(), ElementsAre(1, 2));
   auto r = col.col2im1D(input.shape(), Size1D{2}, Size1D{3}, Size1D{0});
   EXPECT_EQ(r.shape(), input.shape());
   EXPECT_THAT(r.toList(), ElementsAre(1, 2, 0, 0));
 }

 {
   auto input = TensorImpl({1, 2, 3, 4},Device::CUDA);
   input.reshape_({1, 1, 4});

   auto col = input.im2col1D(Size1D{3}, Size1D{2}, Size1D{0});


   EXPECT_THAT(col.shape(), ElementsAre(1, 3));
   EXPECT_THAT(col.toList(), ElementsAre(1, 2, 3));
   auto r = col.col2im1D(input.shape(), Size1D{3}, Size1D{2}, Size1D{0});

   EXPECT_EQ(r.shape(), input.shape());
   EXPECT_THAT(r.toList(), ElementsAre(1, 2, 3, 0));
 }

 {
   auto input = TensorImpl({1, 2, 3, 4},Device::CUDA);
   input.reshape_({1, 1, 4});

   auto col = input.im2col1D(Size1D{2}, Size1D{1}, Size1D{1});

   EXPECT_THAT(col.shape(), ElementsAre(5, 2));
   EXPECT_THAT(col.toList(), ElementsAre(0,1, 1,2, 2,3, 3,4, 4,0));

   auto r = col.col2im1D(input.shape(), Size1D{2}, Size1D{1}, Size1D{1});

   EXPECT_EQ(r.shape(), input.shape());
   EXPECT_THAT(r.toList(), ElementsAre(2, 4, 6, 8));
 }
}