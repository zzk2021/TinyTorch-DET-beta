/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Torch.h"
#include "test.h"
#include <filesystem>


using namespace TinyTorch;


#ifdef USE_OPENCV
TEST(TEST_TensorImpl, opencv_to_tensor) {
    cv::Mat image = cv::imread("../../doc/ChatGPT_LOGO_512.png", cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    auto a = TensorImpl(image);
    auto b = Tensor(std::move(a));
    auto vec = b.data().toList();

    EXPECT_TRUE(!vec.empty()) << "Vector is empty, but contents are: "
                                  << ::testing::PrintToString(vec);
}
#endif


TEST(TEST_TensorImpl, constructor_default) {
  TensorImpl x;

  EXPECT_TRUE(x.empty());
  EXPECT_TRUE(x.dim() == 0);
}


TEST(TEST_TensorImpl, basic_tril) {
  auto x = TensorImpl::tril(TensorImpl::ones({3, 3}));
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList(), ElementsAre(1, 0, 0, 1, 1, 0, 1, 1, 1));

  x = TensorImpl::tril(TensorImpl::ones({2, 3}));
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.toList(), ElementsAre(1, 0, 0, 1, 1, 0));

  x = TensorImpl::tril(TensorImpl::ones({3, 3}), 1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList(), ElementsAre(1, 1, 0, 1, 1, 1, 1, 1, 1));

  x = TensorImpl::tril(TensorImpl::ones({3, 3}), -1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList(), ElementsAre(0, 0, 0, 1, 0, 0, 1, 1, 0));
}

TEST(TEST_TensorImpl, basic_triu) {
  auto x = TensorImpl::triu(TensorImpl::ones({3, 3}));
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList(), ElementsAre(1, 1, 1, 0, 1, 1, 0, 0, 1));

  x = TensorImpl::triu(TensorImpl::ones({2, 3}));
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.toList(), ElementsAre(1, 1, 1, 0, 1, 1));

  x = TensorImpl::triu(TensorImpl::ones({3, 3}), 1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList(), ElementsAre(0, 1, 1, 0, 0, 1, 0, 0, 0));

  x = TensorImpl::triu(TensorImpl::ones({3, 3}), -1);
  EXPECT_THAT(x.shape(), ElementsAre(3, 3));
  EXPECT_THAT(x.toList(), ElementsAre(1, 1, 1, 1, 1, 1, 0, 1, 1));
}


TEST(TEST_TensorImpl, basic_vstack) {
   TensorImpl a({1, 2, 3});
   TensorImpl b({4, 5, 6});
   auto y = TensorImpl::vstack({a, b});
   EXPECT_THAT(y.shape(), ElementsAre(2, 3));
   EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6));

   a = TensorImpl(Array2d({{1}, {2}, {3}}));
   b = TensorImpl(Array2d({{4}, {5}, {6}}));
   y = TensorImpl::vstack({a, b});
   EXPECT_THAT(y.shape(), ElementsAre(6, 1));
   EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6));
 }

 TEST(TEST_TensorImpl, basic_hstack) {
   TensorImpl a({1, 2, 3});
   TensorImpl b({4, 5, 6});
   auto y = TensorImpl::hstack({a, b});
   EXPECT_THAT(y.shape(), ElementsAre(6));
   EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6));

   a = TensorImpl(Array2d({{1}, {2}, {3}}));
   b = TensorImpl(Array2d({{4}, {5}, {6}}));
   y = TensorImpl::hstack({a, b});
   EXPECT_THAT(y.shape(), ElementsAre(3, 2));
   EXPECT_THAT(y.toList(), ElementsAre(1, 4, 2, 5, 3, 6));
 }

 TEST(TEST_TensorImpl, basic_split) {
   TensorImpl x({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
   auto y = x.split(1, 0);
   EXPECT_TRUE(y.size() == 2);
   EXPECT_THAT(y[0].shape(), ElementsAre(1, 2, 3));
   EXPECT_THAT(y[0].toList(), ElementsAre(4, 2, 3, 1, 0, 3));
   EXPECT_THAT(y[1].shape(), ElementsAre(1, 2, 3));
   EXPECT_THAT(y[1].toList(), ElementsAre(4, 2, 3, 1, 0, 3));

   y = x.split(1, 1);
   EXPECT_TRUE(y.size() == 2);
   EXPECT_THAT(y[0].shape(), ElementsAre(2, 1, 3));
   EXPECT_THAT(y[0].toList(), ElementsAre(4, 2, 3, 4, 2, 3));
   EXPECT_THAT(y[1].shape(), ElementsAre(2, 1, 3));
   EXPECT_THAT(y[1].toList(), ElementsAre(1, 0, 3, 1, 0, 3));

   y = x.split(1, 2);
   EXPECT_TRUE(y.size() == 3);
   EXPECT_THAT(y[0].shape(), ElementsAre(2, 2, 1));
   EXPECT_THAT(y[0].toList(), ElementsAre(4, 1, 4, 1));
   EXPECT_THAT(y[1].shape(), ElementsAre(2, 2, 1));
   EXPECT_THAT(y[1].toList(), ElementsAre(2, 0, 2, 0));
   EXPECT_THAT(y[2].shape(), ElementsAre(2, 2, 1));
   EXPECT_THAT(y[2].toList(), ElementsAre(3, 3, 3, 3));
 }

TEST(TEST_TensorImpl, constructor_shape) {
  TensorImpl x = TensorImpl::shape({2, 3});

  EXPECT_FALSE(x.empty());

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_scalar) {
  TensorImpl x = TensorImpl::scalar(2);

  EXPECT_FALSE(x.empty());

  EXPECT_TRUE(x.dim() == 0);
  EXPECT_TRUE(x.numel() == 1);
  EXPECT_THAT(x.toList(), ElementsAre(2));
}

TEST(TEST_TensorImpl, constructor_ones) {
  TensorImpl x = TensorImpl::ones({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toList(), ElementsAre(1, 1, 1, 1, 1, 1));
}

TEST(TEST_TensorImpl, constructor_zeros) {
  TensorImpl x = TensorImpl::zeros({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
  EXPECT_THAT(x.toList(), ElementsAre(0, 0, 0, 0, 0, 0));
}

TEST(TEST_TensorImpl, constructor_rand) {
  TensorImpl x = TensorImpl::rand({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_randn) {
  TensorImpl x = TensorImpl::randn({2, 3});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_bernoulli) {
  TensorImpl x = TensorImpl::bernoulli({2, 3}, 0.5);

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(3, 1));
}

TEST(TEST_TensorImpl, constructor_1d) {
  TensorImpl x({1, 2, 3});

  EXPECT_TRUE(x.dim() == 1);
  EXPECT_TRUE(x.numel() == 3);
  EXPECT_THAT(x.shape(), ElementsAre(3));
  EXPECT_THAT(x.strides(), ElementsAre(1));
  EXPECT_THAT(x.toList(), ElementsAre(1, 2, 3));
}

TEST(TEST_TensorImpl, constructor_2d) {
  TensorImpl x({{1, 2}, {3, 4}, {5, 6}});

  EXPECT_TRUE(x.dim() == 2);
  EXPECT_TRUE(x.numel() == 6);
  EXPECT_THAT(x.shape(), ElementsAre(3, 2));
  EXPECT_THAT(x.strides(), ElementsAre(2, 1));
  EXPECT_THAT(x.toList(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST(TEST_TensorImpl, constructor_3d) {
  TensorImpl x({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});

  EXPECT_TRUE(x.dim() == 3);
  EXPECT_TRUE(x.numel() == 12);
  EXPECT_THAT(x.shape(), ElementsAre(2, 2, 3));
  EXPECT_THAT(x.strides(), ElementsAre(6, 3, 1));
  EXPECT_THAT(x.toList(), ElementsAre(4, 2, 3, 1, 0, 3, 4, 2, 3, 1, 0, 3));
}

TEST(TEST_TensorImpl, basic_reshape) {
  TensorImpl x({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  auto y = TensorImpl::reshape(x, {2, 4});
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = TensorImpl::reshape(x, {2, -1});
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = TensorImpl::reshape(x, {-1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(4, 2));
}

TEST(TEST_TensorImpl, basic_flatten) {
  TensorImpl x({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
  auto y = TensorImpl::flatten(x);
  EXPECT_THAT(y.shape(), ElementsAre(8));

  y = TensorImpl::flatten(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));

  y = TensorImpl::flatten(x, 1, 2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 4));
}

TEST(TEST_TensorImpl, basic_unflatten) {
  auto y = TensorImpl::unflatten(TensorImpl::randn({3, 4, 1}), 1, {2, 2});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 2, 1));

  y = TensorImpl::unflatten(TensorImpl::randn({3, 4, 1}), 1, {-1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 2, 1));

  y = TensorImpl::unflatten(TensorImpl::randn({5, 12, 3}), -2, {2, 2, 3, 1, 1});
  EXPECT_THAT(y.shape(), ElementsAre(5, 2, 2, 3, 1, 1, 3));
}

TEST(TEST_TensorImpl, basic_squeeze) {
  auto x = TensorImpl::zeros({2, 1, 2, 1, 2});
  auto y = TensorImpl::squeeze(x);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
  y = TensorImpl::squeeze(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 1, 2));
  y = TensorImpl::squeeze(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 1, 2));
  y = TensorImpl::squeeze(x, {1, 2, 3});
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
}

TEST(TEST_TensorImpl, basic_unsqueeze) {
  auto x = TensorImpl::zeros({2, 1, 2});
  auto y = TensorImpl::unsqueeze(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 1, 2));
  y = TensorImpl::unsqueeze(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 1, 2));
  y = TensorImpl::unsqueeze(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 1));
}

TEST(TEST_TensorImpl, basic_fill) {
  auto x = TensorImpl::randn({2, 3});

  x.fill_(-1);
  EXPECT_THAT(x.toList(), ElementsAre(-1, -1, -1, -1, -1, -1));

  x.fill_(2);
  EXPECT_THAT(x.toList(), ElementsAre(2, 2, 2, 2, 2, 2));
}

TEST(TEST_TensorImpl, basic_clamp) {
  TensorImpl x;

  x = TensorImpl({1, 2, 3});
  x.clampMin_(2.3f);
  EXPECT_THAT(x.toList(), ElementsAre(2.3, 2.3, 3));

  x = TensorImpl({1, 2, 3});
  x.clampMax_(2.2f);
  EXPECT_THAT(x.toList(), ElementsAre(1, 2, 2.2));

  x = TensorImpl({1, 2, 3});
  x.clamp_(1.2f, 2.2f);
  EXPECT_THAT(x.toList(), ElementsAre(1.2, 2, 2.2));
}

TEST(TEST_TensorImpl, basic_range) {
  auto t = TensorImpl::arange(3, 10, 2);
  EXPECT_THAT(t.shape(), ElementsAre(4));
  EXPECT_THAT(t.toList(), ElementsAre(3, 5, 7, 9));

  t = TensorImpl::linspace(3, 10, 5);
  EXPECT_THAT(t.shape(), ElementsAre(5));
  EXPECT_THAT(t.toList(), ElementsAre(3, 4.75, 6.5, 8.25, 10));
}

TEST(TEST_TensorImpl, basic_indexing) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});

  auto y = x.index(0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3));

  y = x.index(1);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList(), ElementsAre(4, 5, 6));

  y = x.index(0, 0);
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList(), ElementsAre(1));

  y = x.index(1, 1);
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList(), ElementsAre(5));

  auto idx = TensorImpl({-1, 0});
  y = x.index({idx});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(7, 8, 9, 1, 2, 3));

  idx = TensorImpl(Array1d{1});
  y = x.index({idx});
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toList(), ElementsAre(4, 5, 6));

  idx = TensorImpl({0, 1});
  y = x.index({idx});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6));

  auto idx1 = TensorImpl({0, 1});
  auto idx2 = TensorImpl({2, 1});
  y = x.index({idx1, idx2});
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(3, 5));

  idx1 = TensorImpl({-1, 1});
  idx2 = TensorImpl({2, -1});
  y = x.index({idx1, idx2});
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(9, 6));

  x.indexPut_(std::vector<int32_t>{1}, -1);
  EXPECT_THAT(x.toList(), ElementsAre(1, 2, 3, -1, -1, -1, 7, 8, 9));

  x.indexPut_(std::vector<int32_t>{1}, TensorImpl({4, 5, 6}));
  EXPECT_THAT(x.toList(), ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9));

  idx1 = TensorImpl({-1, 1});
  idx2 = TensorImpl({2, -1});
  x.indexPut_({idx1, idx2}, -1);
  EXPECT_THAT(x.toList(), ElementsAre(1, 2, 3, 4, 5, -1, 7, 8, -1));

  idx1 = TensorImpl({-1, 1});
  idx2 = TensorImpl({2, -1});
  x.indexPut_({idx1, idx2}, TensorImpl({1.2, 2.3}));
  EXPECT_THAT(x.toList(), ElementsAre(1, 2, 3, 4, 5, 2.3, 7, 8, 1.2));
}

TEST(TEST_TensorImpl, basic_transpose) {
  auto x = TensorImpl(Array3d{{{1, 2, 3}, {4, 5, 6}}});
  auto y = x.transpose(0, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = x.transpose(1, 2);
  EXPECT_THAT(y.shape(), ElementsAre(1, 3, 2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = x.transpose(0, 2);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 1));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4, 2, 5, 3, 6));
}

TEST(TEST_TensorImpl, basic_t) {
  auto x = TensorImpl(Array2d{{1, 2, 3}, {4, 5, 6}});
  auto y = x.t();
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4, 2, 5, 3, 6));

  x = TensorImpl(Array2d{{1, 2}, {3, 4}, {5, 6}});
  y = x.t();
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 3, 5, 2, 4, 6));
}

TEST(TEST_TensorImpl, basic_permute) {
  TensorImpl x({1, 2, 3});
  auto y = x.permute();
  EXPECT_TRUE(y.shape() == x.shape());
  EXPECT_TRUE(y.toList() == x.toList());

  x = TensorImpl({{1, 2}, {3, 4}, {5, 6}});
  y = x.permute();
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 3, 5, 2, 4, 6));

  x = TensorImpl(Array3d{{{1, 2, 3}, {4, 5, 6}}});
  y = x.permute();
  EXPECT_THAT(y.shape(), ElementsAre(3, 2, 1));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = x.permute({1, 0, 2});
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6));

  x = TensorImpl::arange(0, 8);
  x.reshape_({1, 2, 2, 2});
  y = x.permute({0, 3, 1, 2});
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(0, 2, 4, 6, 1, 3, 5, 7));

  y = x.permute();
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2, 1));
  EXPECT_THAT(y.toList(), ElementsAre(0, 4, 2, 6, 1, 5, 3, 7));
}

TEST(TEST_TensorImpl, basic_stack) {
  TensorImpl a({1, 2, 3});
  TensorImpl b({4, 5, 6});
  TensorImpl c({7, 8, 9});

  auto y = TensorImpl::stack({a, b});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6));

  y = TensorImpl::stack({a, b}, -1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = TensorImpl::stack({a, b}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4, 2, 5, 3, 6));

  y = TensorImpl::stack({a, b, c}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(3, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4, 7, 2, 5, 8, 3, 6, 9));

  TensorImpl t1({{1, 2}, {3, 4}});
  TensorImpl t2({{5, 6}, {7, 8}});

  y = TensorImpl::stack({t1, t2});
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));

  y = TensorImpl::stack({t1, t2}, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 5, 6, 3, 4, 7, 8));
}

TEST(TEST_TensorImpl, math_compare) {
  TensorImpl x({{1, 2, 3}, {3, 4, 5}});
  TensorImpl x2({{3, 4, 5}, {1, 2, 3}});
  auto y = x > 2;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(0, 0, 1, 1, 1, 1));

  y = x < 3;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 1, 0, 0, 0, 0));

  y = x < x2;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 1, 1, 0, 0, 0));

  y = x == TensorImpl({{1, 1, 3}, {1, 4, 5}});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 0, 1, 0, 1, 1));

  y = x != TensorImpl({{1, 1, 3}, {1, 4, 5}});
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(0, 1, 0, 1, 0, 0));

  y = TensorImpl::maximum(x, x2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(3, 4, 5, 3, 4, 5));

  y = TensorImpl::minimum(x, x2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3, 1, 2, 3));
}

TEST(TEST_TensorImpl, math_scalar) {
  TensorImpl x({{1, 2}, {3, 4}});

  // add
  TensorImpl y = 2 + x + 1.5;
  EXPECT_THAT(y.toList(), ElementsAre(4.5, 5.5, 6.5, 7.5));
  y += 0.5;
  EXPECT_THAT(y.toList(), ElementsAre(5, 6, 7, 8));

  // sub
  y = 2 - x - 1.5;
  EXPECT_THAT(y.toList(), ElementsAre(-0.5, -1.5, -2.5, -3.5));
  y -= 0.5;
  EXPECT_THAT(y.toList(), ElementsAre(-1, -2, -3, -4));

  // mul
  y = 2 * x * 1.5;
  EXPECT_THAT(y.toList(), ElementsAre(3, 6, 9, 12));
  y *= 2;
  EXPECT_THAT(y.toList(), ElementsAre(6, 12, 18, 24));

  // div
  y = 12 / x / 2;
  EXPECT_THAT(y.toList(), ElementsAre(6, 3, 2, 1.5));
  y /= 0.5;
  EXPECT_THAT(y.toList(), ElementsAre(12, 6, 4, 3));
}

TEST(TEST_TensorImpl, math_same_shape) {
  TensorImpl x1({{1, 2}, {3, 4}});
  TensorImpl x2({{2, 3}, {4, 5}});

  auto y = x1 + x2;
  EXPECT_THAT(y.toList(), ElementsAre(3, 5, 7, 9));
  y += x1;
  EXPECT_THAT(y.toList(), ElementsAre(4, 7, 10, 13));

  y = x1 - x2;
  EXPECT_THAT(y.toList(), ElementsAre(-1, -1, -1, -1));
  y -= x1;
  EXPECT_THAT(y.toList(), ElementsAre(-2, -3, -4, -5));

  y = x1 * x2;
  EXPECT_THAT(y.toList(), ElementsAre(2, 6, 12, 20));
  y *= x1;
  EXPECT_THAT(y.toList(), ElementsAre(2, 12, 36, 80));

  y = x1 / x2;
  EXPECT_THAT(y.toList(), ElementsAre(0.5, 2.f / 3, 0.75, 0.8));
  y /= x1;
  EXPECT_THAT(y.toList(), ElementsAre(0.5, 1.f / 3, 0.25, 0.2));

  x1 = TensorImpl::scalar(1.f);
  x2 = TensorImpl::scalar(2.f);
  y = x1 - x2;
  EXPECT_THAT(y.toList(), ElementsAre(-1));
}

TEST(TEST_TensorImpl, math_min) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::min(x).item() == 1);

  auto y = TensorImpl::min(x, 0).first;
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3));

  y = TensorImpl::min(x, 0, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2, 3));

  y = TensorImpl::min(x, 1).first;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4));

  y = TensorImpl::min(x, 1, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4));

  y = TensorImpl::min(x, -1).first;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4));

  y = TensorImpl::min(x, -1, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList(), ElementsAre(1, 4));
}

TEST(TEST_TensorImpl, math_max_01) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::max(x).item() == 6);

  auto y = TensorImpl::max(x, 0).first;
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList(), ElementsAre(4, 5, 6));

  y = TensorImpl::max(x, 1).first;
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(3, 6));
}

TEST(TEST_TensorImpl, math_max_02) {
  auto x = TensorImpl::arange(0, 24, 1);
  x.reshape_({2, 3, 4});

  auto y = TensorImpl::max(x, 0, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(1, 3, 4));
  EXPECT_THAT(y.toList(),
              ElementsAre(12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));

  y = TensorImpl::max(x, 1, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 4));
  EXPECT_THAT(y.toList(), ElementsAre(8, 9, 10, 11, 20, 21, 22, 23));

  y = TensorImpl::max(x, 2, true).first;
  EXPECT_THAT(y.shape(), ElementsAre(2, 3, 1));
  EXPECT_THAT(y.toList(), ElementsAre(3, 7, 11, 15, 19, 23));
}

TEST(TEST_TensorImpl, math_meam) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::mean(x).item() == 3.5);

  auto y = TensorImpl::mean(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList(), ElementsAre(2.5, 3.5, 4.5));

  y = TensorImpl::mean(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(2, 5));

  y = TensorImpl::mean(x, {0, 1}, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 1));
  EXPECT_THAT(y.toList(), ElementsAre(3.5));

  y = TensorImpl::mean(x, {0, 1}, false);
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList(), ElementsAre(3.5));
}

TEST(TEST_TensorImpl, math_sum) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_TRUE(TensorImpl::sum(x).item() == 21);

  auto y = TensorImpl::sum(x, 0);
  EXPECT_THAT(y.shape(), ElementsAre(3));
  EXPECT_THAT(y.toList(), ElementsAre(5, 7, 9));

  y = TensorImpl::sum(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(6, 15));

  y = TensorImpl::sum(x, {0, 1}, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 1));
  EXPECT_THAT(y.toList(), ElementsAre(21));
  //
  x = TensorImpl({{{4, 2, 3}, {1, 0, 3}}, {{4, 2, 3}, {1, 0, 3}}});
  y = TensorImpl::sum(x, 2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(9, 4, 9, 4));

  y = TensorImpl::sum(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(5, 2, 6, 5, 2, 6));

  y = TensorImpl::sum(x, 0, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 3));
  EXPECT_THAT(y.toList(), ElementsAre(8, 4, 6, 2, 0, 6));
}

TEST(TEST_TensorImpl, math_var_01) {
  TensorImpl x({{1, 2, 3}, {4, 5, 6}});

  EXPECT_FLOAT_NEAR(TensorImpl::var(x, false).item(), 2.9166666);
  EXPECT_FLOAT_NEAR(TensorImpl::var(x).item(), 3.5);

  auto y = TensorImpl::var(x, (0), true, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 3));
  EXPECT_THAT(y.toList(), ElementsAre(4.5, 4.5, 4.5));

  y = TensorImpl::var(x, (1), true, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_FLOAT_VEC_NEAR(y.toList(), {1.0, 1.0});

  y = TensorImpl::var(x, {0, 1}, true, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 1));
  EXPECT_FLOAT_VEC_NEAR(y.toList(), {3.5});
}

TEST(TEST_TensorImpl, math_var_02) {
  TensorImpl x({3.14, 7.89, 1.23, 4.56, 9.01, 2.34, 5.67, 8.90,
                0.12, 6.78, 3.45, 7.12, 1.56, 4.89, 9.34, 2.67,
                5.89, 8.23, 0.45, 6.12, 3.78, 7.45, 1.89, 4.23,
                9.56, 2.12, 5.34, 8.67, 0.78, 6.45, 3.12, 7.78});
  x.reshape_({2, 2, 2, 4});

  auto y = TensorImpl::var(x, (0), true, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 2, 2, 4));
  EXPECT_FLOAT_VEC_NEAR(
      y.toList(),
      {3.7812, 0.0578, 0.3042, 1.2168, 13.6765, 13.0560, 7.1442, 10.9044,
       44.5568, 10.8578, 1.7861, 1.2013, 0.3042, 1.2168, 19.3442, 13.0561});

  y = TensorImpl::var(x, (1), true, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 2, 4));
  EXPECT_FLOAT_VEC_NEAR(
      y.toList(),
      {4.5602, 0.6160, 2.4642, 3.2768, 27.7513, 3.2512, 6.7345, 19.4064, 6.7345,
       18.6660, 11.9561, 3.2513, 4.5000, 0.5000, 0.7564, 6.3013});

  y = TensorImpl::var(x, {0, 1}, true, true);
  EXPECT_THAT(y.shape(), ElementsAre(1, 1, 2, 4));
  EXPECT_FLOAT_VEC_NEAR(y.toList(), {16.1479, 7.9826, 4.9094, 2.9820, 13.7604,
                                     4.9578, 10.8303, 8.5854});

  y = TensorImpl::var(x, {1, 2}, true, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1, 1, 4));
  EXPECT_FLOAT_VEC_NEAR(y.toList(), {15.2235, 5.9019, 11.9586, 7.5621, 13.6275,
                                     7.4389, 4.2882, 3.8282});
}

TEST(TEST_TensorImpl, math_argmin_01) {
  TensorImpl x({{4, 2, 3}, {1, 0, 3}});

  EXPECT_TRUE(TensorImpl::argmin(x).item() == 4);

  auto y = TensorImpl::argmin(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 1));

  y = TensorImpl::argmin(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList(), ElementsAre(1, 1));
}

TEST(TEST_TensorImpl, math_argmin_02) {
  TensorImpl x({3.14, 7.89, 1.23, 4.56, 9.01, 2.34, 5.67, 8.90,
                0.12, 6.78, 3.45, 7.12, 1.56, 4.89, 9.34, 2.67,
                5.89, 8.23, 0.45, 6.12, 3.78, 7.45, 1.89, 4.23,
                9.56, 2.12, 5.34, 8.67, 0.78, 6.45, 3.12, 7.78});
  x.reshape_({2, 2, 2, 4});
  auto y = TensorImpl::argmin(x, (2), true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2, 1, 4));
  EXPECT_THAT(y.toList(),
              ElementsAre(0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1));
}

TEST(TEST_TensorImpl, math_argmax_01) {
  TensorImpl x({{1, 2, 4}, {1, 0, 3}});

  EXPECT_TRUE(TensorImpl::argmax(x).item() == 2);

  auto y = TensorImpl::argmax(x, -1);
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(2, 2));

  y = TensorImpl::argmax(x, -1, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 1));
  EXPECT_THAT(y.toList(), ElementsAre(2, 2));
}

TEST(TEST_TensorImpl, math_argmax_02) {
  TensorImpl x({1,  2,  3,  5,  6,  7,  9,  10, 11, 2,  3,  4,
                6,  7,  8,  10, 11, 12, 5,  6,  7,  9,  10, 11,
                13, 14, 15, 6,  7,  8,  10, 11, 12, 14, 15, 16});
  x.reshape_({4, 9});

  auto y = TensorImpl::argmax(x, 1);
  EXPECT_THAT(y.shape(), ElementsAre(4));
  EXPECT_THAT(y.toList(), ElementsAre(8, 8, 8, 8));
}

TEST(TEST_TensorImpl, math_sin) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::sin(x);
  auto arr = y.toList();
  EXPECT_NEAR(arr[0], std::sin(1), 1e-4);
  EXPECT_NEAR(arr[1], std::sin(2), 1e-4);
  EXPECT_NEAR(arr[2], std::sin(3), 1e-4);
  EXPECT_NEAR(arr[3], std::sin(4), 1e-4);
}

TEST(TEST_TensorImpl, math_cos) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::cos(x);
  auto arr = y.toList();
  EXPECT_NEAR(arr[0], std::cos(1), 1e-4);
  EXPECT_NEAR(arr[1], std::cos(2), 1e-4);
  EXPECT_NEAR(arr[2], std::cos(3), 1e-4);
  EXPECT_NEAR(arr[3], std::cos(4), 1e-4);
}

TEST(TEST_TensorImpl, math_sqrt) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::sqrt(x);
  auto arr = y.toList();
  EXPECT_NEAR(arr[0], std::sqrt(1), 1e-4);
  EXPECT_NEAR(arr[1], std::sqrt(2), 1e-4);
  EXPECT_NEAR(arr[2], std::sqrt(3), 1e-4);
  EXPECT_NEAR(arr[3], std::sqrt(4), 1e-4);
}

TEST(TEST_TensorImpl, math_tanh) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::tanh(x);
  auto arr = y.toList();
  EXPECT_NEAR(arr[0], std::tanh(1), 1e-4);
  EXPECT_NEAR(arr[1], std::tanh(2), 1e-4);
  EXPECT_NEAR(arr[2], std::tanh(3), 1e-4);
  EXPECT_NEAR(arr[3], std::tanh(4), 1e-4);
}

TEST(TEST_TensorImpl, math_exp) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::exp(x);
  auto arr = y.toList();
  EXPECT_NEAR(arr[0], std::exp(1), 1e-4);
  EXPECT_NEAR(arr[1], std::exp(2), 1e-4);
  EXPECT_NEAR(arr[2], std::exp(3), 1e-4);
  EXPECT_NEAR(arr[3], std::exp(4), 1e-4);
}

TEST(TEST_TensorImpl, math_log) {
  TensorImpl x({{1, 2}, {3, 4}});
  auto y = TensorImpl::log(x);
  auto arr = y.toList();
  EXPECT_NEAR(arr[0], std::log(1), 1e-4);
  EXPECT_NEAR(arr[1], std::log(2), 1e-4);
  EXPECT_NEAR(arr[2], std::log(3), 1e-4);
  EXPECT_NEAR(arr[3], std::log(4), 1e-4);
}

TEST(TEST_TensorImpl, math_pow) {
  auto x1 = TensorImpl::arange(0, 6);
  auto y = TensorImpl::pow(x1, 3);
  EXPECT_THAT(y.toList(), ElementsAre(0, 1, 8, 27, 64, 125));

  auto x2 = TensorImpl({1.0, 2.0, 3.0, 3.0, 2.0, 1.0});
  y = TensorImpl::pow(x1, x2);
  EXPECT_THAT(y.toList(), ElementsAre(0, 1, 8, 27, 16, 5));

  x2 = TensorImpl({{1, 2, 3, 3, 2, 1}, {1, 2, 3, 3, 2, 3}});
  y = TensorImpl::pow(x1, x2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 6));
  EXPECT_THAT(y.toList(),
              ElementsAre(0, 1, 8, 27, 16, 5, 0, 1, 8, 27, 16, 125));
}

TEST(TEST_TensorImpl, math_dot) {
  Array1d d1 = {1, 2, 3};
  Array1d d2 = {4, 5, 6};
  auto y = TensorImpl::dot(TensorImpl(d1), TensorImpl(d2));
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList(), ElementsAre(32));

  y = TensorImpl::dot(TensorImpl(d1), TensorImpl(d1));
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList(), ElementsAre(14));
}

TEST(TEST_TensorImpl, math_matmul) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = TensorImpl::matmul(TensorImpl(d1), TensorImpl(d2));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(10, 13, 22, 29));

  Array2d d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d d4 = {{2, 3}, {4, 5}, {6, 7}};
  y = TensorImpl::matmul(TensorImpl(d3), TensorImpl(d4));
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(28, 34, 64, 79));

  Array2d d5 = {{1, 0}, {0, 1}};
  Array1d d6 = {1, 2};
  y = TensorImpl::matmul(TensorImpl(d5), TensorImpl(d6));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2));

  y = TensorImpl::matmul(TensorImpl(d6), TensorImpl(d5));
  EXPECT_THAT(y.shape(), ElementsAre(2));
  EXPECT_THAT(y.toList(), ElementsAre(1, 2));

  Array1d d7 = {2};
  y = TensorImpl::matmul(TensorImpl(d7), TensorImpl(d7));
  EXPECT_TRUE(y.dim() == 0);
  EXPECT_THAT(y.toList(), ElementsAre(4));

  // broadcast
  auto a = TensorImpl::arange(0, 2 * 2 * 4);
  a.reshape_({2, 2, 4});
  auto b = TensorImpl::arange(0, 2 * 2 * 4);
  b.reshape_({1, 2, 4, 2});
  auto c = TensorImpl::arange(0, 1 * 2 * 4);
  c.reshape_({1, 4, 2});
  auto d = TensorImpl::matmul(a, b);
  auto e = TensorImpl::matmul(a, c);

  EXPECT_THAT(d.shape(), ElementsAre(1, 2, 2, 2));
  EXPECT_THAT(d.toList(), ElementsAre(28, 34, 76, 98, 428, 466, 604, 658));

  EXPECT_THAT(e.shape(), ElementsAre(2, 2, 2));
  EXPECT_THAT(e.toList(), ElementsAre(28, 34, 76, 98, 124, 162, 172, 226));
}

TEST(TEST_TensorImpl, math_matmulTrans) {
  Array2d d1 = {{1, 2}, {3, 4}};
  Array2d d2 = {{2, 3}, {4, 5}};
  auto y = TensorImpl::matmulTrans(TensorImpl(d1), TensorImpl(d2), false, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(8, 14, 18, 32));

  Array2d d3 = {{1, 2, 3}, {4, 5, 6}};
  Array2d d4 = {{2, 4, 6}, {3, 5, 7}};
  y = TensorImpl::matmulTrans(TensorImpl(d3), TensorImpl(d4), false, true);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(28, 34, 64, 79));
}

TEST(TEST_TensorImpl, math_broadcast) {
  Array2d d1 = {{1, 2}};
  Array2d d2 = {{2, 3}, {4, 5}};
  Array2d d3 = {{2}, {4}};
  Array1d d4 = {1, 2};
  Array1d d5 = {1};

  auto y = TensorImpl(d1) + TensorImpl(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(3, 5, 5, 7));

  y = TensorImpl(d2) + TensorImpl(d3);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(4, 5, 8, 9));

  y = TensorImpl(d2) + TensorImpl(d4);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(3, 5, 5, 7));

  y = TensorImpl(d2) + TensorImpl(d5);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(3, 4, 5, 6));

  y = TensorImpl(d2) + TensorImpl::scalar(0.5);
  EXPECT_THAT(y.toList(), ElementsAre(2.5, 3.5, 4.5, 5.5));
}

TEST(TEST_TensorImpl, math_broadcast_inplace) {
  Array2d d1 = {{1, 2}};
  Array2d d2 = {{2, 3}, {4, 5}};
  Array2d d3 = {{2}, {4}};
  Array1d d4 = {1, 2};
  Array1d d5 = {1};

  auto y = TensorImpl(d1);
  y += TensorImpl(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(3, 5, 5, 7));

  y = TensorImpl(d3);
  y -= TensorImpl(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(0, -1, 0, -1));

  y = TensorImpl(d4);
  y *= TensorImpl(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(2, 6, 4, 10));

  y = TensorImpl(d5);
  y /= TensorImpl(d2);
  EXPECT_THAT(y.shape(), ElementsAre(2, 2));
  EXPECT_THAT(y.toList(), ElementsAre(0.5, 1.0 / 3.0, 0.25, 0.2));

  y = TensorImpl(d2);
  y += TensorImpl::scalar(0.5);
  EXPECT_THAT(y.toList(), ElementsAre(2.5, 3.5, 4.5, 5.5));
}

TEST(TEST_TensorImpl, basic_im2col_col2im) {
  auto input = TensorImpl(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});
  input.reshape_({1, 1, 4, 4});
  auto col = input.im2col(2, 2, 0);
  EXPECT_THAT(col.shape(), ElementsAre(4, 4));
  EXPECT_THAT(col.toList(), ElementsAre(1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14,
                                        11, 12, 15, 16));

  auto r = col.col2im(input.shape(), 2, 2, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_EQ(r.toList(), input.toList());

  col = input.im2col(2, 3, 0);
  EXPECT_THAT(col.shape(), ElementsAre(1, 4));
  EXPECT_THAT(col.toList(), ElementsAre(1, 2, 5, 6));

  r = col.col2im(input.shape(), 2, 3, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_THAT(r.toList(),
              ElementsAre(1, 2, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

  col = input.im2col(3, 2, 0);
  EXPECT_THAT(col.shape(), ElementsAre(1, 9));
  EXPECT_THAT(col.toList(), ElementsAre(1, 2, 3, 5, 6, 7, 9, 10, 11));

  r = col.col2im(input.shape(), 3, 2, 0);
  EXPECT_EQ(r.shape(), input.shape());
  EXPECT_THAT(r.toList(),
              ElementsAre(1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 0, 0, 0, 0));
}

TEST(TEST_TensorImpl, basic_im2col_col2im_1d) {
  {
    auto input = TensorImpl({1, 2, 3, 4});
    input.reshape_({1, 1, 4}); // [N=1, C=1, L=4]

    auto col = input.im2col1D(Size1D{2}, Size1D{2}, Size1D{0});

    EXPECT_THAT(col.shape(), ElementsAre(2, 2));
    EXPECT_THAT(col.toList(), ElementsAre(1, 2, 3, 4));

    auto r = col.col2im1D(input.shape(), Size1D{2}, Size1D{2}, Size1D{0});

    EXPECT_EQ(r.shape(), input.shape());
    EXPECT_EQ(r.toList(), input.toList());
  }
  {
    auto input = TensorImpl({1, 2, 3, 4});
    input.reshape_({1, 1, 4});
    auto col = input.im2col1D(Size1D{2}, Size1D{3}, Size1D{0});
    EXPECT_THAT(col.shape(), ElementsAre(1, 2));
    EXPECT_THAT(col.toList(), ElementsAre(1, 2));
    auto r = col.col2im1D(input.shape(), Size1D{2}, Size1D{3}, Size1D{0});
    EXPECT_EQ(r.shape(), input.shape());
    EXPECT_THAT(r.toList(), ElementsAre(1, 2, 0, 0));
  }

  {
    auto input = TensorImpl({1, 2, 3, 4});
    input.reshape_({1, 1, 4});

    auto col = input.im2col1D(Size1D{3}, Size1D{2}, Size1D{0});


    EXPECT_THAT(col.shape(), ElementsAre(1, 3));
    EXPECT_THAT(col.toList(), ElementsAre(1, 2, 3));
    auto r = col.col2im1D(input.shape(), Size1D{3}, Size1D{2}, Size1D{0});

    EXPECT_EQ(r.shape(), input.shape());
    EXPECT_THAT(r.toList(), ElementsAre(1, 2, 3, 0));
  }

  {
    auto input = TensorImpl({1, 2, 3, 4});
    input.reshape_({1, 1, 4});

    auto col = input.im2col1D(Size1D{2}, Size1D{1}, Size1D{1});

    EXPECT_THAT(col.shape(), ElementsAre(5, 2));
    EXPECT_THAT(col.toList(), ElementsAre(0,1, 1,2, 2,3, 3,4, 4,0));

    auto r = col.col2im1D(input.shape(), Size1D{2}, Size1D{1}, Size1D{1});

    EXPECT_EQ(r.shape(), input.shape());
    EXPECT_THAT(r.toList(), ElementsAre(2, 4, 6, 8));
  }
}