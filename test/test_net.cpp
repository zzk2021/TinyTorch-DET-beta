

#include <Torch.h>
#include "test.h"
using namespace TinyTorch;

TEST(TEST_Net, mnist_net) {
class Net1 : public nn::Module {
    public:
        Net1() { registerModules({conv1, conv21, dropout1, dropout2, fc1, fc2}); }
        Tensor forward(Tensor &x) override {
            x = conv1(x);
        x = Function::relu(x);
        x = conv21(x);
        x = Function::maxPool2d(x, 2);
        x = dropout1(x);
        x = Tensor::flatten(x, 1);

        x = fc1(x);
        x = Function::relu(x);
        x = dropout2(x);
        x = fc2(x);
        x = Function::logSoftmax(x, 1);
        return x;
    }
        private:
        nn::Conv2D conv1{1, 32, 3, 1};
        nn::Conv2D conv21{32, 64, 3, 1};
        nn::Dropout dropout1{0.25};
        nn::Dropout dropout2{0.5};
        nn::Linear fc1{9216, 128};
        nn::Linear fc2{128, 10};
    };


auto model = Net1();
model.to(Device::CUDA);

auto a = Tensor::zeros({1,1,28,28});
a.to(Device::CUDA);

auto p = model(a).toList();
auto a1 = Tensor::zeros({1,1,28,28});
model.to(Dtype::float16);
a1.to(Device::CUDA);
a1.to(Dtype::float16);
auto p1 = model(a1).to(Dtype::float32).toList();

for (size_t i = 0; i < p.size(); ++i) {
   ASSERT_NEAR(p[i], p1[i],0.15);
}
}