# TinyVision

<!-- 使用 p 标签与 align 属性 -->
<h3>
<p align="center">
  <strong>A deep learning framework for faster vision task</strong>
</p>
</h3>
<p align="center">
  <img src="doc/ChatGPT_LOGO_512.png" alt="Centered Image" width="512" height="512">
</p>

This project is a fork and extension of the [TinyTorch](https://github.com/keith2018/TinyTorch) by [keith2018](https://github.com/keith2018).

# TinyTorch

Tiny deep learning training framework implemented from scratch in C++ that follows PyTorch's API. For more details, see [Write a nn training framework from scratch](https://robot9.me/write-nn-framework-from-scratch-tinytorch/)

[![CMake Linux](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_linux.yml/badge.svg)](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_linux.yml)
[![CMake MacOS](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_macos.yml/badge.svg)](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_macos.yml)
[![CMake Windows](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_windows.yml/badge.svg)](https://github.com/keith2018/TinyTorch/actions/workflows/cmake_windows.yml)

## Components

- Module
  - Linear
  - Conv2D
  - BatchNorm2D
  - MaxPool2D
  - Dropout
  - Softmax
  - LogSoftmax
  - Relu
  - Sequential
  - UpSample
  - Concat
  - Split
- Loss
  - MSELoss
  - NLLLoss
  - BCELoss
  - BCELossWithSigmoid
- Optimizer
  - SGD
  - Adagrad
  - RMSprop
  - AdaDelta
  - Adam
  - AdamW
- Data
  - Dataset
  - DataLoader
  - Transform

## Automatic differentiation

![](doc/AD.png)

## FP16 BF16 support
```c++
#include "Torch.h"

using namespace TinyTorch;

class Net : public nn::Module {
 public:
  Net()
  {
    registerModules({conv1,conv21,dropout1,dropout2,fc1,dropout2,fc2});
    this->to(Device::CUDA);   // use .to(Device::CUDA) before use to(Dtype::float16)
    this->to(Dtype::float16); 
  }
  Tensor forward(Tensor &x) override {
    x = Function::changetype(x, Dtype::float16);  // use changetype function in Net
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
    x = Function::changetype(x, Dtype::float32);  // use changetype function in Net
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

```

## MNIST training demo:
```c++
#include "Torch.h"

using namespace TinyTorch;

// https://github.com/pytorch/examples/blob/main/mnist/main.py
class Net : public nn::Module {
 public:
  Net() { registerModules({conv1, conv2, dropout1, dropout2, fc1, fc2}); }

  Tensor forward(Tensor &x) override {
    x = conv1(x);
    x = Function::relu(x);
    x = conv2(x);
    x = Function::relu(x);
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
  nn::Conv2D conv2{32, 64, 3, 1};
  nn::Dropout dropout1{0.25};
  nn::Dropout dropout2{0.5};
  nn::Linear fc1{9216, 128};
  nn::Linear fc2{128, 10};
};

void train(json &args, nn::Module &model, Device device,
           data::DataLoader &dataLoader, optim::Optimizer &optimizer,
           int32_t epoch) {
  model.train();

  Timer timer;
  timer.start();
  const float loss_scale = 2.0f;
  for (auto [batchIdx, batch] : dataLoader) {
    auto &data = batch[0].to(device);//.to(Dtype::float16);
    auto &target = batch[1].to(device);
    optimizer.zeroGrad();
    Tensor output = model(data);
    auto loss = Function::nllloss(output, target);
    loss = loss * loss_scale;
    loss.backward();

    for (auto& p : model.parameters()) {
      if (p->isRequiresGrad()) {
        p->getGrad().data() = p->getGrad().data() / loss_scale;
      }
    }
    optimizer.step();

    if (batchIdx % args.at("logInterval").get<int>() == 0) {
      timer.mark();
      auto currDataCnt = batchIdx * dataLoader.batchSize();
      auto totalDataCnt = dataLoader.dataset().size();
      auto elapsed = (float)timer.elapseMillis() / 1000.f;  // seconds
      LOGD("Train Epoch: %d [%d/%d (%.0f%%)] Loss: %.6f, Elapsed: %.2fs", epoch,
           currDataCnt, totalDataCnt, 100.f * currDataCnt / (float)totalDataCnt,
           loss.item(), elapsed);
      if (args.at("dryRun")) {
        break;
      }
    }
  }
}

void test(nn::Module &model, Device device, data::DataLoader &dataLoader) {
  model.eval();
  Timer timer;
  timer.start();
  auto testLoss = 0.f;
  auto correct = 0;
  withNoGrad {
    for (auto [batchIdx, batch] : dataLoader) {
    auto &data = batch[0].to(device);//.to(Dtype::float16);
    auto &target = batch[1].to(device);
      auto output = model(data);
      testLoss += Function::nllloss(output, target, SUM).item();
      auto pred = output.data().argmax(1, true);
      correct +=
          (int32_t)(pred == target.data().view(pred.shape())).sum().item();
    }
  }
  auto total = dataLoader.dataset().size();
  testLoss /= (float)total;
  timer.mark();
  auto elapsed = (float)timer.elapseMillis() / 1000.f;  // seconds
  LOGD(
      "Test set: Average loss: %.4f, Accuracy: %d/%d (%.0f%%), Elapsed: "
      "%.2fs",
      testLoss, correct, total, 100. * correct / (float)total, elapsed);
}

void demo_mnist() {
  LOGD("demo_mnist ...");
  Timer timer;
  timer.start();
  auto workdir = currentPath();
  fs::path subsir = "..\\config\\mnist.json";
  auto args = loadConfig((workdir / subsir).string());
  manualSeed(args.at("seed"));
  auto useCuda = (!args.at("noCuda")) && Tensor::deviceAvailable(Device::CUDA);
  Device device = useCuda ? Device::CUDA : Device::CPU;
  LOGD("Train with device: %s", useCuda ? "CUDA" : "CPU");

  auto transform = std::make_shared<data::transforms::Compose>(
      data::transforms::Normalize(0.1307f, 0.3081f));

  auto dataDir = "./data/";
  auto trainDataset = std::make_shared<data::DatasetMNIST>(
      dataDir, data::DatasetMNIST::TRAIN, transform);
  auto testDataset = std::make_shared<data::DatasetMNIST>(
      dataDir, data::DatasetMNIST::TEST, transform);

  if (trainDataset->size() == 0 || testDataset->size() == 0) {
    LOGE("Dataset invalid.");
    return;
  }

  auto trainDataloader = data::DataLoader(trainDataset, args.at("batchSize"), true, false);
  auto testDataloader = data::DataLoader(testDataset, args.at("testBatchSize"), true, false);
  auto model = Net();
  auto optimizer = optim::AdaDelta(model.parameters(), args.at("lr"));
  auto scheduler = optim::lr_scheduler::StepLR(optimizer, 1, args.at("gamma"));

  for (auto epoch = 1; epoch < args.at("epochs").get<int>() + 1; epoch++) {
    train(args, model, device, trainDataloader, optimizer, epoch);
    test(model, device, testDataloader);
    scheduler.step();
  }

  if (args.at("saveModel")) {
    save(model, "mnist_cnn.model");
  }

  timer.mark();
  LOGD("Total Time cost: %lld ms", timer.elapseMillis());
}
```

In config/minst.json

```
{
  "batchSize": 64,
  "testBatchSize": 1000,
  "epochs": 3,
  "lr": 0.1,
  "gamma": 0.7,
  "noCuda": false,
  "dryRun": false,
  "seed": 1,
  "logInterval": 10,
  "saveModel": false
}
```

## Build
```bash
mkdir build
cmake -B ./build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build --config Release
```

## Demo
```bash
cd demo/bin
./TinyTorch_demo
```

## Test
```bash
cd build
ctest
```

## Dependencies
- `CUDA` (optional for nvidia CUDA support)
- `OpenBLAS` (optional for `gemm` on CPU mode) [https://github.com/OpenMathLib/OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)
- `OpenCV`   (optional for `process(cv::Mat& input)` to read picture by opencv) [https://opencv.org/releases]
- `nlohmann/json` (to read json config) [https://github.com/nlohmann/json.git]
- `cuDNN` (optional for speed up `Conv` and `BatchNorm`) 

## Acknowledgments

Special thanks to [keith2018] for creating the initial version of this project, and to all contributors who have helped improve it over time.

## License
This code is licensed under the MIT License (see [LICENSE](LICENSE)).
