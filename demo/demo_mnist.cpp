/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Torch.h"
#include "tools/tools.h"
using namespace TinyTorch;

class fp32Net :public nn::Module {
 public:
  fp32Net() { registerModules({fc2}); }
  Tensor forward(Tensor &x) override {
    return x;
  }
 private:
  nn::Linear fc2{128, 10};
};

// https://github.com/pytorch/examples/blob/main/mnist/main.py
class Net : public nn::Module {
 public:
  Net()
  {
    registerModules({conv1,conv21,fc1,fc2,dropout1,dropout2});
    this->to(Device::CUDA);
    this->to(Dtype::float16);
  }
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
    x = Function::changetype(x, Dtype::float32);
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

void train(json &args, nn::Module &model, Device device,
           data::DataLoader &dataLoader, optim::Optimizer &optimizer,
           int32_t epoch) {
  model.train();

  Timer timer;
  timer.start();
  const float loss_scale = 2.0f;
  for (auto [batchIdx, batch] : dataLoader) {
    auto &data = batch[0].to(device).to(Dtype::float16);
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
    auto &data = batch[0].to(device).to(Dtype::float16);
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
