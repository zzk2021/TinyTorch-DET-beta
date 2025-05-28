#include "Torch.h"

using namespace TinyTorch;

class BasicBlock : public nn::Module {
 public:

  BasicBlock(int32_t inplanes,std::vector<int32_t> planes):
                                                              conv1(inplanes, planes[0], 1, 1, 0, false),
                                                              bn1(planes[0]),
                                                              conv2(planes[0], planes[1], 3, 1, 1, false),
                                                              bn2(planes[1])
  {
    registerModules(
        {conv1,bn1,
         conv2,bn2 });
}
  std::string name() const override { return "BasicBlock"; }
  Tensor forward(Tensor &x) override {
    auto residual = x;
    x = conv1(x);
    x = bn1(x);
    x = Function::leakyrelu(x, 0.1);
    x = conv2(x);
    x = bn2(x);
    x = x + residual;
    return Function::leakyrelu(x,0.1);
  }
 private:
  nn::Conv2D conv1;
  nn::BatchNorm2D bn1;
  nn::Conv2D conv2;
  nn::BatchNorm2D bn2;

};

class DarkNet1 : public nn::Module {
 public:
  DarkNet1(const std::vector<int>& layers = {1, 2, 8, 8, 4}):
                                                              inplanes(32),
                                                              conv1(3, inplanes, 3, 1, 1, false),
                                                              bn1(inplanes)
  {
    layers_out_filters = {64, 128, 256, 512, 1024};
    layer1 = _make_layer({inplanes, layers_out_filters[0]}, layers[0]);
    layer2 = _make_layer({layers_out_filters[0], layers_out_filters[1]}, layers[1]);
    layer3 = _make_layer({layers_out_filters[1], layers_out_filters[2]}, layers[2]);
    layer4 = _make_layer({layers_out_filters[2], layers_out_filters[3]}, layers[3]);
    layer5 = _make_layer({layers_out_filters[3], layers_out_filters[4]}, layers[4]);
    registerModules({conv1, bn1,layer1, layer2, layer3, layer4, layer5, linear});

  }

  Tensor forward(Tensor &x) override {
    x = conv1(x);
    x = bn1(x);
    x = Function::relu(x);
    x = layer1(x);


    x = layer2(x);


     x = layer3(x);

     x = layer4(x);
     x = layer5(x);
    x = Function::maxPool2d(x, 7).squeeze();
    x = linear(x);
    x = Function::logSoftmax(x, 1);
    return x;
  }
  std::string name() const override { return "DarkNet"; }
 private:
  int inplanes;
  std::vector<int> layers_out_filters;
  nn::Conv2D conv1;
  nn::BatchNorm2D bn1;
  nn::Sequential layer1;
  nn::Sequential layer2;
  nn::Sequential layer3;
  nn::Sequential layer4;
  nn::Sequential layer5;
  nn::Linear linear = nn::Linear(1024,10);
  nn::Sequential _make_layer(const std::vector<int>& planes, int blocks) {
    auto seq = nn::Sequential();
    seq.pushBack(
        nn::Conv2D(inplanes, planes[1], 3, 2, 1, false)
    );
    float gain = sqrt(2.0 / (1 + pow(0.1, 2)));
    seq.pushBack(
        nn::BatchNorm2D(planes[1])
    );
    seq.pushBack(
        nn::LeakyRelu(0.1)
    );
    inplanes = planes[1];
    for (int i = 0; i < blocks; ++i) {
      const std::shared_ptr<Module> blk = std::make_shared<BasicBlock>(inplanes, planes);
      seq.pushBack(blk);
    }
    return seq;
  }
 public:
  std::vector<int> get_layers_out_filters() { return layers_out_filters;}
};

// Training settings
struct TrainArgs1 {
  // input batch size for training (default: 64)
  static constexpr int32_t batchSize = 8;

  // input batch size for testing (default: 1000)
  static constexpr int32_t testBatchSize = 8;

  // number of epochs to train (default: 1)
  static constexpr int32_t epochs = 1;

  // learning rate (default: 1.0)
  static constexpr float lr = 0.01f;

  // Learning rate step gamma (default: 0.7)
  static constexpr float gamma = 0.7f;

  // disables CUDA training
  static constexpr bool noCuda = false;

  // quickly check a single pass
  static constexpr bool dryRun = false;

  // random seed (default: 1)
  static constexpr unsigned long seed = 1;

  // how many batches to wait before logging training status
  static constexpr int32_t logInterval = 1;

  // For Saving the current Model
  static constexpr bool saveModel = false;

  static inline const std::vector<int32_t> input_shape = {128, 128};

  inline static const std::string train_annotation_path = R"(E:\data\cifar10\train_annotations.txt)";

  inline static const std::string test_annotation_path = R"(E:\data\cifar10\test_annotations.txt)";
};

void train1(TrainArgs1 &args, nn::Module &model,  Device device,
           data::DataLoader &dataLoader, optim::Optimizer &optimizer,
           int32_t epoch) {
  model.train();
  Timer timer;
  timer.start();
  auto correct = 0;
  auto all_num =0;
  for (auto [batchIdx, batch] : dataLoader) {
    auto &data = batch[0].to(device);
    auto &target = batch[1].to(device);
    optimizer.zeroGrad();
    auto output = model(data);
    auto loss = Function::nllloss(output, target);
    auto pred = output.data().argmax(1, true);
    correct += (int32_t)(pred == target.data().view(pred.shape())).sum().item();
    all_num+=dataLoader.batchSize();
    loss.backward();
    optimizer.step();
    if (batchIdx % args.logInterval == 0) {
      timer.mark();
      auto currDataCnt = batchIdx * dataLoader.batchSize();
      auto totalDataCnt = dataLoader.dataset().size();
      auto elapsed = (float)timer.elapseMillis() / 1000.f;  // seconds
      LOGD("Train Epoch: %d [%d/%d (%.0f%%)] Loss: %.6f, Elapsed: %.2fs Accuracy: %d/%d (%.0f%%)", epoch,
           currDataCnt, totalDataCnt, 100.f * currDataCnt / (float)totalDataCnt,
           loss.item(), elapsed, correct, all_num, 100. * correct / (float)all_num);

      if (args.dryRun) {
        break;
      }
    }
  }
}

void test1(nn::Module &model, Device device, data::DataLoader &dataLoader) {
  model.eval();
  Timer timer;
  timer.start();
  auto testLoss = 0.f;
  auto correct = 0;
  withNoGrad {
    for (auto [batchIdx, batch] : dataLoader) {
      auto &data = batch[0].to(device);
      auto &target = batch[1].to(device);
      auto output = model(data);
      testLoss +=  Function::nllloss(output, target).item();
      auto pred = output.data().argmax(1, true);
      correct += (int32_t)(pred == target.data().view(pred.shape())).sum().item();
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


void demo_cifar10() {
  LOGD("demo_cifar10 ...");
  Timer timer;
  timer.start();

  TrainArgs1 args;
  manualSeed(args.seed);
  // the anchors
  // input shape
  const std::vector<int> input_shape = {416, 416};

  auto useCuda = (!args.noCuda) && Tensor::deviceAvailable(Device::CUDA);
  Device device = useCuda ? Device::CUDA : Device::CPU;
  LOGD("Train with device: %s", useCuda ? "CUDA" : "CPU");

  auto transform = std::make_shared<data::transforms::Compose>(
      data::transforms::ResizeToTensor(args.input_shape),
      data::transforms::Normalize(0.5f, 0.2f)
  );

  auto trainDataset = std::make_shared<data::DatasetCIFAR10>(
      args.train_annotation_path, data::DatasetCIFAR10::TRAIN, transform);

  auto testDataset = std::make_shared<data::DatasetCIFAR10>(
      args.test_annotation_path, data::DatasetCIFAR10::TEST, transform);

  if (trainDataset->size() == 0 || testDataset->size() == 0) {
    LOGE("Dataset invalid.");
    return;
  }

  auto trainDataloader = data::DataLoader(trainDataset, args.batchSize, true);
  auto testDataloader = data::DataLoader(testDataset, args.testBatchSize, true);
  auto model = DarkNet1();
  model.to(device);
  std::cout << model.getTopologyText() << std::endl;
  auto optimizer = optim::AdaDelta(model.parameters(), args.lr);
  auto scheduler = optim::lr_scheduler::StepLR(optimizer, 1, args.gamma);

  for (auto epoch = 1; epoch < args.epochs + 1; epoch++) {
    train1(args, model, device, trainDataloader, optimizer, epoch);
    test1(model, device, testDataloader);
    scheduler.step();
  }

  if (args.saveModel) {
    save(model, "mnist_cnn.model");
  }

  timer.mark();
  LOGD("Total Time cost: %lld ms", timer.elapseMillis());
}