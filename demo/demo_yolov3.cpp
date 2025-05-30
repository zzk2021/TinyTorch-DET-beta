#include <utility>

#include "Torch.h"
#include "Init.h"
#include "../TinyTorch/Objectdetection/Function_explore.h"
#include <mdspan>
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
      initialize_weights();
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
  void initialize_weights() {
        for (auto& module : subModules_) {
                float gain = sqrt(2.0 / (1 + pow(0.1, 2)));
                auto derivedPtr = dynamic_cast<nn::Conv2D*>(&module.get());
                if (derivedPtr) {
                    LOGD("init successful");
                    TinyTorch::nn::Init::kaimingUniform(derivedPtr->weights(), gain, TinyTorch::nn::FanMode::FAN_IN);
                }
        }
    }
};

class DarkNet : public nn::Module {
  public:
    DarkNet(const std::vector<int>& layers = {1, 2, 8, 8, 4}):
      inplanes(32),
      conv1(3, inplanes, 3, 1, 1, false),
      bn1(inplanes)
    {
        //registerModules({conv1, bn1});
        layers_out_filters = {64, 128, 256, 512, 1024};
        layer1 = _make_layer({inplanes, layers_out_filters[0]}, layers[0]);
        layer2 = _make_layer({layers_out_filters[0], layers_out_filters[1]}, layers[1]);
        layer3 = _make_layer({layers_out_filters[1], layers_out_filters[2]}, layers[2]);
        layer4 = _make_layer({layers_out_filters[2], layers_out_filters[3]}, layers[3]);
        layer5 = _make_layer({layers_out_filters[3], layers_out_filters[4]}, layers[4]);
        registerModules({conv1, bn1,layer1, layer2, layer3, layer4, layer5});
        initialize_weights();
    }
     std::string name() const override { return "DarkNet"; }
    std::vector<Tensor> forward(Tensor &x, bool l) override {
        x = conv1(x);
        x = bn1(x);
        x = Function::relu(x);
        x = layer1(x);
        x = layer2(x);
        Tensor out3 = layer3(x);
        Tensor out4 = layer4(out3);
        Tensor out5 = layer5(out4);
        return {out3, out4, out5};
    }

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

    void initialize_weights() {
        for (auto& module : subModules_) {
                float gain = sqrt(2.0 / (1 + pow(0.1, 2)));
                auto derivedPtr = dynamic_cast<nn::Conv2D*>(&module.get());
                if (derivedPtr) {
                    LOGD("init successful");
                    TinyTorch::nn::Init::kaimingUniform(derivedPtr->weights(), gain, TinyTorch::nn::FanMode::FAN_IN);
                }
        }
    }
  public:
     std::vector<int> get_layers_out_filters() { return layers_out_filters;}
};

class YoloBody : public nn::Module {
  public:
    YoloBody(std::vector<int32_t> anchors_mask_length, int32_t num_classes,DarkNet backbone, bool pretrained):
       num_classes_(num_classes),
       backbone_(backbone)
      {
         auto out_filters = backbone.get_layers_out_filters();
         last_layer0 = _make_last_layer({512, 1024}, out_filters[out_filters.size()-1], anchors_mask_length[0] * (num_classes + 5));
         last_layer1_conv   = conv2d(512, 256, 1);
         last_layer1 = _make_last_layer({256, 512}, out_filters[out_filters.size()-2] + 256, anchors_mask_length[1] * (num_classes + 5));
         last_layer2_conv   = conv2d(256, 128, 1);
         last_layer2 = _make_last_layer({128, 256}, out_filters[out_filters.size()-3] + 128, anchors_mask_length[2] * (num_classes + 5));
         registerModules({backbone_,last_layer0,last_layer1,last_layer1_conv,last_layer2_conv,last_layer2});
      }

    std::vector<Tensor> forward(Tensor &x, bool l) override {
            auto x_ = backbone_.forward(x, true);
            auto x2 =  x_[0];auto x1 =  x_[1];auto x0 =  x_[2];
            auto out0_branch = last_layer0[{0, 5}](x0);
            auto out0 = last_layer0[{5, static_cast<int>(last_layer0.getsize())}](out0_branch);
            auto x1_in = last_layer1_conv(out0_branch);
            x1_in = Function::upsample(x1_in,2);
            x1_in = Function::concat(x1_in, x1, 1);
            auto out1_branch = last_layer1[{0, 5}](x1_in);
            auto out1  = last_layer1[{5,static_cast<int>(last_layer1.getsize())}](out1_branch);
            auto x2_in = last_layer2_conv(out1_branch);
            x2_in = Function::upsample(x2_in,2);
            x2_in = Function::concat(x2_in, x2, 1);
            auto out2 = last_layer2(x2_in);

            return {out0, out1, out2};
        }

  private:
      int32_t num_classes_;
      DarkNet backbone_;
      nn::Sequential last_layer0;
      nn::Sequential last_layer1;
      nn::Sequential last_layer1_conv;
      nn::Sequential last_layer2;
      nn::Sequential last_layer2_conv;
      nn::Sequential conv2d(int32_t filter_in,int32_t filter_out,int32_t kernel_size){
          auto pad = (kernel_size - 1) / 2; // if kernel_size else 0
          auto seq = nn::Sequential();
          seq.pushBack(
            nn::Conv2D(filter_in, filter_out, kernel_size, 1, pad, false)
          );
          seq.pushBack(
             nn::BatchNorm2D(filter_out)
          );
          seq.pushBack(
             nn::LeakyRelu(0.1)
          );
          return seq;
      };
     nn::Sequential _make_last_layer(const std::vector<int>& filters_list, int in_filters,int out_filter) {
        auto seq = nn::Sequential();
        seq.pushBack(
            conv2d(in_filters, filters_list[0] , 1)
        );
        seq.pushBack(
            conv2d(filters_list[0], filters_list[1] , 3)
        );
        seq.pushBack(
            conv2d(filters_list[1], filters_list[0] , 1)
        );
        seq.pushBack(
            conv2d(filters_list[0], filters_list[1] , 3)
        );
        seq.pushBack(
            conv2d(filters_list[1], filters_list[0] , 1)
        );
        seq.pushBack(
            conv2d(filters_list[0], filters_list[1] , 3)
        );
        seq.pushBack(
            nn::Conv2D(filters_list[1], out_filter, 1, 1, 0, true)
        );
        return seq;
    };
};

class YoloLoss : public nn::Module {
    public:
       YoloLoss( int32_t num_classes, Shape input_shape, std::vector<std::vector<int32_t>> anchors):
            num_classes_(num_classes),
            box_ratio_(0.05),
            ignore_threshold_(0.5)
            {
               bbox_attrs_ = 5 + num_classes;
                cls_ratio_ = 1.f * ((float)num_classes / 80);
               input_shape_ = std::move(input_shape);
               anchors_ = std::move(anchors);
               obj_ratio_ = 5.f * ((float)input_shape_[0] * (float)input_shape_[1]) / (std::pow(416,2));
               anchors_mask_ = {{6,7,8}, {3,4,5}, {0,1,2}};

            }
        Tensor forward(std::vector<Tensor> &all) override{
           auto inputs = std::vector{all[0], all[1], all[2]};
           auto targets = all[3];
           Tensor loss  = Tensor::scalar(0, true, inputs[0].device(),inputs[0].type());
           for (int l=0;l < inputs.size();l+=1){
            Tensor input = inputs[l];
            auto bs      = input.shape()[0];
            auto in_h    = input.shape()[2];
            auto in_w    =  input.shape()[3];
            auto stride_h = input_shape_[0] / in_h;
            auto stride_w = input_shape_[1] / in_w;
            std::vector<std::vector<float>> scaled_anchors;
            for (const auto& anchor : anchors_) {
                float a_w = anchor[0];
                float a_h = anchor[1];
                float scaled_w = a_w / static_cast<float>(stride_w);
                float scaled_h = a_h / static_cast<float>(stride_h);
                scaled_anchors.push_back({scaled_w, scaled_h});
            }
            auto prediction = input.reshape({bs,
                                         static_cast<int>(anchors_mask_[l].size()),
                                         bbox_attrs_, in_h, in_w});
            auto x = prediction[{{},{},{0,1},{},{}}];
            auto y = prediction[{{},{},{1,2},{},{}}];
            auto w = prediction[{{},{},{2,3},{},{}}];
            auto h = prediction[{{},{},{3,4},{},{}}];
            auto conf = prediction[{{},{},{4,5},{},{}}];
            auto pred_cls = prediction[{{},{},{5,-1},{},{}}];
            auto [y_true, noobj_mask, box_loss_scale, obj_mask, n] = get_target_(l, targets, scaled_anchors, in_h, in_w);
            Tensor y_true_t = Tensor(y_true, false);
            Tensor box_loss_scale_t =  Tensor(box_loss_scale, false);
            Tensor obj_mask_t =  Tensor(obj_mask, false);
            Tensor noobj_mask_t =  Tensor(noobj_mask, false);
            box_loss_scale_t = 2 - box_loss_scale_t;
            box_loss_scale_t.to(input.device());
            obj_mask_t.to(input.device());
            y_true_t.to(input.device());
            noobj_mask_t.to(input.device());
            if (n != 0){
                auto loss_x  = (Function::bceLossWithSigmoid(x.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{0,1}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_y  = (Function::bceLossWithSigmoid(y.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{1,2}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_w  = (Function::mseLoss(w.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{2,3}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_h  = (Function::mseLoss(h.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{3,4}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_loc = (loss_x + loss_y + loss_h + loss_w) * 0.1;
                auto loss_cls = Function::bceLossWithSigmoid(Tensor(pred_cls.data().permute({0,1,3,4,2}), true)[obj_mask_t.unsqueeze(-1)],
                                                             y_true_t[{{},{},{},{},{5,-1}}][obj_mask_t.unsqueeze(-1)], TinyTorch::MEAN);
                loss  += loss_loc * box_ratio_ + loss_cls * cls_ratio_;
                LOGD("loss_loc: %f , loss_cls:%f",loss_loc.item(),loss_cls.item());
            }
            auto loss_conf   = (Function::bceLossWithSigmoid(conf.squeeze(), obj_mask_t, TinyTorch::NONE)[noobj_mask_t + obj_mask_t]).mean();
            LOGD("loss_conf: %f",loss_conf.item());
            if (loss_conf.toList().size() == 0 || std::isnan(loss_conf.item())) {
              LOGW("Warning: loss_conf is NaN, skipping this batch.");
            }
            else
              loss += loss_conf * obj_ratio_;
           }
           return loss;
        }
    private:
      int32_t num_classes_;
      Shape input_shape_;
      float box_ratio_ ;
      int bbox_attrs_ ;
      float obj_ratio_ ;
      float cls_ratio_ ;
      float ignore_threshold_;
      nn::MSELoss mseloss;
      std::vector<std::vector<int32_t>> anchors_;
      std::vector<std::vector<int32_t>> anchors_mask_ ;
      std::tuple<Array5d, Array4d, Array4d, Array4d, int> get_target_(int32_t l, Tensor targets, Array2d anchors, int32_t in_h, int32_t in_w){
        int bs               = targets.shape()[0];
        Array4d noobj_mask(bs,
            Array3d(
                static_cast<int>(anchors_mask_[l].size()),
                Array2d(in_h,Array1d(in_w, 1.0f))));

        Array4d box_loss_scale(bs,
            Array3d(
                static_cast<int>(anchors_mask_[l].size()),
                Array2d(in_h,Array1d(in_w, 0.0f))));

        std::vector<Array4d> y_true(bs,
            Array4d(
                static_cast<int>(anchors_mask_[l].size()),
                Array3d(in_h,Array2d (in_w,
                Array1d(bbox_attrs_,0.0f)))));

        Array4d obj_mask(bs,
            Array3d(
            static_cast<int>(anchors_mask_[l].size()),
            Array2d(in_h, Array1d(in_w, 0.0f))));
        int n = 0;
        for(int b=0;b<bs;b+=1){
          if (targets[{b,0,0}] == -1)
            continue;
          std::vector<std::vector<float>> batch_target;
          batch_target.reserve(targets.shape()[1]);
          for (int t = 0; t < targets.shape()[1]; ++t) {
            if (targets[{b,t,0}] == -1) {
              break;
            }
            std::vector<float> target_box(5, 0.0f);
            target_box[0] = targets[{b,t,1}] * in_w;
            target_box[2] = targets[{b,t,3}] * in_w;
            target_box[1] = targets[{b,t,2}] * in_h;
            target_box[3] = targets[{b,t,4}] * in_h;
            target_box[4] = targets[{b,t,0}];

            batch_target.push_back(target_box);
          }

          int M = batch_target.size();

          Array2d anchor_shapes(M, std::vector<float>(4, 0.0f));
          for (size_t i = 0; i < M; ++i) {
            anchor_shapes[i][2] = anchors[i][0]; // w
            anchor_shapes[i][3] = anchors[i][1]; // h
          }
          Array2d gt_box(M, std::vector<float>(4, 0.0f));
          for (size_t i = 0; i < M; ++i) {
            gt_box[i][2] = batch_target[i][2];
            gt_box[i][3] = batch_target[i][3];
          }

          auto best_ns = FindBestAnchors(CalculateIOU(gt_box, anchor_shapes));
          for (int t=0;t<best_ns.size();t+=1){
            auto best_n = best_ns[t];
            if (std::find(anchors_mask_[l].begin(), anchors_mask_[l].end(), best_n) == anchors_mask_[l].end()) {
                continue;
            }
            if (best_n < 0 || best_n >= static_cast<int>(anchors.size())) {
                continue;
            }

            int k = std::distance(anchors_mask_[l].begin(),
                          std::find(anchors_mask_[l].begin(), anchors_mask_[l].end(), best_n));
            int i = static_cast<int>(std::floor(batch_target[t][0]));
            int j = static_cast<int>(std::floor(batch_target[t][1]));
            int c = static_cast<int>(batch_target[t][4]);
            noobj_mask[b][k][j][i] = 0.0f;
            y_true[b][k][j][i][0] = batch_target[t][0] - static_cast<float>(i);
            y_true[b][k][j][i][1] = batch_target[t][1] - static_cast<float>(j);
            y_true[b][k][j][i][2] = std::log(batch_target[t][2] / anchors[best_n][0]); //batch_target[t][0] - static_cast<float>(i);
            y_true[b][k][j][i][3] = std::log(batch_target[t][3] / anchors[best_n][0]); //batch_target[t][0] - static_cast<float>(i);
            y_true[b][k][j][i][4] = 1;
            y_true[b][k][j][i][c+5] = 1;
            box_loss_scale[b][k][j][i] = batch_target[t][2] * batch_target[t][3] / in_w / in_h;
            obj_mask[b][k][j][i] = 1.0f;
            n+=1;
          }
        }
        return std::make_tuple(y_true, noobj_mask, box_loss_scale, obj_mask, n);
      }
};

// Training settings
struct TrainArgs {
  // input batch size for training (default: 64)
  static constexpr int32_t batchSize = 2;

  // input batch size for testing (default: 1000)
  static constexpr int32_t testBatchSize = 2;

  // number of epochs to train (default: 1)
  static constexpr int32_t epochs = 1;

  // learning rate (default: 1.0)
  static constexpr float lr = 0.001f;

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

  static inline const std::vector<std::vector<int32_t>> anchors = {
      {10,13}, {16,30}, {33,23},
      {30,61},{62,45}, {59,119},
      {116,90}, {156,198}, {373,326}
  };

  static inline const std::vector<int32_t> input_shape = {416, 416};

  inline static const std::string train_annotation_path = R"(E:\data\coco128\coco128\train_annotation.txt)";

  inline static const std::string test_annotation_path = R"(E:\data\coco128\coco128\test_annotation.txt)";
};
void train(TrainArgs &args, nn::Module &model, nn::Module &loss_fun, Device device,
           data::DataLoader &dataLoader, optim::Optimizer &optimizer,
           int32_t epoch) {
  model.train();

  Timer timer;
  timer.start();
  for (auto [batchIdx, batch] : dataLoader) {
    auto &data = batch[0].to(device);
    auto &target = batch[1].to(device);
    optimizer.zeroGrad();
    auto output = model.forward(data, true);
    output.push_back(target);
    auto loss = loss_fun.forward(output);
    loss.backward();
    optimizer.step();

    if (batchIdx % args.logInterval == 0) {
      timer.mark();
      auto currDataCnt = batchIdx * dataLoader.batchSize();
      auto totalDataCnt = dataLoader.dataset().size();
      auto elapsed = (float)timer.elapseMillis() / 1000.f;  // seconds
      LOGD("Train Epoch: %d [%d/%d (%.0f%%)] Loss: %.6f, Elapsed: %.2fs", epoch,
           currDataCnt, totalDataCnt, 100.f * currDataCnt / (float)totalDataCnt,
           loss.item(), elapsed);

      if (args.dryRun) {
        break;
      }
    }
  }
}

void test(nn::Module &model, nn::Module &loss_fun, Device device, data::DataLoader &dataLoader) {
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
      testLoss += loss_fun(output, target).item();
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


void demo_yolov3() {
  LOGD("demo_yolov3 ...");
  Timer timer;
  timer.start();

  TrainArgs args;
  manualSeed(args.seed);
  // the anchors
  const std::vector<std::vector<int32_t>> anchors = {
      {10,13}, {16,30}, {33,23},
      {30,61}, {62,45}, {59,119},
      {116,90}, {156,198}, {373,326}
  };

  // input shape
  const std::vector<int> input_shape = {416, 416};

  auto useCuda = (!args.noCuda) && Tensor::deviceAvailable(Device::CUDA);
  Device device = useCuda ? Device::CUDA : Device::CPU;
  LOGD("Train with device: %s", useCuda ? "CUDA" : "CPU");

  auto transform = std::make_shared<data::transforms::Compose>(
          data::transforms::ResizeToTensor(args.input_shape),
          data::transforms::Normalize(0.5f, 0.2f)
      );

  auto trainDataset = std::make_shared<data::DatasetYOLO>(
      args.train_annotation_path, data::DatasetYOLO::TRAIN, transform);

  auto testDataset = std::make_shared<data::DatasetYOLO>(
      args.test_annotation_path, data::DatasetYOLO::TEST, transform);

  if (trainDataset->size() == 0 || testDataset->size() == 0) {
    LOGE("Dataset invalid.");
    return;
  }

  auto trainDataloader = data::DataLoader(trainDataset, args.batchSize, true);
  auto testDataloader = data::DataLoader(testDataset, args.testBatchSize, true);
  auto darknet = DarkNet();
  darknet.to(Device::CUDA);
  darknet.train();
  auto model = YoloBody({3,3,3},
                        trainDataset->getNumClass(), darknet,false);
  model.to(device);
  std::cout << model.getTopologyText() << std::endl;
  auto loss = YoloLoss(trainDataset->getNumClass(), args.input_shape, args.anchors);
  loss.to(device);
  auto optimizer = optim::Adam(model.parameters(), args.lr);
  auto scheduler = optim::lr_scheduler::StepLR(optimizer, 1, args.gamma);

  for (auto epoch = 1; epoch < args.epochs + 1; epoch++) {
    train(args, model, loss, device, trainDataloader, optimizer, epoch);
    //test(model, loss, device, testDataloader);
    scheduler.step();
  }

  if (args.saveModel) {
    save(model, "mnist_cnn.model");
  }

  timer.mark();
  LOGD("Total Time cost: %lld ms", timer.elapseMillis());
}