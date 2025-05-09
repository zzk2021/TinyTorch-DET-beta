#include "Torch.h"
#include "Init.h"
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
  Tensor forward(Tensor &x) override {
      auto residual = x;
      x = conv1(x);
      x = bn1(x);
      x = Function::leakyrelu(x, 0.1);
      x = conv2(x);
      x = bn2(x);
      x = Function::leakyrelu(x, 0.1);
      x = x + residual;
      return x;
    }
 private:
  nn::Conv2D conv1;
  nn::BatchNorm2D bn1;
  nn::Conv2D conv2;
  nn::BatchNorm2D bn2;
};

class DarkNet : public nn::Module {
  public:
    DarkNet(const std::vector<int>& layers = {1, 2, 8, 8, 4}):
      inplanes(32),
      conv1(3, inplanes, 3, 1, 1, false),
      bn1(inplanes)
    {
        registerModules({conv1, bn1});
        layers_out_filters = {64, 128, 256, 512, 1024};
        layer1 = _make_layer({inplanes, layers_out_filters[0]}, layers[0]);
        layer2 = _make_layer({layers_out_filters[0], layers_out_filters[1]}, layers[1]);
        layer3 = _make_layer({layers_out_filters[1], layers_out_filters[2]}, layers[2]);
        layer4 = _make_layer({layers_out_filters[2], layers_out_filters[3]}, layers[3]);
        layer5 = _make_layer({layers_out_filters[3], layers_out_filters[4]}, layers[4]);
        registerModules({layer1, layer2, layer3, layer4, layer5});
        initialize_weights();
    }

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
                nn::Conv2D* derivedPtr = dynamic_cast<nn::Conv2D*>(&module.get());
                if (derivedPtr) {
                    TinyTorch::nn::Init::kaimingUniform(derivedPtr->weights(), std::sqrt(2.f), TinyTorch::nn::FanMode::FAN_OUT);
                }
        }
    }
  public:
     std::vector<int> get_layers_out_filters() { return layers_out_filters;}
};

class YoloBody : public nn::Module {
  public:
    YoloBody(std::vector<int32_t> anchors_mask_length, int32_t num_classes, bool pretrained):
       num_classes_(num_classes)
      {
         backbone = DarkNet();
         auto out_filters = backbone.get_layers_out_filters();
         last_layer0 = _make_last_layer({512, 1024}, out_filters[-1], anchors_mask_length[0] * (num_classes + 5));
         last_layer1_conv   = conv2d(512, 256, 1);
         last_layer1 = _make_last_layer({256, 512}, out_filters[-2], anchors_mask_length[1] * (num_classes + 5));
         last_layer2_conv   = conv2d(256, 128, 1);
         last_layer2 = _make_last_layer({128, 256}, out_filters[-3], anchors_mask_length[2] * (num_classes + 5));
      }

    std::vector<Tensor> forward(Tensor &x, bool l) override {
            auto x_ = backbone.forward(x, true);
            auto x2 =  x_[0];auto x1 =  x_[1];auto x0 =  x_[2];
            auto out0_branch = last_layer0[{0, 5}](x1);
            auto out0 = last_layer0[{5, last_layer0.getsize()}](out0_branch);
            // # 13,13,512 -> 13,13,256 -> 26,26,256
            //        x1_in = self.last_layer1_conv(out0_branch)
            //        x1_in = self.last_layer1_upsample(x1_in)
            auto x1_in = last_layer1_conv(out0_branch);
            x1_in = Function::upsample(x1_in,2);
            //   # 26,26,256 + 26,26,512 -> 26,26,768
            //        x1_in = torch.cat([x1_in, x1], 1)
            x1_in = Function::concat(x1_in, x1, -1);
            auto out1_branch = last_layer1[{0, 5}](x1_in);
            auto out1  = last_layer1[{5,last_layer1.getsize()}](out1_branch);
            //        # 26,26,256 -> 26,26,128 -> 52,52,128
            //        x2_in = self.last_layer2_conv(out1_branch)
            //        x2_in = self.last_layer2_upsample(x2_in)
            auto x2_in = last_layer2_conv(out1_branch);
            x2_in = Function::upsample(x2_in,2);
            //# 52,52,128 + 52,52,256 -> 52,52,384
            //        x2_in = torch.cat([x2_in, x2], 1)
            //        #---------------------------------------------------#
            //        #   out3 = (batch_size,255,52,52)
            //        #---------------------------------------------------#
            //        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
            //        out2 = self.last_layer2(x2_in)
            x2_in = Function::concat(x2_in, x2, 1);
            auto out2 = last_layer2(x2_in);
            return {out0, out1, out2};
        }

  private:
      int32_t num_classes_;
      DarkNet backbone;
      nn::Sequential last_layer0;
      nn::Sequential last_layer1;
      nn::Sequential last_layer1_conv;
      nn::Sequential last_layer2;
      nn::Sequential last_layer2_conv;
      nn::Sequential conv2d(int32_t filter_in,int32_t filter_out,int32_t kernel_size){
          auto pad = (kernel_size - 1); // 2 if kernel_size else 0
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
       YoloLoss(std::vector<std::vector<int32_t>> anchors, int32_t num_classes, Shape input_shape, std::vector<std::vector<int32_t>> anchors_mask = {{6,7,8}, {3,4,5}, {0,1,2}}):
            num_classes_(num_classes),
            anchors_(anchors),
            input_shape_(input_shape),
            anchors_mask_(anchors_mask),
            box_ratio_(0.05),
            obj_ratio_(5 * (input_shape_[0] * input_shape_[1]) / (pow(416,2))),
            cls_ratio_(1 * (num_classes / 80)),
            ignore_threshold_(0.5)
            {


            }
    private:
      int32_t num_classes_;
      std::vector<std::vector<int32_t>> anchors_;
      Shape input_shape_;
      std::vector<std::vector<int32_t>> anchors_mask_ ;
      float box_ratio_ ;
      float obj_ratio_ ;
      float cls_ratio_ ;
      float ignore_threshold_;
      nn::MSELoss mseloss;
};
