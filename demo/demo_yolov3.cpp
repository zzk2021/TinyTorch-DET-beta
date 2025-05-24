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
                auto derivedPtr = dynamic_cast<nn::Conv2D*>(&module.get());
                if (derivedPtr) {
                    TinyTorch::nn::Init::kaimingUniform(derivedPtr->weights(), std::sqrt(2.f), TinyTorch::nn::FanMode::FAN_OUT);
                }
        }
    }
  public:
     std::vector<int> get_layers_out_filters() { return layers_out_filters;}
};

class YoloHead : public nn::Module {
  public:
    YoloHead(std::vector<int32_t> anchors_mask_length, int32_t num_classes, bool pretrained):
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
            auto out0 = last_layer0[{5, static_cast<int>(last_layer0.getsize())}](out0_branch);
            auto x1_in = last_layer1_conv(out0_branch);
            x1_in = Function::upsample(x1_in,2);
            x1_in = Function::concat(x1_in, x1, -1);
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
      DarkNet backbone;
      nn::Sequential last_layer0;
      nn::Sequential last_layer1;
      nn::Sequential last_layer1_conv;
      nn::Sequential last_layer2;
      nn::Sequential last_layer2_conv;
      static nn::Sequential conv2d(int32_t filter_in,int32_t filter_out,int32_t kernel_size){
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
            anchors_(std::move(anchors)),
            input_shape_(std::move(input_shape)),
            anchors_mask_(std::move(anchors_mask)),
            box_ratio_(0.05),
            bbox_attrs_(5 + num_classes),
            obj_ratio_(5.f * (input_shape_[0] * input_shape_[1]) / (pow(416,2))),
            cls_ratio_(1.f * ((float)num_classes / 80)),
            ignore_threshold_(0.5)
            {


            }
        Tensor forward(Tensor &inputs, Array3d &targets) override {
           Tensor loss  = Tensor::scalar(0);
           for (int l=0;l < inputs.shape()[0];l+=1){
            Tensor input = inputs[{{l,l+1},{},{},{},{}}];
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
            auto x = Function::sigmoid(prediction[{{},{},{0,1},{},{}}]);
            auto y = Function::sigmoid(prediction[{{},{},{1,2},{},{}}]);
            auto w = Function::sigmoid(prediction[{{},{},{2,3},{},{}}]);
            auto h = Function::sigmoid(prediction[{{},{},{3,4},{},{}}]);
            auto conf = Function::sigmoid(prediction[{{},{},{4,5},{},{}}]);
            auto pred_cls = Function::sigmoid(prediction[{{},{},{5,-1},{},{}}]);
            auto [y_true, noobj_mask, box_loss_scale, obj_mask, n] = get_target_(l, targets, scaled_anchors, in_h, in_w);
            Tensor y_true_t = Tensor(y_true, false);
            Tensor box_loss_scale_t =  Tensor(box_loss_scale, false);
            Tensor obj_mask_t =  Tensor(obj_mask, false);
            Tensor noobj_mask_t =  Tensor(noobj_mask, false);
            box_loss_scale_t = 2 - box_loss_scale_t;
            box_loss_scale_t.to(input.device());
            obj_mask_t.to(input.device());
            y_true_t.to(input.device());
            if (n != 0){
                auto loss_x  = (Function::bceLoss(x.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{0,1}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_y  = (Function::bceLoss(y.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{1,2}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_w  = (Function::mseLoss(w.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{2,3}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_h  = (Function::mseLoss(h.squeeze()[obj_mask_t], y_true_t[{{},{},{},{},{3,4}}].squeeze()[obj_mask_t], TinyTorch::NONE) * box_loss_scale_t[obj_mask_t]).mean();
                auto loss_loc = (loss_x + loss_y + loss_h + loss_w) * 0.1;
                auto loss_cls = Function::bceLoss(pred_cls.squeeze()[obj_mask_t],y_true_t[{{},{},{},{},{5,-1}}].squeeze()[obj_mask_t], TinyTorch::MEAN);
                loss  += loss_loc * box_ratio_ + loss_cls * cls_ratio_;
            }
            auto loss_conf   = Function::bceLoss(conf, obj_mask_t, TinyTorch::MEAN)[noobj_mask_t + obj_mask_t];
            loss        += loss_conf  * obj_ratio_;
           }
           return loss;
        }
    private:
      int32_t num_classes_;
      std::vector<std::vector<int32_t>> anchors_;
      Shape input_shape_;
      std::vector<std::vector<int32_t>> anchors_mask_ ;
      float box_ratio_ ;
      int bbox_attrs_ ;
      float obj_ratio_ ;
      float cls_ratio_ ;
      float ignore_threshold_;
      nn::MSELoss mseloss;
      std::tuple<Array5d, Array4d, Array4d, Array4d, int> get_target_(int32_t l, Array3d targets, Array2d anchors, int32_t in_h, int32_t in_w){
        int bs               = targets.size();
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
            Array2d(in_h, Array1d(in_w, false))));
        int n = 0;
        for(int b=0;b<bs;b+=1){
          if (targets[b].empty())
            continue;
          Array2d batch_target(targets[b].size(), std::vector<float>(5, 0.0f));

          for (size_t t = 0; t < batch_target.size(); ++t) {
            batch_target[t][0] = targets[b][t][0] * in_w;
            batch_target[t][2] = targets[b][t][2] * in_w;
            batch_target[t][1] = targets[b][t][1] * in_h;
            batch_target[t][3] = targets[b][t][3] * in_h;
            batch_target[t][4] = targets[b][t][4];
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
            obj_mask[b][k][j][i] = true;
            n+=1;
          }
        }
        return std::make_tuple(y_true, noobj_mask, box_loss_scale, obj_mask, n);
      }


};
