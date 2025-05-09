
#include "../Function.h"
#include "Function_explore.h"

namespace TinyTorch {

Tensor Function::giou(const Tensor& a, const Tensor& b) {
    //        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    //        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
  return std::make_shared<FuncGIOU>()->callForward({&a, &b});
}

TensorImpl FuncGIOU::forward(const std::vector<const Tensor *> &inputs)  {
    throw std::runtime_error("We have not implement in CPU yet");
}
std::vector<TensorImpl> FuncGIOU::backward(const TinyTorch::TensorImpl &grad)  {
    throw std::runtime_error("We have not implement in CPU yet");
}

}
