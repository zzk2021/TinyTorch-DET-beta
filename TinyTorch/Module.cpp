/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Module.h"

#include <cassert>

#include "Function.h"
#include "Init.h"
#include "Enums.h"
namespace TinyTorch::nn {

void Module::getTopologyTextHelper(std::stringstream& ss, int depth) const {
    ss << std::string(depth * 2, ' ') << "|--" << name() << std::endl;
    for (const auto& submod : subModules_) {
        submod.get().getTopologyTextHelper(ss, depth + 1);
    }
}

std::vector<Tensor *> Module::parameters() {
  std::vector<Tensor *> ret;
  for (auto &module : subModules_) {
    for (auto p : module.get().parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

std::vector<Tensor *> Module::states() {
  std::vector<Tensor *> ret;
  for (auto &module : subModules_) {
    for (auto p : module.get().states()) {
      ret.push_back(p);
    }
  }
  return ret;
}

void Module::resetParameters() {
  for (auto &module : subModules_) {
    module.get().resetParameters();
  }
}

void Module::zeroGrad() {
  for (auto &module : subModules_) {
    module.get().zeroGrad();
  }
}

void Module::to(Device device) {
  for (auto &module : subModules_) {
    for (auto p : module.get().states()) {
      p->to(device);
    }
  }
}

void Module::to(Dtype type) {
  for (auto &module : subModules_) {
    for (auto p : module.get().states()) {
      p->to(type);
    }
  }
}

Tensor Sequential::forward(Tensor &input) {
  Tensor ret = {input};
  for (auto &module : modules_) {
    ret = (*module)(ret);
  }

  return ret;
}

std::vector<Tensor *> Sequential::parameters() {
  std::vector<Tensor *> ret;
  for (auto &module : modules_) {
    for (auto p : module->parameters()) {
      ret.push_back(p);
    }
  }
  return ret;
}

std::vector<Tensor *> Sequential::states() {
  std::vector<Tensor *> ret;
  for (auto &module : modules_) {
    for (auto p : module->states()) {
      ret.push_back(p);
    }
  }
  return ret;
}

void Sequential::resetParameters() {
  for (auto &module : modules_) {
    module->resetParameters();
  }
}

void Sequential::zeroGrad() {
  for (auto &module : modules_) {
    module->zeroGrad();
  }
}

void Sequential::setTraining(bool mode) {
  Module::setTraining(mode);
  for (auto &module : modules_) {
    module->train(mode);
  }
}

FlashSelfAttention::FlashSelfAttention(int32_t inFeatures, int32_t head, AttentionMethod method)
    : inFeatures_(inFeatures),  head_(head), method_(method) {
}

Tensor FlashSelfAttention::forward(Tensor &Q, Tensor &K, Tensor &V) {
  if (method_ == AttentionMethod::FalshAttentionV2) {
    return Function::flashattention(Q,K,V,head_);
  }
  else if (method_ == AttentionMethod::Attention){
    return Function::flashattention(Q,K,V,head_);
  }
}


Linear::Linear(int32_t inFeatures, int32_t outFeatures, bool bias)
    : inFeatures_(inFeatures), outFeatures_(outFeatures), useBias_(bias) {
  weights_ = Tensor::shape({outFeatures, inFeatures}, true);
  if (bias) {
    bias_ = Tensor::shape({outFeatures}, true);
  }
  Linear::resetParameters();
}

Tensor Linear::forward(Tensor &input) {
  return Function::linear(input, weights_, bias_);
}

std::vector<Tensor *> Linear::parameters() {
  if (useBias_) {
    return {&weights_, &bias_};
  }
  return {&weights_};
}



std::vector<Tensor *> Linear::states() { return parameters(); }

void Linear::resetParameters() {
  Init::kaimingUniform(weights_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weights_).first;
    const auto bound = fanIn > 0 ? 1.f / std::sqrt((float)fanIn) : 0;
    Init::uniform(bias_, -bound, bound);
  }
}

void Linear::zeroGrad() {
  weights_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

Tensor Flatten::forward(Tensor &input) {
  return Function::flatten(input, startDim_, endDim_);
}

Tensor Relu::forward(Tensor &input) { return Function::relu(input); }

Tensor LeakyRelu::forward(Tensor &input) { return Function::leakyrelu(input, rate_); }

Tensor Dropout::forward(Tensor &input) {
  return Function::dropout(input, p_, training_);
}

Tensor Softmax::forward(Tensor &input) {
  return Function::softmax(input, dim_);
}

Tensor LogSoftmax::forward(Tensor &input) {
  return Function::logSoftmax(input, dim_);
}

Tensor MaxPool2D::forward(Tensor &input) {
  return Function::maxPool2d(input, kernelSize_, stride_, padding_);
}

Conv1D::Conv1D(int32_t inFeatures, int32_t outFeatures, Size1D kernelSize,
               Size1D stride, Size1D padding, bool bias)
    : inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      kernelSize_(kernelSize),
      stride_(stride),
      padding_(padding),
      useBias_(bias){
  weights_ = Tensor::shape(
      {outFeatures, inFeatures, kernelSize_.d}, true);
  if (bias) {
    bias_ = Tensor::shape({outFeatures}, true);
  }
  Conv1D::resetParameters();
}

Tensor Conv1D::forward(Tensor &input) {
  return Function::conv1d(input, weights_, bias_, stride_, padding_);
}

std::vector<Tensor *> Conv1D::parameters() {
  if (useBias_) {
    return {&weights_, &bias_};
  }
  return {&weights_};
}


std::vector<Tensor *> Conv1D::states() { return parameters(); }

void Conv1D::resetParameters() {
  Init::kaimingUniform(weights_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weights_).first;
    if (fanIn != 0) {
      const auto bound = 1.f / std::sqrt((float)fanIn);
      Init::uniform(bias_, -bound, bound);
    }
  }
}

void Conv1D::zeroGrad() {
  weights_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

Conv2D::Conv2D(int32_t inFeatures, int32_t outFeatures, Size2D kernelSize,
               Size2D stride, Size2D padding, bool bias, Dtype fw_type ,
         Dtype bw_type )
    : inFeatures_(inFeatures),
      outFeatures_(outFeatures),
      kernelSize_(kernelSize),
      stride_(stride),
      padding_(padding),
      useBias_(bias),
      fw_type_(fw_type),
      bw_type_(bw_type){
  weights_ = Tensor::shape(
      {outFeatures, inFeatures, kernelSize_.h, kernelSize_.w}, true);
  if (bias) {
    bias_ = Tensor::shape({outFeatures}, true);
  }
  Conv2D::resetParameters();
}

Tensor Conv2D::forward(Tensor &input) {
  return Function::conv2d(input, weights_, bias_, stride_, padding_);
}

std::vector<Tensor *> Conv2D::parameters() {
  if (useBias_) {
    return {&weights_, &bias_};
  }
  return {&weights_};
}

std::vector<Tensor *> Conv2D::states() { return parameters(); }

void Conv2D::resetParameters() {
  Init::kaimingUniform(weights_, std::sqrt(5.f));
  if (useBias_) {
    auto fanIn = Init::calculateFan(weights_).first;
    if (fanIn != 0) {
      const auto bound = 1.f / std::sqrt((float)fanIn);
      Init::uniform(bias_, -bound, bound);
    }
  }
}

void Conv2D::zeroGrad() {
  weights_.zeroGrad();
  if (useBias_) {
    bias_.zeroGrad();
  }
}

BatchNorm2D::BatchNorm2D(int32_t numFeatures, float eps, float momentum,
                         bool affine, bool trackRunningStats)
    : numFeatures_(numFeatures),
      eps_(eps),
      momentum_(momentum),
      affine_(affine),
      trackRunningStats_(trackRunningStats),
      numBatchesTracked_(0) {
  if (affine_) {
    weights_ = Tensor::shape({numFeatures_}, true);
    bias_ = Tensor::shape({numFeatures_}, true);
  }
  if (trackRunningStats_) {
    runningMean_ = Tensor::shape({numFeatures_}, true);
    runningVar_ = Tensor::shape({numFeatures_}, true);
  }

  BatchNorm2D::resetParameters();
}

Tensor BatchNorm2D::forward(Tensor &input) {
  assert(input.dim() == 4);
  if (training_ && trackRunningStats_) {
    numBatchesTracked_++;
  }

  bool bnTrain = training_ || !trackRunningStats_;
  return Function::batchNorm(input, runningMean_, runningVar_, weights_, bias_,
                             bnTrain, momentum_, eps_);
}

std::vector<Tensor *> BatchNorm2D::parameters() {
  if (affine_) {
    return {&weights_, &bias_};
  }
  return {};
}

std::vector<Tensor *> BatchNorm2D::states() {
  std::vector<Tensor *> ret({&runningMean_, &runningVar_});
  if (affine_) {
    ret.push_back(&weights_);
    ret.push_back(&bias_);
  }
  return ret;
}

void BatchNorm2D::resetParameters() {
  if (affine_) {
    weights_.data().fill_(1.f);
    bias_.data().fill_(0.f);
  }

  if (trackRunningStats_) {
    runningMean_.data().fill_(0.f);
    runningVar_.data().fill_(1.f);
    numBatchesTracked_ = 0;
  }
}

void BatchNorm2D::zeroGrad() {
  if (affine_) {
    weights_.zeroGrad();
    bias_.zeroGrad();
  }
}

}  // namespace TinyTorch::nn
