/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */
#include "traceback.h"
#include "Function.h"
#include "Objectdetection/Function_explore.h"
#include "Objectdetection/header.h"
#include <numeric>
#include <cassert>
#include <set>
#include <unordered_map>
#include "Tensor.h"

namespace TinyTorch {

#define FUNC_ENUM_TO_STRING(value) {value, #value}

std::unordered_map<FunctionType, std::string> Function::funcTypeToString_ = {
    FUNC_ENUM_TO_STRING(Function_Leaf),
    FUNC_ENUM_TO_STRING(Function_Add),
    FUNC_ENUM_TO_STRING(Function_Sub),
    FUNC_ENUM_TO_STRING(Function_Mul),
    FUNC_ENUM_TO_STRING(Function_Div),
    FUNC_ENUM_TO_STRING(Function_Max),
    FUNC_ENUM_TO_STRING(Function_Sin),
    FUNC_ENUM_TO_STRING(Function_Cos),
    FUNC_ENUM_TO_STRING(Function_Pow),
    FUNC_ENUM_TO_STRING(Function_PowScalar),
    FUNC_ENUM_TO_STRING(Function_Sum),
    FUNC_ENUM_TO_STRING(Function_Relu),
    FUNC_ENUM_TO_STRING(Function_Flatten),
    FUNC_ENUM_TO_STRING(Function_UnFlatten),
    FUNC_ENUM_TO_STRING(Function_FlashAttention),
    FUNC_ENUM_TO_STRING(Function_UpSample),
    FUNC_ENUM_TO_STRING(Function_ConCat),
    FUNC_ENUM_TO_STRING(Function_Slice),
    FUNC_ENUM_TO_STRING(Function_Squeeze),
    FUNC_ENUM_TO_STRING(Function_Unsqueeze),
    FUNC_ENUM_TO_STRING(Function_Reshape),
    FUNC_ENUM_TO_STRING(Function_Linear),
    FUNC_ENUM_TO_STRING(Function_Dropout),
    FUNC_ENUM_TO_STRING(Function_Softmax),
    FUNC_ENUM_TO_STRING(Function_LogSoftmax),
    FUNC_ENUM_TO_STRING(Function_MaxPool2D),
    FUNC_ENUM_TO_STRING(Function_Conv2D),
    FUNC_ENUM_TO_STRING(Function_BatchNorm),
    FUNC_ENUM_TO_STRING(Function_MSELoss),
    FUNC_ENUM_TO_STRING(Function_NLLLoss),
    FUNC_ENUM_TO_STRING(Function_ChangeType),
    OBJDECT_EXPLORE_funcTypeToString_()
};

Tensor Function::changetype(const Tensor& input, Dtype T) {
  return std::make_shared<FuncChangeType>(T, input.type())->callForward({&input});
}

Tensor Function::upsample(const Tensor& input, int32_t scale_factor) {
  return std::make_shared<FuncUpSample>(scale_factor)->callForward({&input});
}

Tensor Function::add(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncAdd>()->callForward({&a, &b});
}

Tensor Function::from_slice(const Tensor& a, const std::vector<int>& start, const std::vector<int>& end) {
  return std::make_shared<FuncSlice>(a.shape(), start, end)->callForward({&a});
}

Tensor Function::from_mask(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncMask>(a.shape())->callForward({&a, &b});
}

Tensor Function::concat(const Tensor& a, const Tensor& b, int32_t dim) {
  return std::make_shared<FuncConCat>(dim, a.shape()[dim])->callForward({&a, &b});
}

Tensor Function::sub(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncSub>()->callForward({&a, &b});
}

Tensor Function::mul(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncMul>()->callForward({&a, &b});
}

Tensor Function::div(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncDiv>()->callForward({&a, &b});
}

Tensor Function::sin(const Tensor& a) {
  return std::make_shared<FuncSin>()->callForward({&a});
}

Tensor Function::cos(const Tensor& a) {
  return std::make_shared<FuncCos>()->callForward({&a});
}

Tensor Function::pow(const Tensor& a, const float& b) {
  return std::make_shared<FuncPowScalar>(b)->callForward({&a});
}

Tensor Function::pow(const Tensor& a, const Tensor& b) {
  return std::make_shared<FuncPow>()->callForward({&a, &b});
}

Tensor Function::sum(const Tensor& a) {
  return std::make_shared<FuncSum>()->callForward({&a});
}

Tensor Function::mean(const Tensor& a) {
  return std::make_shared<FuncMean>()->callForward({&a});
}

Tensor Function::max(const Tensor& a, int32_t dim ,bool keepdim) {
  return std::make_shared<FuncMax>(dim, keepdim)->callForward({&a});
}

Tensor Function::relu(const Tensor& input) {
  return std::make_shared<FuncRelu>()->callForward({&input});
}

Tensor Function::leakyrelu(const Tensor& input, float rate) {
  return std::make_shared<FuncLeakyRelu>(rate)->callForward({&input});
}

Tensor Function::flatten(const Tensor& input, int32_t startDim,
                         int32_t endDim) {
  return std::make_shared<FuncFlatten>(startDim, endDim)->callForward({&input});
}

Tensor Function::unflatten(const Tensor& input, int32_t dim,
                           const std::vector<int32_t>& sizes) {
  return std::make_shared<FuncUnFlatten>(dim, sizes)->callForward({&input});
}

Tensor Function::squeeze(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncSqueeze>(dim)->callForward({&input});
}

Tensor Function::unsqueeze(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncUnsqueeze>(dim)->callForward({&input});
}

Tensor Function::reshape(const Tensor& input, const Shape& shape) {
  return std::make_shared<FuncReshape>(shape)->callForward({&input});
}

Tensor Function::flashattention(const Tensor& Q,const Tensor& K,const Tensor& V, int32_t head) {
  return std::make_shared<FuncFlashAttention>(head)->callForward({&Q, &K, &V});
}

Tensor Function::linear(const Tensor& input, const Tensor& weight,
                        const Tensor& bias) {
  return std::make_shared<FuncLinear>()->callForward({&input, &weight, &bias});
}

Tensor Function::dropout(const Tensor& input, float p, bool training) {
  return std::make_shared<FuncDropout>(p, training)->callForward({&input});
}

Tensor Function::softmax(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncSoftmax>(dim)->callForward({&input});
}

Tensor Function::sigmoid(const Tensor& input) {
  return std::make_shared<FuncSigmoid>()->callForward({&input});
}

Tensor Function::logSoftmax(const Tensor& input, int32_t dim) {
  return std::make_shared<FuncLogSoftmax>(dim)->callForward({&input});
}

Tensor Function::maxPool2d(const Tensor& input, Size2D kernelSize,
                           std::optional<Size2D> stride, Size2D padding) {
  return std::make_shared<FuncMaxPool2D>(
             kernelSize, stride.has_value() ? stride.value() : kernelSize,
             padding)
      ->callForward({&input});
}

Tensor Function::conv2d(const Tensor& input, const Tensor& weight,
                        const Tensor& bias, Size2D stride, Size2D padding) {
  return std::make_shared<FuncConv2D>(stride, padding)
      ->callForward({&input, &weight, &bias});
}

Tensor Function::conv1d(const Tensor& input, const Tensor& weight,
                        const Tensor& bias, Size1D stride, Size1D padding) {
  return std::make_shared<FuncConv1D>(stride, padding)
      ->callForward({&input, &weight, &bias});
}

Tensor Function::layerNorm(const Tensor& x, const Tensor& g, const Tensor& b, float eps) {
  return std::make_shared<FuncLayerNorm>(eps)
      ->callForward({&x, &g, &b});
}

Tensor Function::batchNorm(const Tensor& input, Tensor& runningMean,
                           Tensor& runningVar, const Tensor& weight,
                           const Tensor& bias, bool training, float momentum,
                           float eps) {
  return std::make_shared<FuncBatchNorm>(runningMean, runningVar, momentum, eps,
                                         training)
      ->callForward({&input, &weight, &bias});
}

Tensor Function::mseLoss(const Tensor& input, const Tensor& target,
                         LossReduction reduction) {
  return std::make_shared<FuncMSELoss>(reduction)->callForward(
      {&input, &target});
}

Tensor Function::bceLoss(const Tensor& input, const Tensor& target,
                         LossReduction reduction) {
  return std::make_shared<FuncBCELoss>(reduction)->callForward(
      {&input, &target});
}

Tensor Function::bceLossWithSigmoid(const Tensor& input, const Tensor& target,
                         LossReduction reduction) {
  return std::make_shared<FuncBCELossWithSigmoid>(reduction)->callForward(
      {&input, &target});
}

Tensor Function::nllloss(const Tensor& input, const Tensor& target,
                         LossReduction reduction) {
  return std::make_shared<FuncNLLLoss>(reduction)->callForward(
      {&input, &target});
}

Tensor Function::callForward(const std::vector<const Tensor*>& inputs) {
  auto output = forward(inputs);

  auto requiresGrad = false;
  for (const auto input : inputs) {
    if (input->isRequiresGrad()) {
      requiresGrad = true;
      break;
    }
  }

  std::shared_ptr<Function> gradFunc = nullptr;
  if (NoGradScope::isGradEnabled() && requiresGrad) {
    for (const auto input : inputs) {
      if (input->isRequiresGrad()) {
        nextFuncs_.push_back(input->getGradFunc());
      }
    }
    gradFunc = shared_from_this();
  }

  return Tensor(std::move(output), requiresGrad, gradFunc);
}

std::vector<TensorImpl> Function::callBackward(const TensorImpl& grad) {
  auto ret = backward(grad);
  savedTensors_.clear();
  return ret;
}

TensorImpl FuncLeaf::forward(const std::vector<const Tensor*>& inputs) {
  return {};
}

std::vector<TensorImpl> FuncLeaf::backward(const TensorImpl& grad) {
  auto owner = owner_.lock();
  if (owner == nullptr) {
    return {grad};
  }

  // for broadcast
  if (owner->grad_.data_->shape() != grad.shape()) {
    TensorImpl retGrad =
        TensorImpl::sum(grad, 0, owner->grad_.data_->dim() != 0);
    ASSERT(retGrad.shape() == owner->grad_.data_->shape());
    *owner->grad_.data_ = std::move(retGrad);
    return {*owner->grad_.data_};
  }

  ASSERT(grad.shape() == owner->grad_.data_->shape());

  *owner->grad_.data_ = grad;
  return {grad};
}


TensorImpl FuncUpSample::forward(const std::vector<const Tensor*>& inputs) {
    auto a_in_place = Tensor({0}, inputs[0]->isRequiresGrad());
    saveForBackward({&a_in_place});

    // Get input tensor information
    const auto& input_data = inputs[0]->data().data();  // Get raw data pointer
    const auto& input_shape = inputs[0]->data().shape();  // Get input shape
    const int32_t scale_factor = scale_factor_;  // Get scale factor

    // Determine number of dimensions (assuming NCHW format)
    const int num_dims = input_shape.size();
    if (num_dims != 4) {
        throw std::runtime_error("Upsample only supports 4D tensors (NCHW format)");
    }
    if (inputs[0]->device() == Device::CUDA){
        auto ret = inputs[0]->data().ops()->upsample_forward(inputs[0]->data(), scale_factor);
        return ret;
    }

    // Calculate output shape
    std::vector<int32_t> output_shape(input_shape);
    for (int i = 2; i < num_dims; ++i) {  // Only scale height and width (last two dims)
        output_shape[i] *= scale_factor;
    }
    TensorImpl output  = TensorImpl::zeros(output_shape);
    auto output_data = output.data();
    const int32_t N = input_shape[0];  // batch size
    const int32_t C = input_shape[1];  // channels
    const int32_t H = input_shape[2];  // height
    const int32_t W = input_shape[3];  // width
    const int32_t out_H = output_shape[2];
    const int32_t out_W = output_shape[3];
    // Perform nearest neighbor upsampling
    for (int32_t n = 0; n < N; ++n) {
        for (int32_t c = 0; c < C; ++c) {
            for (int32_t h = 0; h < out_H; ++h) {
                const int32_t in_h = h / scale_factor;
                for (int32_t w = 0; w < out_W; ++w) {
                    const int32_t in_w = w / scale_factor;
                    // Calculate input and output indices
                    const int32_t input_idx = n * C * H * W + c * H * W + in_h * W + in_w;
                    const int32_t output_idx = n * C * out_H * out_W + c * out_H * out_W + h * out_W + w;
                    // Copy value from input to output
                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }
    }
    return output;
}

std::vector<TensorImpl> FuncUpSample::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (!savedTensors.empty() && savedTensors[0].isRequiresGrad()) {
    const int num_dims = grad.shape().size();
    if (num_dims != 4) {
      throw std::runtime_error("Upsample only supports 4D tensors (NCHW format)");
    }
    const int32_t scale = scale_factor_;
    if (grad.device() == Device::CUDA){
      auto ret1 = grad.ops()->upsample_backward(grad, scale);
     return {ret1};
    }
    TensorImpl input_grad = TensorImpl::zeros({grad.shape()[0],
           grad.shape()[1], grad.shape()[2] / scale, grad.shape()[3] / scale});
    auto output_grad = grad.data();
    const int32_t N = input_grad.shape()[0];
    const int32_t C = input_grad.shape()[1];
    const int32_t H = input_grad.shape()[2];
    const int32_t W = input_grad.shape()[3];
    const int32_t out_H = H * scale;
    const int32_t out_W = W * scale;
    for (int32_t n = 0; n < N; ++n) {
        for (int32_t c = 0; c < C; ++c) {
            for (int32_t h = 0; h < out_H; ++h) {
                const int32_t src_h = std::min(h / scale, H - 1);
                for (int32_t w = 0; w < out_W; ++w) {
                    const int32_t src_w = std::min(w / scale, W - 1);
                    const int32_t input_idx = ((n * C + c) * H + src_h) * W + src_w;
                    const int32_t output_idx = ((n * C + c) * out_H + h) * out_W + w;
                    input_grad.data()[input_idx] += output_grad[output_idx];
                }
            }
        }
    }
    ret.push_back(input_grad);

    }
  return ret;
}

TensorImpl FuncMask::forward(const std::vector<const Tensor*>& inputs) {
    // notice! we could not save tensor b grad, tensor b for mask can't backpropagation
    const Tensor& a = *inputs[0];
    const Tensor& b = *inputs[1];
    ASSERT(!b.isRequiresGrad());
    //ASSERT(!b.isRequiresGrad() && "the mask must be no Grad");
    Tensor a_in_place = Tensor({0},a.isRequiresGrad());
    saveForBackward({&a_in_place});
    auto ret = a.data().ops()->from_mask(a.data(), b.data());
    indice_ = ret.second;
    return ret.first;
}

std::vector<TensorImpl> FuncMask::backward(const TensorImpl& grad) {
    const auto& savedTensors = getSavedTensors();
    std::vector<TensorImpl> ret;
    if (!savedTensors.empty() && savedTensors[0].isRequiresGrad()) {
        auto grad_output = grad.ops()->from_mask_backward(grad, indice_, a_shape_);
        ret.push_back(grad_output);
    }
    return ret;
}

TensorImpl FuncSlice::forward(const std::vector<const Tensor*>& inputs) {
    Tensor a_in_place = Tensor({0}, inputs[0]->isRequiresGrad());
    saveForBackward({&a_in_place});
    const Tensor& t = *inputs[0];
    const auto& shape = t.data().shape();
    return t.data().ops()->from_slice(t.data(), start_, end_);
}

std::vector<TensorImpl> FuncSlice::backward(const TensorImpl& grad) {
    const auto& savedTensors = getSavedTensors();
    std::vector<TensorImpl> ret;
    if (!savedTensors.empty() && savedTensors[0].isRequiresGrad()) {
        const TensorImpl grad_input = TensorImpl::zeros(a_shape_, grad.device(), grad.type());
        grad_input.ops()->from_slice_backward(const_cast<TensorImpl &>(grad_input), grad, start_, end_);
        ret.push_back(grad_input);
    }
    return ret;
}

TensorImpl FuncConCat::forward(const std::vector<const Tensor*>& inputs) {
    Tensor a_in_place = Tensor({0}, inputs[0]->isRequiresGrad());
    Tensor b_in_place = Tensor({0}, inputs[1]->isRequiresGrad());
    saveForBackward({&a_in_place, &b_in_place});
    ASSERT(inputs.size() == 2);
    const Tensor& t1 = *inputs[0];
    const Tensor& t2 = *inputs[1];
    return  t1.data().concat(t1.data(), t2.data(), dim_);
}

std::vector<TensorImpl> FuncConCat::backward(const TensorImpl& grad) {
    const auto& savedTensors = getSavedTensors();
    std::vector<TensorImpl> ret;
    ASSERT(savedTensors.size() == 2);
    const bool need_grad1 = savedTensors[0].isRequiresGrad();
    const bool need_grad2 = savedTensors[1].isRequiresGrad();
    if (!need_grad1 && !need_grad2) {
        return ret;
    }
      auto grad_list = grad.split(grad, a_shape_, dim_,0);
      if (need_grad1) ret.push_back(grad_list[0]);
      if (need_grad2) ret.push_back(grad_list[1]);
    return ret;
}

TensorImpl FuncChangeType::forward(const std::vector<const Tensor*>& inputs) {
  Tensor a_in_place = Tensor::scalar(0, inputs[0]->isRequiresGrad());
  saveForBackward({&a_in_place});
  return inputs[0]->data().to(T_);
}

std::vector<TensorImpl> FuncChangeType::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret  ;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad.to(Ori_T_));
  }
  return ret;
}


TensorImpl FuncAdd::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() + inputs[1]->data();
}

std::vector<TensorImpl> FuncAdd::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(grad);
  }
  return ret;
}

TensorImpl FuncSub::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() - inputs[1]->data();
}

std::vector<TensorImpl> FuncSub::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(-1 * grad);
  }
  return ret;
}

TensorImpl FuncMul::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() * inputs[1]->data();
}

std::vector<TensorImpl> FuncMul::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(savedTensors[1].data() * grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(savedTensors[0].data() * grad);
  }
  return ret;
}

TensorImpl FuncDiv::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return inputs[0]->data() / inputs[1]->data();
}

std::vector<TensorImpl> FuncDiv::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad / savedTensors[1].data());
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(-1.f * grad * savedTensors[0].data() /
                  (savedTensors[1].data() * savedTensors[1].data()));
  }
  return ret;
}

TensorImpl FuncSin::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::sin(inputs[0]->data());
}

std::vector<TensorImpl> FuncSin::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(TensorImpl::cos(savedTensors[0].data()) * grad);
  }
  return ret;
}

TensorImpl FuncCos::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::cos(inputs[0]->data());
}

std::vector<TensorImpl> FuncCos::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(-1.f * TensorImpl::sin(savedTensors[0].data()) * grad);
  }
  return ret;
}

TensorImpl FuncPow::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::pow(inputs[0]->data(), inputs[1]->data());
}

std::vector<TensorImpl> FuncPow::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(
        savedTensors[1].data() *
        TensorImpl::pow(savedTensors[0].data(), savedTensors[1].data() - 1) *
        grad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(savedTensors[0].data() * grad);
  }
  return ret;
}

TensorImpl FuncPowScalar::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::pow(inputs[0]->data(), exp_);
}

std::vector<TensorImpl> FuncPowScalar::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(exp_ * TensorImpl::pow(savedTensors[0].data(), exp_ - 1.f) *
                  grad);
  }
  return ret;
}


TensorImpl FuncMax::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto maxRet = TensorImpl::max(inputs[0]->data(),dim_, keep_dim_);
  maxIndices_ = maxRet.second;
  return maxRet.first;
}

std::vector<TensorImpl> FuncMax::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (!savedTensors.empty() && savedTensors[0].isRequiresGrad()) {
      ASSERT(savedTensors.size() == 1);
      const auto& inputShape = savedTensors[0].data().shape();
      std::vector<int32_t> Indices_tuple;
      auto gradInput = TensorImpl::zeros(inputShape, grad.device(),grad.type());
        int32_t elemSize = maxIndices_.numel();
        const auto& maxIndicesShape = maxIndices_.shape();
        std::vector<int32_t> indices(maxIndicesShape.size(), 0);
        std::vector<int32_t> strides(maxIndicesShape.size(), 1);
        for (int i = maxIndicesShape.size() - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * maxIndicesShape[i + 1];
        }
        for (int32_t n = 0; n < elemSize; ++n) {
            std::vector<int32_t> coord(maxIndicesShape.size());
            int32_t remainder = n;
            for (size_t dim = 0; dim < maxIndicesShape.size(); ++dim) {
                coord[dim] = remainder / strides[dim];
                remainder = remainder % strides[dim];
            }
            int32_t maxIndex = maxIndices_.data()[n];
            std::vector<int32_t> gradInputIndices;
                for (size_t dim = 0; dim < coord.size(); ++dim) {
                    if (dim == dim_)
                        gradInputIndices.push_back(maxIndex);
                    else
                        gradInputIndices.push_back(coord[dim]);
                }
        gradInput.indexPut_(gradInputIndices, grad.data()[n]);
      }
        ret.push_back(gradInput);
    }
  return ret;
}

TensorImpl FuncMean::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::sum(inputs[0]->data()) / (float)inputs[0]->numel();
}

std::vector<TensorImpl> FuncMean::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  auto& input = savedTensors[0];
  if (input.isRequiresGrad()) {
    ret.push_back(grad  / (float)input.numel());
  }
  return ret;
}

TensorImpl FuncSum::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::sum(inputs[0]->data());
}

std::vector<TensorImpl> FuncSum::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  auto& input = savedTensors[0];
  if (input.isRequiresGrad()) {
      ret.push_back(grad * TensorImpl::onesLike(input.data(), input.device(), input.type()));
  }
  return ret;
}

TensorImpl FuncLeakyRelu::forward(const std::vector<const Tensor*>& inputs) {
  auto [ret,mask] = inputs[0]->data().ops()->leakyrelu(inputs[0]->data(), rate_);
  Tensor mask_t = Tensor(std::move(mask), inputs[0]->isRequiresGrad());
  saveForBackward({&mask_t});

  return ret;
}

std::vector<TensorImpl> FuncLeakyRelu::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
        ret.push_back(grad.ops()->leakyrelu_backward(grad, savedTensors[0].data(), rate_));
  }
  return ret;
}

TensorImpl FuncRelu::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::clampMin(inputs[0]->data(), 0);
}

std::vector<TensorImpl> FuncRelu::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(grad * (savedTensors[0].data() > 0));
  }
  return ret;
}

TensorImpl FuncFlatten::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::flatten(inputs[0]->data(), startDim_, endDim_);
}

std::vector<TensorImpl> FuncFlatten::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ASSERT(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncUnFlatten::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::unflatten(inputs[0]->data(), dim_, sizes_);
}

std::vector<TensorImpl> FuncUnFlatten::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ASSERT(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncSqueeze::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::squeeze(inputs[0]->data(), dim_);
}

std::vector<TensorImpl> FuncSqueeze::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ASSERT(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncUnsqueeze::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::unsqueeze(inputs[0]->data(), dim_);
}

std::vector<TensorImpl> FuncUnsqueeze::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ASSERT(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncReshape::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  return TensorImpl::reshape(inputs[0]->data(), shape_);
}

std::vector<TensorImpl> FuncReshape::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ASSERT(grad.numel() == savedTensors[0].numel());
    ret.push_back(TensorImpl::reshape(grad, savedTensors[0].shape()));
  }
  return ret;
}

TensorImpl FuncFlashAttention::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto output = TensorImpl::flashattentionv2(inputs[0]->data() ,
    inputs[1]->data() , inputs[2]->data(), head_);
  return output;
}

std::vector<TensorImpl> FuncFlashAttention::backward(const TensorImpl& grad) {
  std::vector<TensorImpl> ret;
  return ret;
}

TensorImpl FuncLinear::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto output = TensorImpl::matmulTrans(inputs[0]->data(), inputs[1]->data(),
                                        false, true);
  if (!inputs[2]->empty()) {
    output += inputs[2]->data();
  }
  return output;
}

std::vector<TensorImpl> FuncLinear::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(TensorImpl::matmul(grad, savedTensors[1].data()));
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(
        TensorImpl::matmulTrans(grad, savedTensors[0].data(), true, false));
  }
  if (!savedTensors[2].empty() && savedTensors[2].isRequiresGrad()) {
    ret.push_back(TensorImpl::sum(grad, 0));
  }
  return ret;
}

TensorImpl FuncDropout::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  if (training_) {
    mask_ = TensorImpl::bernoulli(inputs[0]->shape(), 1.f - p_,
                                  inputs[0]->device(), inputs[0]->type());
    return mask_ * inputs[0]->data() / (1.f - p_);
  } else {
    return inputs[0]->data();
  }
}

std::vector<TensorImpl> FuncDropout::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(mask_ * grad / (1.f - p_));
  }
  return ret;
}



TensorImpl FuncSoftmax::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto max = TensorImpl::max(inputs[0]->data(), dim_, true).first;
  auto shifted = inputs[0]->data() - max;
  auto exp = TensorImpl::exp(shifted);
  auto sumExp = TensorImpl::sum(exp, dim_, true);
  forwardOutput_ = exp / sumExp;
  return forwardOutput_;
}

TensorImpl FuncSigmoid::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  forwardOutput_ = 1 / (1 + 1 / inputs[0]->data().ops()->exp(inputs[0]->data()));
  return forwardOutput_;
}

std::vector<TensorImpl> FuncSigmoid::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto retGrad = grad * forwardOutput_ * (1- forwardOutput_);
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

std::vector<TensorImpl> FuncSoftmax::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto retGrad = forwardOutput_ *
                   (grad - TensorImpl::sum(grad * forwardOutput_, dim_, true));
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

TensorImpl FuncLogSoftmax::forward(const std::vector<const Tensor*>& inputs) {
  Tensor a_in_place = Tensor::scalar(0,inputs[0]->isRequiresGrad());
  saveForBackward({&a_in_place});
  auto max = TensorImpl::max(inputs[0]->data(), dim_, true).first;
  auto logSumExp = TensorImpl::log(
      TensorImpl::sum(TensorImpl::exp(inputs[0]->data() - max), dim_, true));
  forwardOutput_ = inputs[0]->data() - max - logSumExp;
  return forwardOutput_;
}

std::vector<TensorImpl> FuncLogSoftmax::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto softmax = TensorImpl::exp(forwardOutput_);
    auto retGrad = grad - TensorImpl::sum(grad, dim_, true) * softmax;
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

TensorImpl FuncMaxPool2D::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto& shape = inputs[0]->shape();
  ASSERT(shape.size() == 3 || shape.size() == 4);
  int32_t batch = (shape.size() == 4) ? shape[0] : 1;
  int32_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int32_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int32_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernelSize_.h + 2 * padding_.h) / stride_.h + 1;
  auto outW = (width - kernelSize_.w + 2 * padding_.w) / stride_.w + 1;

  auto col = inputs[0]->data().im2col(kernelSize_, stride_, padding_);
  col.reshape_({-1, kernelSize_.h * kernelSize_.w});

  auto maxRet = TensorImpl::max(col, 1);
  maxIndices_ = maxRet.second;

  auto ret = maxRet.first;
  ret.reshape_({batch, channels, outH, outW});
  return ret;
}

std::vector<TensorImpl> FuncMaxPool2D::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& shape = savedTensors[0].shape();
  ASSERT(shape.size() == 3 || shape.size() == 4);
  int32_t batch = (shape.size() == 4) ? shape[0] : 1;
  int32_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int32_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int32_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernelSize_.h + 2 * padding_.h) / stride_.h + 1;
  auto outW = (width - kernelSize_.w + 2 * padding_.w) / stride_.w + 1;
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto gradCol = TensorImpl::zeros(
        {grad.numel(), kernelSize_.h * kernelSize_.w}, grad.device(),grad.type());
    auto gradIdx =
        TensorImpl::arange(0, (float)grad.numel(), 1.f, grad.device());
    gradCol.indexPut_({gradIdx, maxIndices_}, grad);
    gradCol.reshape_(
        {batch * outH * outW, channels * kernelSize_.h * kernelSize_.w});
    auto retGrad = gradCol.col2im(shape, kernelSize_, stride_, padding_);
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

TensorImpl FuncConv1D::forward(const std::vector<const Tensor*>& inputs) {
    saveForBackward(inputs);
    auto& input = inputs[0]->data();
    auto& weight = inputs[1]->data();
    auto& bias = inputs[2]->data();

    ASSERT(input.dim() == 3);  // [batch, in_channels, length]
    ASSERT(weight.dim() == 3); // [out_channels, in_channels, kernel_size]
    ASSERT(input.shape()[1] == weight.shape()[1]); //

    const int32_t batch = input.shape()[0];
    const int32_t outChannels = weight.shape()[0];
    const int32_t inChannels = weight.shape()[1];
    const int32_t kernelSize = weight.shape()[2];
    const int32_t inputLength = input.shape()[2];

    const int32_t outLength = (inputLength - kernelSize + 2 * padding_.d) / stride_.d + 1;
    col_ = input.im2col1D(kernelSize, stride_, padding_); // [batch, outLength, inChannels*kernelSize]
    auto colW = TensorImpl::reshape(weight, {outChannels, -1});
    auto ret = TensorImpl::matmulTrans(col_, colW, false, true);
    if (!bias.empty()) {
        ASSERT(bias.dim() == 1);
        ASSERT(bias.shape()[0] == outChannels);
        ret += bias; // [batch, outLength, outChannels]
    }
    ret.reshape_({batch, outChannels, outLength});
    return ret;
}

TensorImpl FuncConv2D::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto& input = inputs[0]->data();
  auto& weight = inputs[1]->data();
  auto& bias = inputs[2]->data();
  ASSERT(input.dim() == 4);
  ASSERT(weight.dim() == 4);
  ASSERT(input.shape()[1] == weight.shape()[1]);
  int32_t batch = input.shape()[0];
  int32_t outChannels = weight.shape()[0];
  // int32_t inChannels = weight.shape()[1];
  int32_t height = input.shape()[2];
  int32_t width = input.shape()[3];
  Size2D kernelSize = {weight.shape()[2], weight.shape()[3]};
  auto outH = (height - kernelSize.h + 2 * padding_.h) / stride_.h + 1;
  auto outW = (width - kernelSize.w + 2 * padding_.w) / stride_.w + 1;
  col_ = input.im2col(kernelSize, stride_, padding_);
  auto colW = TensorImpl::reshape(weight, {outChannels, -1});
  auto ret = TensorImpl::matmulTrans(col_, colW, false, true);
  if (!bias.empty()) {
    ASSERT(bias.dim() == 1);
    ASSERT(bias.shape()[0] == outChannels);
    ret += bias;
  }
  ret.reshape_({batch, outChannels, outH, outW});
  return ret;
}

std::vector<TensorImpl> FuncConv1D::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& input = savedTensors[0].data();
  auto& weight = savedTensors[1].data();

  int32_t outChannels = weight.shape()[0];
  Size1D kernelSize {weight.shape()[2]};

  auto gradW = TensorImpl::reshape(grad, {-1, outChannels}); // [N*L, O]

  std::vector<TensorImpl> ret;

  if (savedTensors[0].isRequiresGrad()) {
    auto colW = TensorImpl::reshape(weight, {outChannels, -1}); // [O, I*K]
    auto gradCol = TensorImpl::matmul(gradW, colW);             // [N*L, I*K]
    auto dx = gradCol.col2im1D(input.shape(),
                             kernelSize,
                             stride_,
                             padding_);
    ret.push_back(std::move(dx));
  }

  if (savedTensors[1].isRequiresGrad()) {
    auto gradColW = TensorImpl::matmulTrans(col_, gradW, true, false); // [I*K, O]
    auto dw = TensorImpl::reshape(gradColW.permute(), weight.shape()); // [O, I, K]
    ret.push_back(std::move(dw));
  }

  if (!savedTensors[2].empty() && savedTensors[2].isRequiresGrad()) {
    auto db = TensorImpl::sum(gradW, 0);
    ret.push_back(std::move(db));
  }
  return ret;
}

std::vector<TensorImpl> FuncConv2D::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& input = savedTensors[0].data();
  auto& weight = savedTensors[1].data();
  // auto& bias = savedTensors[2].data();

  int32_t outChannels = weight.shape()[0];
  Size2D kernelSize = {weight.shape()[2], weight.shape()[3]};

  auto gradW = TensorImpl::reshape(grad, {-1, outChannels});
  auto colW = TensorImpl::reshape(weight, {outChannels, -1});

  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    auto gradCol = TensorImpl::matmul(gradW, colW);
    auto dx = gradCol.col2im(input.shape(), kernelSize, stride_, padding_);
    ret.push_back(std::move(dx));
  }
  if (savedTensors[1].isRequiresGrad()) {
    auto gradColW = TensorImpl::matmulTrans(col_, gradW, true, false);
    auto dw = TensorImpl::reshape(gradColW.permute(), weight.shape());
    ret.push_back(std::move(dw));
  }
  if (!savedTensors[2].empty() && savedTensors[2].isRequiresGrad()) {
    auto db = TensorImpl::sum(gradW, 0);
    ret.push_back(std::move(db));
  }
  return ret;
}

TensorImpl FuncLayerNorm::forward(const std::vector<const Tensor*>& inputs) {
  auto& input = inputs[0]->data();
  auto& weight = inputs[1]->data();
  auto& bias = inputs[2]->data();

  auto& shape = input.shape();
  ASSERT(shape.size() == 3);
  viewShape_ = {1, 1, shape[2]};
  Tensor mean;
  Tensor var;
  Tensor inputNorm;
  mean.data() = TensorImpl::mean(input, -1, true);
  var.data() = TensorImpl::var(input, -1, false, true);
  saveForBackward({inputs[0], &mean, &var});
  inputNorm.data() = (input - mean.data()) / TensorImpl::sqrt(var.data() + eps_);
  if (!weight.empty()) {
    inputNorm.data() *= weight.view(viewShape_);
  }
  if (!bias.empty()) {
    inputNorm.data() += bias.view(viewShape_);
  }
  return inputNorm.data();
}

std::vector<TensorImpl> FuncLayerNorm::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& input = savedTensors[0].data();
  auto& weight = savedTensors[1].data();
  auto& inputNorm = savedTensors[3].data();
  auto& inputCentered = savedTensors[4].data();
  auto& std = savedTensors[5].data();

  std::vector<TensorImpl> ret;

  return ret;
}


TensorImpl FuncBatchNorm::forward(const std::vector<const Tensor*>& inputs) {
  auto& input = inputs[0]->data();
  auto& weight = inputs[1]->data();
  auto& bias = inputs[2]->data();

  auto& shape = input.shape();
  ASSERT(shape.size() == 3 || shape.size() == 4);

  if (shape.size() == 3) {
    dims_ = {0, 2};
    viewShape_ = {1, shape[1], 1};
  } else {
    dims_ = {0, 2, 3};
    viewShape_ = {1, shape[1], 1, 1};
  }

  Tensor mean;
  Tensor var;
  if (training_) {
    mean.data() = input.mean(dims_, true);
    var.data() = input.var(dims_, false, true);
    auto varUnbiased = input.var(dims_, true, true);

    if (!runningMean_.empty() && !runningVar_.empty()) {
      runningMean_.data() *= 1.f - momentum_;
      runningMean_.data() += TensorImpl::squeeze(mean.data()) * momentum_;
      runningVar_.data() *= 1.f - momentum_;
      runningVar_.data() += TensorImpl::squeeze(varUnbiased) * momentum_;
    }
  } else {
    if (!runningMean_.empty() && !runningVar_.empty()) {
      mean = runningMean_;
      var = runningVar_;
    } else {
      mean.data() = input.mean(dims_, true);
      var.data() = input.var(dims_, true, true);
    }
  }

  auto inputCentered = Tensor(input - mean.data());
  auto std = Tensor((var.data() + eps_).sqrt());
  auto inputNorm = Tensor(inputCentered / std);

  saveForBackward(inputs);
  saveForBackward({&inputNorm, &inputCentered, &std});

  if (!weight.empty()) {
    inputNorm.data() *= weight.view(viewShape_);
  }
  if (!bias.empty()) {
    inputNorm.data() += bias.view(viewShape_);
  }
  return inputNorm.data();
}

std::vector<TensorImpl> FuncBatchNorm::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto& input = savedTensors[0].data();
  auto& weight = savedTensors[1].data();
  // auto& bias = savedTensors[2].data();
  auto& inputNorm = savedTensors[3].data();
  auto& inputCentered = savedTensors[4].data();
  auto& std = savedTensors[5].data();

  std::vector<TensorImpl> ret;
  // grad of input
  if (savedTensors[0].isRequiresGrad()) {
    auto dInputNorm = grad;
    if (!weight.empty()) {
      dInputNorm *= weight.view(viewShape_);
    }
    int32_t N = 1;
    for (int dim : dims_) {
      N *= input.shape()[dim];
    }
    auto dVar =
        (dInputNorm * inputCentered * -0.5f * std.pow(-3.f)).sum(dims_, true);
    auto dMean = (dInputNorm * -1.f / std).sum(dims_, true) +
                 dVar * (inputCentered * -2.f / (float)N).sum(dims_, true);
    auto dInput = dInputNorm / std + dVar * 2.f * inputCentered / (float)N +
                  dMean / (float)N;
    ret.push_back(std::move(dInput));
  }
  // grad of weight
  if (savedTensors[1].isRequiresGrad()) {
    auto dWeight = (grad * inputNorm).sum(dims_);
    ret.push_back(std::move(dWeight));
  }

  // grad of bias
  if (savedTensors[2].isRequiresGrad()) {
    auto dBias = grad.sum(dims_);
    ret.push_back(std::move(dBias));
  }
  return ret;
}

TensorImpl FuncBCELossWithSigmoid::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  Tensor input = *(inputs[0]);
  Tensor target = *(inputs[1]);
  auto ret = TensorImpl::clampMin(inputs[0]->data(), 0)  - target.data() * input.data() +
       TensorImpl::log(1 + TensorImpl::exp(0 - TensorImpl::abs(input.data())));
  switch (reduction_) {
    case MEAN:
      return ret.mean();
    case SUM:
      return ret.sum();
    default:
      break;
  }
  return ret;
}

std::vector<TensorImpl> FuncBCELossWithSigmoid::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto retGrad = grad * (savedTensors[0] - savedTensors[1]).data();
  switch (reduction_) {
    case MEAN:
      retGrad /= (float)savedTensors[0].numel();
    default:
      break;
  }
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(retGrad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(grad * (0 - savedTensors[0]).data());
  }
  return ret;
}

TensorImpl FuncBCELoss::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto ret = 0 - (inputs[1]->data() * TensorImpl::log(inputs[0]->data().clamp(eps_, 1 - eps_)) +
          (1.0f - inputs[1]->data()) * TensorImpl::log((1.0f - inputs[0]->data()).clamp(eps_, 1 - eps_)));
  switch (reduction_) {
    case MEAN:
      return ret.mean();
    case SUM:
      return ret.sum();
    default:
      break;
  }
  return ret;
}


std::vector<TensorImpl> FuncBCELoss::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto local_grad = (savedTensors[0].data() - savedTensors[1].data())
          / (savedTensors[0].data() * (1.0f - savedTensors[0].data()) + eps_);
  auto retGrad = grad * local_grad;
  switch (reduction_) {
    case MEAN:
      retGrad /= (float)savedTensors[0].numel();
    default:
      break;
  }
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(retGrad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    auto log_y_pred = TensorImpl::log(savedTensors[0].data().clamp(eps_, 1 - eps_));
    auto log_one_minus_y_pred = TensorImpl::log((1.0f - savedTensors[0].data()).clamp(eps_, 1 - eps_));
    auto target_grad = 0 - (log_y_pred - log_one_minus_y_pred);
    ret.push_back(grad * target_grad);
  }
  return ret;
}

TensorImpl FuncMSELoss::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  auto ret = TensorImpl::pow(inputs[0]->data() - inputs[1]->data(), 2);
  switch (reduction_) {
    case MEAN:
      return ret.mean();
    case SUM:
      return ret.sum();
    default:
      break;
  }
  return ret;
}

std::vector<TensorImpl> FuncMSELoss::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto retGrad = grad * 2 * (savedTensors[0].data() - savedTensors[1].data());
  switch (reduction_) {
    case MEAN:
      retGrad /= (float)savedTensors[0].numel();
    default:
      break;
  }
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(retGrad);
  }
  if (savedTensors[1].isRequiresGrad()) {
    ret.push_back(-1 * retGrad);
  }
  return ret;
}

TensorImpl FuncNLLLoss::forward(const std::vector<const Tensor*>& inputs) {
  saveForBackward(inputs);
  ASSERT(inputs[1]->dim() == 1);
  ASSERT(inputs[0]->type() == inputs[1]->type());
  auto batchSize = (int32_t)inputs[0]->shape()[0];
  auto idx = TensorImpl::arange(0, (float)batchSize, 1.f, inputs[0]->device());
  auto ret = -1 * inputs[0]->data().index({idx, inputs[1]->data()});
  switch (reduction_) {
    case MEAN:
      return ret.mean();
    case SUM:
      return ret.sum();
    default:
      break;
  }
  return ret;
}

std::vector<TensorImpl> FuncNLLLoss::backward(const TensorImpl& grad) {
  const auto& savedTensors = getSavedTensors();
  auto batchSize = (int32_t)savedTensors[0].shape()[0];
  auto retGrad = TensorImpl::zeros(savedTensors[0].shape(), grad.device(), savedTensors[0].type());
  auto idx =
      TensorImpl::arange(0, (float)batchSize, 1.f, savedTensors[0].device());
  retGrad.indexPut_({idx, savedTensors[1].data()}, -1.f);
  switch (reduction_) {
    case MEAN:
      retGrad /= (float)batchSize;
    default:
      break;
  }
  std::vector<TensorImpl> ret;
  if (savedTensors[0].isRequiresGrad()) {
    ret.push_back(std::move(retGrad));
  }
  return ret;
}

}  // namespace TinyTorch
