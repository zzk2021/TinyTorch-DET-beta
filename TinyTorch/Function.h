/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once
#include "Objectdetection/header.h"
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Loss.h"

namespace TinyTorch {

enum FunctionType {
  Function_Leaf,
  Function_Add,
  Function_Sub,
  Function_Mul,
  Function_Div,
  Function_Sin,
  Function_Cos,
  Function_Pow,
  Function_PowScalar,
  Function_Sum,
  Function_Mean,
  Function_Max,
  Function_Relu,
  Function_LeakyRelu,
  Function_Flatten,
  Function_UnFlatten,
  Function_FlashAttention,
  Function_UpSample,
  Function_ConCat,
  Function_Slice,
  Function_Mask,
  Function_Attention,
  Function_Squeeze,
  Function_Unsqueeze,
  Function_Reshape,
  Function_Linear,
  Function_Dropout,
  Function_Softmax,
  Function_LogSoftmax,
  Function_MaxPool2D,
  Function_Conv2D,
  Function_Conv1D,
  Function_BatchNorm,
  Function_LayerNorm,
  Function_MSELoss,
  Function_BCELoss,
  Function_BCELossWithSigmoid,
  Function_NLLLoss,
  Function_ChangeType,
  // object detection support
  OBJDECT_EXPLORE_FUNCTIONTYPE()
};

class Function;
class TensorImpl;
class Tensor;
class Function : public std::enable_shared_from_this<Function> {
 public:
  OBJDECT_EXPLORE_FUNCTION();
  static Tensor add(const Tensor& a, const Tensor& b);
  static Tensor from_slice(const Tensor& a,
                           const std::vector<int>& start,
                           const std::vector<int>& end);
  static Tensor from_mask(const Tensor& a, const Tensor& b);
  static Tensor concat(const Tensor& a, const Tensor& b, int32_t dim);
  static Tensor sub(const Tensor& a, const Tensor& b);
  static Tensor mul(const Tensor& a, const Tensor& b);
  static Tensor div(const Tensor& a, const Tensor& b);
  static Tensor sin(const Tensor& a);
  static Tensor cos(const Tensor& a);
  static Tensor pow(const Tensor& a, const float& b);
  static Tensor pow(const Tensor& a, const Tensor& b);
  static Tensor sum(const Tensor& a);
  static Tensor mean(const Tensor& a);
  static Tensor max(const Tensor& a, int32_t dim_ ,bool keepdim);
  static Tensor relu(const Tensor& input);
  static Tensor leakyrelu(const Tensor& input, float rate = 0.1);
  static Tensor flatten(const Tensor& input, int32_t startDim, int32_t endDim);
  static Tensor upsample(const Tensor& input, int32_t scale_factor);
  static Tensor changetype(const Tensor& input, Dtype T);
  static Tensor unflatten(const Tensor& input, int32_t dim,
                          const std::vector<int32_t>& sizes);

  static Tensor squeeze(const Tensor& input, int32_t dim = -1);
  static Tensor unsqueeze(const Tensor& input, int32_t dim);
  static Tensor reshape(const Tensor& input, const Shape& shape);
  static Tensor linear(const Tensor& input, const Tensor& weight,
                       const Tensor& bias);
  static Tensor flashattention(const Tensor& Q,const Tensor& K,
   const Tensor& V, int32_t head);
  static Tensor dropout(const Tensor& input, float p = 0.5f,bool training = true);
  static Tensor softmax(const Tensor& input, int32_t dim);
  static Tensor sigmoid(const Tensor& input);
  static Tensor logSoftmax(const Tensor& input, int32_t dim);
  static Tensor maxPool2d(const Tensor& input, Size2D kernelSize,
                          std::optional<Size2D> stride = std::nullopt,
                          Size2D padding = 0);

  static Tensor conv2d(const Tensor& input, const Tensor& weight,
                       const Tensor& bias = {}, Size2D stride = 1,
                       Size2D padding = 0);

  static Tensor conv1d(const Tensor& input, const Tensor& weight,
                       const Tensor& bias = {}, Size1D stride = 1,
                       Size1D padding = 0);

  static Tensor layerNorm(const Tensor& input, const Tensor& weight,
                          const Tensor& bias, float eps = 1e-5);

  static Tensor batchNorm(const Tensor& input, Tensor& runningMean,
                          Tensor& runningVar, const Tensor& weight,
                          const Tensor& bias, bool training = false,
                          float momentum = 0.1f, float eps = 1e-5);

  static Tensor nllloss(const Tensor& input, const Tensor& target,
                        LossReduction reduction = MEAN);

  static Tensor mseLoss(const Tensor& input, const Tensor& target,
                        LossReduction reduction = MEAN);

  static Tensor bceLoss(const Tensor& input, const Tensor& target,
                        LossReduction reduction = MEAN);

  static Tensor bceLossWithSigmoid(const Tensor& input, const Tensor& target,
                        LossReduction reduction = MEAN);

  virtual ~Function() = default;

  const std::vector<std::shared_ptr<Function>>& nextFuncs() const {
    return nextFuncs_;
  }

  virtual FunctionType type() const = 0;
  std::string typeString() const { return funcTypeToString_[type()]; }
  Tensor callForward(const std::vector<const Tensor*>& inputs);
  std::vector<TensorImpl> callBackward(const TensorImpl& grad);
  void setOwner(const std::shared_ptr<AutogradMeta>& meta) { owner_ = meta; }

 protected:
  virtual TensorImpl forward(const std::vector<const Tensor*>& inputs) = 0;
  virtual std::vector<TensorImpl> backward(const TensorImpl& grad) = 0;

  void saveForBackward(const std::vector<const Tensor*>& tensors) {
    if (!NoGradScope::isGradEnabled()) {
      return;
    }
    savedTensors_.reserve(savedTensors_.size() + tensors.size());
    for (const auto& t : tensors) {
      savedTensors_.push_back(*t);
    }
  }
  std::vector<Tensor>& getSavedTensors() { return savedTensors_; };

  std::weak_ptr<AutogradMeta> owner_;
  std::vector<Tensor> savedTensors_;
  std::vector<std::shared_ptr<Function>> nextFuncs_;

 private:
  static std::unordered_map<FunctionType, std::string> funcTypeToString_;
};



#define DEFINE_FUNCTION_MEMBERS(TYPE)                                    \
  FunctionType type() const override { return TYPE; }                    \
  TensorImpl forward(const std::vector<const Tensor*>& inputs) override; \
  std::vector<TensorImpl> backward(const TensorImpl& grad) override;

class FuncLeaf : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Leaf)
};

class FuncUpSample : public Function {
 public:
  FuncUpSample(int32_t scale_factor)
        : scale_factor_(scale_factor){}
  DEFINE_FUNCTION_MEMBERS(Function_UpSample)
 private:
  int32_t scale_factor_;
};


class FuncAdd : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Add)
};

class FuncConCat : public Function {
 public:
    FuncConCat(int32_t dim, int32_t a_shape)
     : dim_(dim), a_shape_(a_shape){}
  DEFINE_FUNCTION_MEMBERS(Function_ConCat)
 private:
  int32_t dim_;
  int32_t a_shape_;
};

class FuncSlice : public Function {
 public:
    FuncSlice(Shape a_shape, std::vector<int> start, std::vector<int> end)
     : a_shape_(a_shape), start_(std::move(start)), end_(std::move(end)){}
  DEFINE_FUNCTION_MEMBERS(Function_Slice)
 private:
  Shape a_shape_;
  std::vector<int> start_;
  std::vector<int> end_;
};

class FuncMask : public Function {
 public:
    FuncMask(Shape a_shape)
     : a_shape_(std::move(a_shape)){}
  DEFINE_FUNCTION_MEMBERS(Function_Mask)
 private:
  TensorImpl indice_;
  Shape a_shape_;
};

class FuncSub : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Sub)
};

class FuncMul : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Mul)
};

class FuncDiv : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Div)
};

class FuncSin : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Sin)
};

class FuncCos : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Cos)
};

class FuncPow : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Pow)
};

class FuncPowScalar : public Function {
 public:
  explicit FuncPowScalar(float exp) : exp_(exp) {}
  DEFINE_FUNCTION_MEMBERS(Function_PowScalar)

 private:
  float exp_;
};

class FuncSum : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Sum)
};

class FuncMean : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Mean)
};

class FuncMax : public Function {
 public:
  FuncMax(int32_t dim, bool keep_dim)
    : dim_(dim), keep_dim_(keep_dim) {}
  DEFINE_FUNCTION_MEMBERS(Function_Max)
 private:
  int32_t dim_;
  bool keep_dim_;
  TensorImpl maxIndices_;
};

class FuncMin : public Function {
 public:
  FuncMin(int32_t dim, bool keep_dim)
    : dim_(dim), keep_dim_(keep_dim) {}
  DEFINE_FUNCTION_MEMBERS(Function_Max)
 private:
  int32_t dim_;
  bool keep_dim_;
  TensorImpl maxIndices_;
};

class FuncRelu : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Relu)
};

class FuncLeakyRelu : public Function {
 public:
  FuncLeakyRelu(float rate)
      :  rate_(rate) {}
  DEFINE_FUNCTION_MEMBERS(Function_LeakyRelu)
  private:
    float rate_;
};

class FuncFlatten : public Function {
 public:
  FuncFlatten(int32_t startDim, int32_t endDim)
      : startDim_(startDim), endDim_(endDim) {}
  DEFINE_FUNCTION_MEMBERS(Function_Flatten)

 private:
  int32_t startDim_;
  int32_t endDim_;
};

class FuncUnFlatten : public Function {
 public:
  FuncUnFlatten(int32_t dim, const std::vector<int32_t>& sizes)
      : dim_(dim), sizes_(sizes) {}
  DEFINE_FUNCTION_MEMBERS(Function_UnFlatten)

 private:
  int32_t dim_;
  std::vector<int32_t> sizes_;
};

class FuncSqueeze : public Function {
 public:
  explicit FuncSqueeze(int32_t dim = -1) : dim_(dim) {}
  DEFINE_FUNCTION_MEMBERS(Function_Squeeze)

 private:
  int32_t dim_;
};

class FuncUnsqueeze : public Function {
 public:
  explicit FuncUnsqueeze(int32_t dim) : dim_(dim) {}
  DEFINE_FUNCTION_MEMBERS(Function_Unsqueeze)
 private:
  int32_t dim_;
};

class FuncReshape : public Function {
 public:
  explicit FuncReshape(const Shape& shape) : shape_(shape) {}
  DEFINE_FUNCTION_MEMBERS(Function_Reshape)
 private:
  Shape shape_;
};

class FuncLinear : public Function {
 public:
  DEFINE_FUNCTION_MEMBERS(Function_Linear)
};

class FuncFlashAttention : public Function {
 public:
  explicit FuncFlashAttention(int32_t head): head_(head){}
  DEFINE_FUNCTION_MEMBERS(Function_FlashAttention)
 private:
  int32_t head_;
};

class FuncAttention : public Function {
 public:
  explicit FuncAttention(int32_t head): head_(head){}
  DEFINE_FUNCTION_MEMBERS(Function_Attention)
 private:
  int32_t head_;
};

class FuncDropout : public Function {
 public:
  FuncDropout(float p, bool training) : p_(p), training_(training) {}
  DEFINE_FUNCTION_MEMBERS(Function_Dropout)

 private:
  float p_;
  bool training_;
  TensorImpl mask_;
};

class FuncSoftmax : public Function {
 public:
  explicit FuncSoftmax(int32_t dim) : dim_(dim) {}
  DEFINE_FUNCTION_MEMBERS(Function_Softmax)

 private:
  int32_t dim_;
  TensorImpl forwardOutput_;
};

class FuncSigmoid : public Function {
 public:
  explicit FuncSigmoid()= default;
  DEFINE_FUNCTION_MEMBERS(Function_Softmax)
 private:
  TensorImpl forwardOutput_;
};


class FuncChangeType : public Function {
 public:
  explicit FuncChangeType(Dtype T, Dtype ori_T) : T_(T), Ori_T_(ori_T) {}
  DEFINE_FUNCTION_MEMBERS(Function_ChangeType)
 private:
  Dtype  T_;
  Dtype  Ori_T_;
};

class FuncLogSoftmax : public Function {
 public:
  explicit FuncLogSoftmax(int32_t dim) : dim_(dim) {}
  DEFINE_FUNCTION_MEMBERS(Function_LogSoftmax)

 private:
  int32_t dim_;
  TensorImpl forwardOutput_;
};

class FuncMaxPool2D : public Function {
 public:
  FuncMaxPool2D(Size2D kernelSize, Size2D stride, Size2D padding)
      : kernelSize_(kernelSize), stride_(stride), padding_(padding) {}
  DEFINE_FUNCTION_MEMBERS(Function_MaxPool2D)

 private:
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
  TensorImpl maxIndices_;
};

class FuncConv2D : public Function {
 public:
  explicit FuncConv2D(Size2D stride, Size2D padding)
      : stride_(stride), padding_(padding) {}
  DEFINE_FUNCTION_MEMBERS(Function_Conv2D)

 private:
  Size2D stride_;
  Size2D padding_;
  TensorImpl col_;
};

class FuncConv1D : public Function {
 public:
  explicit FuncConv1D(Size1D stride, Size1D padding)
      : stride_(stride), padding_(padding) {}
  DEFINE_FUNCTION_MEMBERS(Function_Conv2D)
 private:
  Size1D stride_;
  Size1D padding_;
  TensorImpl col_;
};

class FuncLayerNorm : public Function {
 public:
  explicit FuncLayerNorm(float eps = 1e-5)
      : eps_(eps) {}
  DEFINE_FUNCTION_MEMBERS(Function_LayerNorm)

 private:
  float eps_ = 1e-5;
  std::vector<int32_t> viewShape_;
};

class FuncBatchNorm : public Function {
 public:
  explicit FuncBatchNorm(Tensor& runningMean, Tensor& runningVar,
                         float momentum, float eps, bool training)
      : runningMean_(runningMean),
        runningVar_(runningVar),
        momentum_(momentum),
        eps_(eps),
        training_(training) {}
  DEFINE_FUNCTION_MEMBERS(Function_BatchNorm)

 private:
  Tensor& runningMean_;
  Tensor& runningVar_;
  std::vector<int32_t> dims_;
  std::vector<int32_t> viewShape_;

  float momentum_;
  float eps_;
  bool training_;
};

class FuncMSELoss : public Function {
 public:
  explicit FuncMSELoss(LossReduction reduction) : reduction_(reduction) {}
  DEFINE_FUNCTION_MEMBERS(Function_MSELoss)

 private:
  LossReduction reduction_;
};

class FuncBCELossWithSigmoid : public Function {
 public:
  explicit FuncBCELossWithSigmoid(LossReduction reduction) : reduction_(reduction) {}
  DEFINE_FUNCTION_MEMBERS(Function_BCELossWithSigmoid)
 private:
  LossReduction reduction_;
  float eps_ = 1e-8;
};


class FuncBCELoss : public Function {
 public:
  explicit FuncBCELoss(LossReduction reduction) : reduction_(reduction) {}
  DEFINE_FUNCTION_MEMBERS(Function_BCELoss)
 private:
  LossReduction reduction_;
  float eps_ = 1e-8;
};

class FuncNLLLoss : public Function {
 public:
  explicit FuncNLLLoss(LossReduction reduction) : reduction_(reduction) {}
  DEFINE_FUNCTION_MEMBERS(Function_NLLLoss)

 private:
  LossReduction reduction_;
};

}  // namespace TinyTorch
