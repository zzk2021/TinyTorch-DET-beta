/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include "Tensor.h"

#include <deque>
#include <set>
#include <unordered_map>

#include "Function.h"
#include "Logger.h"

namespace TinyTorch {

thread_local bool NoGradScope::gradEnabled = true;

Tensor::Tensor(const Array1d &values1d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values1d)) {
  initAutograd(requiresGrad);
}

Tensor::Tensor(const Array2d &values2d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values2d)) {
  initAutograd(requiresGrad);
}

Tensor::Tensor(const Array3d &values3d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values3d)) {
  initAutograd(requiresGrad);
}

Tensor::Tensor(const Array4d &values4d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values4d)) {
  initAutograd(requiresGrad);
}

Tensor::Tensor(const Array5d &values5d, bool requiresGrad)
    : data_(std::make_shared<TensorImpl>(values5d)) {
  initAutograd(requiresGrad);
}

Tensor Tensor::from_slice(std::vector<int> starts, std::vector<int> ends) const{
    std::vector<int32_t> ret_shape;
    for (int i=0; i<starts.size(); i+=1){
        ret_shape.push_back(ends[i] - starts[i]);
    }
    TensorImpl ret = this->data().from_slice(this->data(),starts, ends);
    return Tensor(std::move(ret), this->isRequiresGrad());
}


Slice::Slice(std::initializer_list<int> list) {
    if (list.size() == 0) {
        start = std::nullopt;
        end = std::nullopt;
    } else if (list.size() == 2) {
        auto it = list.begin();
        start = *it;
        end = *(it + 1);
    } else {
        throw std::invalid_argument("Slice must be empty or have exactly two integers");
    }
}

Tensor Tensor::operator[](const Tensor& b) const {
    assert(b.device() == this->device() && "Invalid Device");
    assert(this->shape().size() >= b.shape().size() && "Invalid shape");
    return Function::from_mask(*this, b);
}

Tensor Tensor::operator[](const std::vector<Slice>& slices) const {
    if (slices.size() > this->dim()) {
        throw std::invalid_argument("Number of slices exceeds tensor dimensions");
    }
    std::vector<int> starts;
    std::vector<int> ends;
    for (int i = 0; i < this->dim(); ++i) {
        if (i >= slices.size()) {
            starts.push_back(0);
            ends.push_back(this->shape()[i]);
            continue;
        }
        const Slice& slice = slices[i];
        int dimSize = this->shape()[i];
        int start = slice.start.value_or(0);
        int end = slice.end.value_or(dimSize);
        if (end < 0 ){
            end += (dimSize + 1);
        }
        if (start < 0 || start > dimSize) {
            throw std::out_of_range("Start index out of range in dimension " + std::to_string(i));
        }
        if (end > dimSize) {
            throw std::out_of_range("End index out of range in dimension " + std::to_string(i));
        }
        if (start > end) {
            throw std::invalid_argument("Start must be <= end in dimension " + std::to_string(i));
        }
        starts.push_back(start);
        ends.push_back(end);
    }

    return Function::from_slice(*this, starts, ends);
}

float Tensor::operator[](const std::vector<int>& indices) const {
    if (indices.size() != this->dim()) {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    for (int i = 0; i < this->dim(); ++i) {
        int idx = indices[i];
        int dimSize = this->shape()[i];
        if (idx < 0 || idx >= dimSize) {
            throw std::out_of_range("Index out of range in dimension " + std::to_string(i));
        }
    }
    size_t offset = 0;
    for (int i = 0; i < this->dim(); ++i) {
        offset += indices[i] * this->data_->strides()[i];
    }

    auto retTensor = TensorImpl::scalar(0,this->device(),this->type());
    if (this->type() == Dtype::float32)
        this->data_->ops()->copyOnDevice(retTensor.data(), &this->data_->data()[offset],sizeof(float));
    else if (this->type() == Dtype::bfloat16 || this->type() == Dtype::float16)
        this->data_->ops()->copyOnDevice(retTensor.data(), &this->data_->data()[offset],sizeof(float) / 2);

    return retTensor.toList()[0];
}

Tensor Tensor::shape(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::shape(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::scalar(const float &value, bool requiresGrad) {
  auto ret = TensorImpl::scalar(value);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::scalar(const float &value, bool requiresGrad, Device device, Dtype type) {
  auto ret = TensorImpl::scalar(value, device,type);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::ones(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::ones(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::onesLike(const Tensor &t, bool requiresGrad) {
  auto ret = TensorImpl::onesLike(*t.data_);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::zeros(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::zeros(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::randn(const Shape &shape, bool requiresGrad) {
  auto ret = TensorImpl::randn(shape);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::arange(float start, float stop, float step, bool requiresGrad) {
  auto ret = TensorImpl::arange(start, stop, step);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::linspace(float start, float end, int steps, bool requiresGrad) {
  auto ret = TensorImpl::linspace(start, end, steps);
  return Tensor(std::move(ret), requiresGrad);
}

Tensor Tensor::reshape(const Shape &shape) const {
  return Function::reshape(*this, shape);
}

Tensor Tensor::reshape(const Tensor &t, const Shape &shape) {
  return Function::reshape(t, shape);
}

Tensor Tensor::flatten(int32_t startDim, int32_t endDim) const {
  return Function::flatten(*this, startDim, endDim);
}

Tensor Tensor::flatten(const Tensor &t, int32_t startDim, int32_t endDim) {
  return Function::flatten(t, startDim, endDim);
}

Tensor Tensor::unflatten(int32_t dim, const std::vector<int32_t> &sizes) const {
  return Function::unflatten(*this, dim, sizes);
}

Tensor Tensor::unflatten(const Tensor &t, int32_t dim,
                         const std::vector<int32_t> &sizes) {
  return Function::unflatten(t, dim, sizes);
}

Tensor Tensor::operator+(const Tensor &other) const {
  return Function::add(*this, other);
}

Tensor Tensor::operator-(const Tensor &other) const {
  return Function::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor &other) const {
  return Function::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor &other) const {
  return Function::div(*this, other);
}

Tensor Tensor::operator+(const float &other) const {
  return Function::add(*this, scalar(other, false, this->device(),this->type()));
}

Tensor Tensor::operator-(const float &other) const {
  return Function::sub(*this, scalar(other, false, this->device(),this->type()));
}

Tensor Tensor::operator*(const float &other) const {
  return Function::mul(*this, scalar(other, false, this->device(),this->type()));
}

Tensor Tensor::operator/(const float &other) const {
  return Function::div(*this, scalar(other, false, this->device(),this->type()));
}

Tensor operator+(const float &other, const Tensor &obj) {
  return Function::add(Tensor::scalar(other, false, obj.device(),obj.type()), obj);
}

Tensor operator-(const float &other, const Tensor &obj) {
  return Function::sub(Tensor::scalar(other, false, obj.device(),obj.type()), obj);
}

Tensor operator*(const float &other, const Tensor &obj) {
  return Function::mul(Tensor::scalar(other, false, obj.device(),obj.type()), obj);
}

Tensor operator/(const float &other, const Tensor &obj) {
  return Function::div(Tensor::scalar(other, false, obj.device(),obj.type()), obj);
}

void Tensor::operator+=(const Tensor &other) {
  *this = Function::add(*this, other);
}

void Tensor::operator-=(const Tensor &other) {
  *this = Function::sub(*this, other);
}

void Tensor::operator*=(const Tensor &other) {
  *this = Function::mul(*this, other);
}

void Tensor::operator/=(const Tensor &other) {
  *this = Function::div(*this, other);
}

void Tensor::operator+=(const float &other) {
  *this = Function::add(*this, scalar(other));
}

void Tensor::operator-=(const float &other) {
  *this = Function::sub(*this, scalar(other));
}

void Tensor::operator*=(const float &other) {
  *this = Function::mul(*this, scalar(other));
}

void Tensor::operator/=(const float &other) {
  *this = Function::div(*this, scalar(other));
}

Tensor Tensor::sin() const { return Function::sin(*this); }

Tensor Tensor::cos() const { return Function::cos(*this); }

Tensor Tensor::pow(const float &exp) const { return Function::pow(*this, exp); }

Tensor Tensor::pow(const Tensor &exp) const {
  return Function::pow(*this, exp);
}

Tensor Tensor::sum() const { return Function::sum(*this); }

Tensor Tensor::mean() const { return Function::mean(*this); }

Tensor Tensor::var() const { return Function::pow((*this - Function::mean(*this)) , 2).mean() ; }

Tensor Tensor::squeeze(int32_t dim) const {
  return Function::squeeze(*this, dim);
}

Tensor Tensor::unsqueeze(int32_t dim) const {
  return Function::unsqueeze(*this, dim);
}

Tensor::Tensor(TensorImpl &&data, bool requiresGrad,
               const std::shared_ptr<Function> &gradFunc)
    : data_(std::make_shared<TensorImpl>()) {
  *data_ = std::move(data);
  initAutograd(requiresGrad, gradFunc);
}

bool Tensor::isRequiresGrad() const { return requiresGrad_; }

void Tensor::setRequiresGrad(bool require) { initAutograd(require); }

void Tensor::backward(const Tensor &grad) {
  if (requiresGrad_) {
    gradMeta_->backward(grad);
  }
}

const Tensor &Tensor::getGrad() const {
  if (isLeaf()) {
    return gradMeta_->grad_;
  }
  LOGE(
      "The 'grad' attribute of a Tensor that is not a leaf Tensor is being "
      "accessed");
  static Tensor empty;
  return empty;
}

void Tensor::setGrad(const Tensor &grad) {
  if (requiresGrad_) {
    gradMeta_->grad_.data_ = grad.data_;
  }
}

void Tensor::zeroGrad() {
  if (requiresGrad_) {
    gradMeta_->grad_.data_->fill_(0.f);
  }
}

void Tensor::initAutograd(bool requiresGrad,
                          const std::shared_ptr<Function> &gradFunc) {
  requiresGrad_ = requiresGrad;
  if (requiresGrad_) {
    gradMeta_ = std::make_shared<AutogradMeta>();
    gradMeta_->setGradFunc(gradFunc);
    *gradMeta_->grad_.data_ =
        TensorImpl::shape(data_->shape(), data_->device());

    if (isLeaf()) {
      gradMeta_->gradLeaf_ = std::make_shared<FuncLeaf>();
      gradMeta_->gradLeaf_->setOwner(gradMeta_);
    } else {
      gradMeta_->gradLeaf_ = nullptr;
    }
  } else {
    gradMeta_ = nullptr;
  }
}

bool Tensor::isLeaf() const {
  return requiresGrad_ && gradMeta_->gradFunc_ == nullptr;
}

std::shared_ptr<Function> Tensor::getGradFunc() const {
  if (requiresGrad_) {
    return isLeaf() ? gradMeta_->gradLeaf_ : gradMeta_->gradFunc_;
  }
  return nullptr;
}

void AutogradMeta::setGradFunc(const std::shared_ptr<Function> &gradFunc) {
  gradFunc_ = gradFunc;
  if (gradFunc_) {
    gradFunc_->setOwner(shared_from_this());
  }
}

void AutogradMeta::backward(const Tensor &grad) {
  if (gradFunc_ == nullptr) {
    LOGE("error call backward: gradFunc == nullptr");
    return;
  }

  if (grad.empty()) {
    if (grad_.data_->dim() != 0) {
      LOGE("error call backward: input grad must not be omitted");
      return;
    }
  } else {
    if (grad.data_->shape() != grad_.data_->shape()) {
      LOGE("error call backward: input grad shape mismatch");
      return;
    }
  }

  if (backwardGraph_.empty()) {
    buildBackwardGraph();
  }

  std::unordered_map<std::shared_ptr<Function>, TensorImpl> inputs = {
      {gradFunc_,
       grad.empty() ? TensorImpl::scalar(1.f, grad_.device()) : *grad.data_}};
  for (auto &currFunc : backwardGraph_) {
    auto outputs = currFunc->callBackward(inputs[currFunc]);

    for (int i = 0; i < currFunc->nextFuncs().size(); i++) {
      auto &nextFunc = currFunc->nextFuncs()[i];

      if (inputs.find(nextFunc) != inputs.end()) {
        inputs[nextFunc] += outputs[i];
      } else {
        inputs[nextFunc] = std::move(outputs[i]);
      }
    }
  }
}

void AutogradMeta::buildBackwardGraph() {
  std::unordered_map<std::shared_ptr<Function>, int> deps;
  std::deque<std::shared_ptr<Function>> q;

  std::set traversed = {gradFunc_};
  q.push_back(gradFunc_);
  while (!q.empty()) {
    const auto curr = q.front();
    q.pop_front();
    for (const auto &next : curr->nextFuncs()) {
      deps[next] += 1;
      if (traversed.find(next) == traversed.end()) {
        q.push_back(next);
        traversed.insert(next);
      }
    }
  }

  q.push_back(gradFunc_);
  while (!q.empty()) {
    const auto currFunc = q.front();
    backwardGraph_.push_back(currFunc);
    q.pop_front();
    for (const auto &nextFunc : currFunc->nextFuncs()) {
      deps[nextFunc] -= 1;
      if (deps[nextFunc] == 0) {
        q.push_back(nextFunc);
      }
    }
  }
}

void Tensor::print() const {
  const auto& shape = this->shape();
  size_t rows = shape[0];
  size_t cols = shape[1];
  for (size_t i = 0; i < (rows < 5 ? rows : 5); ++i) {
    printf("  [");
    for (size_t j = 0; j < (cols < 10 ? cols : 10); ++j) {
      printf("%f", this->toList()[i * cols + j]);
      if (j < cols - 1) {
        printf(", ");
      }
    }
    printf("]\n");
  }
}

}  // namespace TinyTorch
