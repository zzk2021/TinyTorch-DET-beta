/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <optional>

#include "Tensor.h"
#include "Enums.h"
namespace TinyTorch::nn {



class Module {


 public:
  virtual ~Module() = default;
  virtual std::vector<Tensor *> parameters();
  virtual std::vector<Tensor *> states();
  virtual void resetParameters();
  virtual void zeroGrad();
  virtual std::string name() const { return "Module"; }
  virtual Tensor forward(Tensor &x) { return {}; }
  virtual Tensor forward(Tensor &x1, Tensor &x2) { return {}; }
  virtual Tensor forward(Tensor &x1, Tensor &x2, Tensor &x3) { return {}; }
  virtual Tensor forward(std::vector<Tensor> &x) { return {}; }
  virtual std::vector<Tensor> forward(Tensor &x1, bool many) { return {}; }
  virtual std::vector<Tensor> forward(std::vector<Tensor> &x, bool many) { return {}; }
  template <typename... Args>
  Tensor operator()(Args &&...args) {
    return forward(std::forward<Args>(args)...);
  }
  void registerModules(
      const std::vector<std::reference_wrapper<Module>> &modules) {
    subModules_.reserve(modules.size());
    for (auto module : modules) {
      subModules_.emplace_back(module.get());
    }
  }
  void registerModule(const std::reference_wrapper<Module> &module) {
    subModules_.push_back(module);
  }
  void to(Device device);
  void to(Dtype T);
  void eval() { train(false); }
  void train(bool mode = true) { setTraining(mode); }
    std::string getTopologyText() const {
            std::stringstream ss;
            getTopologyTextHelper(ss, 0);
            return ss.str();
        }
   virtual void getTopologyTextHelper(std::stringstream& ss, int depth) const;
 protected:

  virtual void setTraining(bool mode) { training_ = mode; }

  bool training_ = true;
  std::vector<std::reference_wrapper<Module>> subModules_;

};

class Sequential : public Module {
 public:

  std::string name() const override { return "Sequential"; }
  struct Slice {
    int start;
    int end;

    Slice(std::initializer_list<int> list) {
    if (list.size() != 2) {
      throw std::invalid_argument("Slice must have exactly two integers");
    }
    start = *list.begin();
    end = *(list.begin() + 1);
    }
  };

  template <typename... Modules>
  explicit Sequential(Modules &&...modules) {
    modules_.reserve(sizeof...(Modules));
    pushBack(std::forward<Modules>(modules)...);
  }

  Sequential(std::initializer_list<std::shared_ptr<Module>> modules) {
    modules_.reserve(modules.size());
    for (const auto &module : modules) {
      modules_.emplace_back(module);
    }
  }

  template <typename ModuleType>
  void pushBack(ModuleType &&module) {
    modules_.push_back(
        std::make_shared<ModuleType>(std::forward<ModuleType>(module)));
  }

  void pushBack(const std::shared_ptr<Module> &module) {
    modules_.emplace_back(module);
  }

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;
  size_t getsize() {return modules_.size();};
  Module &operator[](const int index) { return *modules_[index]; }
  Sequential operator[](const Slice& slice) const {
    if (slice.start < 0 || slice.start >= modules_.size()) {
      throw std::out_of_range("Start index out of range");
    }
    if (slice.end < 0 || slice.end > modules_.size()) {
      throw std::out_of_range("End index out of range");
    }
    if (slice.start > slice.end) {
      throw std::invalid_argument("Start index must be <= end index");
    }
    Sequential result;
    result.modules_.reserve(slice.end -  slice.start);
    for (int i = slice.start; i < slice.end; ++i) {
      result.pushBack(modules_[i]);
    }
    return result;
  }

 private:
  void setTraining(bool mode) override;
  template <typename First, typename Second, typename... Rest>
  void pushBack(First &&first, Second &&second, Rest &&...rest) {
    pushBack(std::forward<First>(first));
    pushBack(std::forward<Second>(second), std::forward<Rest>(rest)...);
  }
  void pushBack() {}
  std::vector<std::shared_ptr<Module>> modules_;
    void getTopologyTextHelper(std::stringstream& ss, int depth) const override {
        ss << std::string(depth * 2, ' ') << "|--Sequential" << std::endl;
        for (const auto& module : modules_) {
            module->getTopologyTextHelper(ss, depth + 1); // 合法：派生类访问基类 protected 方法
        }
    }
};


class FlashSelfAttention : public Module {
 public:
  explicit FlashSelfAttention(int32_t inFeatures, int32_t head = 8, AttentionMethod method = AttentionMethod::FalshAttentionV2);
  Tensor forward(Tensor &Q, Tensor &K, Tensor &V) override;
 private:
  int32_t inFeatures_;
  int32_t head_;
  AttentionMethod method_;
};

class Linear : public Module {
 public:
   std::string name() const override { return "Linear"; }
  Linear(int32_t inFeatures, int32_t outFeatures, bool bias = true);
  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }

 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
};

class Flatten : public Module {
 public:
    std::string name() const override { return "Flatten"; }
  explicit Flatten(int32_t startDim = 0, int32_t endDim = -1)
      : startDim_(startDim), endDim_(endDim) {}

  Tensor forward(Tensor &input) override;

 private:
  int32_t startDim_;
  int32_t endDim_;
};

class Relu : public Module {
 public:
    std::string name() const override { return "Relu"; }
  Tensor forward(Tensor &input) override;
};

class LeakyRelu : public Module {
 public:
    std::string name() const override { return "LeakyRelu"; }
  explicit LeakyRelu(float rate=0.1) : rate_(rate) {}
  Tensor forward(Tensor &input) override;
 private:
  float rate_;
};

class Dropout : public Module {
 public:
    std::string name() const override { return "Dropout"; }
  explicit Dropout(float p = 0.5f) : p_(p) {}

  Tensor forward(Tensor &x) override;

 private:
  float p_;
};

class Softmax : public Module {
 public:
    std::string name() const override { return "Softmax"; }
  explicit Softmax(int32_t dim) : dim_(dim) {}

  Tensor forward(Tensor &x) override;

 private:
  int32_t dim_;
};

class LogSoftmax : public Module {
 public:
    std::string name() const override { return "LogSoftmax"; }
  explicit LogSoftmax(int32_t dim) : dim_(dim) {}

  Tensor forward(Tensor &x) override;

 private:
  int32_t dim_;
};

class MaxPool2D : public Module {
 public:
     std::string name() const override { return "MaxPool2D"; }
  explicit MaxPool2D(Size2D kernelSize,
                     std::optional<Size2D> stride = std::nullopt,
                     Size2D padding = 0)
      : kernelSize_(kernelSize),
        stride_(stride.has_value() ? stride.value() : kernelSize),
        padding_(padding) {}

  Tensor forward(Tensor &x) override;

 private:
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
};


class Conv2D : public Module {
 public:
    std::string name() const override { return "Conv2D"; }
  Conv2D(int32_t inFeatures, int32_t outFeatures, Size2D kernelSize,
         Size2D stride = 1, Size2D padding = 0, bool bias = true,
         Dtype fw_type = Dtype::float32,
         Dtype bw_type = Dtype::float32);

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }
  Size2D &kernelsize() {return kernelSize_;}
  int32_t &outfeatures() {return outFeatures_;}
 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  Size2D kernelSize_;
  Size2D stride_;
  Size2D padding_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
  Dtype bw_type_;
  Dtype fw_type_;
};

class Conv1D : public Module {
 public:
    std::string name() const override { return "Conv1D"; }
  Conv1D(int32_t inFeatures, int32_t outFeatures, Size1D kernelSize,
         Size1D stride = 1, Size1D padding = 0, bool bias = true);
  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;
  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }
  Size1D &kernelsize() {return kernelSize_;}
  int32_t &outfeatures() {return outFeatures_;}
 private:
  int32_t inFeatures_;
  int32_t outFeatures_;
  Size1D kernelSize_;
  Size1D stride_;
  Size1D padding_;
  bool useBias_;
  Tensor weights_;
  Tensor bias_;
};

class BatchNorm2D : public Module {
 public:
     std::string name() const override { return "BatchNorm2D"; }
  explicit BatchNorm2D(int32_t numFeatures, float eps = 1e-5,
                       float momentum = 0.1f, bool affine = true,
                       bool trackRunningStats = true);

  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> states() override;
  void resetParameters() override;
  void zeroGrad() override;

  Tensor &weights() { return weights_; }
  Tensor &bias() { return bias_; }

  Tensor &runningMean() { return runningMean_; }
  Tensor &runningVar() { return runningVar_; }

 private:
  int32_t numFeatures_;
  float eps_;
  float momentum_;
  bool affine_;
  bool trackRunningStats_;

  Tensor weights_;
  Tensor bias_;

  Tensor runningMean_;
  Tensor runningVar_;
  int32_t numBatchesTracked_;
};

}  // namespace TinyTorch::nn
