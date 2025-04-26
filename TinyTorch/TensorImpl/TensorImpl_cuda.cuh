/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <cublas_v2.h>

#include "TensorImpl.h"

namespace TinyTorch {

template <Dtype T>
struct CublasDataType;

template <>
struct CublasDataType<Dtype::float32> {
    static constexpr cudaDataType_t value = CUDA_R_32F;
};

template <>
struct CublasDataType<Dtype::bfloat16> {
    static constexpr cudaDataType_t value = CUDA_R_16BF;
};

template <>
struct CublasDataType<Dtype::float16> {
    static constexpr cudaDataType_t value = CUDA_R_16F;
};

struct TINYTORCH_ALIGN(TENSOR_MEM_ALIGN) TensorCudaCtx {
  int32_t dimCount_;
  int32_t elemCount_;
  int32_t shape_[TENSOR_MAX_DIMS];
  int32_t strides_[TENSOR_MAX_DIMS];
  float *data_;
};

class AllocatorCUDA : public Allocator {
 public:
  void allocate(void **ptr, size_t size) override;
  void deallocate(void *ptr) override;
};

class RandomGeneratorCUDA {
 public:
  static void setSeed(unsigned long seed) { seed_ = seed; }
  static unsigned long getSeed() { return seed_; }
  static unsigned long nextSequence() { return sequence_++; }

 private:
  static unsigned long seed_;
  static unsigned long sequence_;
};

class TensorOpsCUDA : public TensorOperations {
 public:
  explicit TensorOpsCUDA(int32_t device = 0, size_t blockSize = 512);
  ~TensorOpsCUDA() override;

  unsigned int getGridSize(size_t n, int32_t batch = 1) const {
    return (unsigned int)((n + (blockSize_ * batch) - 1) /
                          (blockSize_ * batch));
  }
  unsigned int getBlockSize() const { return (unsigned int)blockSize_; }

  cublasHandle_t getCublasHandle();
  static TensorCudaCtx getTensorCtx(const TensorImpl &t);

  TENSOR_OPS_DECLARE(, override)
 protected:
  // op single
  template <typename OP>
  void opSingle_(TensorImpl &t) const;
  template <typename OP>
  TensorImpl opSingle(const TensorImpl &t) const;
  // op pair
  template <typename OP>
  TensorImpl opPair(const TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPair(const TensorImpl &a, float b) const;
  template <typename OP>
  TensorImpl opPair(float a, const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPairScalarFirst(const TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPairScalarSecond(const TensorImpl &a, const TensorImpl &b) const;

  // op pair inplace
  template <typename OP>
  void opPair_(TensorImpl &t, float b) const;
  template <typename OP>
  void opPair_(TensorImpl &t, const TensorImpl &b) const;
  template <typename OP>
  void opPairScalarFirst_(TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  void opPairScalarSecond_(TensorImpl &a, const TensorImpl &b) const;

  // op pair broadcast
  template <typename OP>
  void broadcastImpl(TensorImpl &result, const TensorImpl &a,
                     const TensorImpl &b) const;
  template <typename OP>
  TensorImpl opPairBroadcast(const TensorImpl &a, const TensorImpl &b) const;
  template <typename OP>
  void opPairBroadcast_(TensorImpl &a, const TensorImpl &b) const;

  // reduce
  template <typename OP>
  using KernelFunc = void (*)(float *, const float *, int32_t, int32_t);
  template <typename OP>
  void reduceAllImpl(float *dOutput, const float *dInput, int32_t n, int32_t m,
                     KernelFunc<OP> kernel);
  template <typename OP>
  void reduceAll(float *dOutput, const float *dInput, int32_t n, int32_t m = 1);
  template <typename OP>
  void reduceAllIdx(float *dOutput, const float *dInput, int32_t n,
                    int32_t m = 1);
  template <typename OP>
  void reduceAllLastDim(float *dOutput, const float *dInput, int32_t n,
                        int32_t m = 1);
  template <typename OP>
  std::pair<TensorImpl, TensorImpl> reduceDim(const TensorImpl &t, int32_t dim,
                                              bool keepDims);

  // transpose
  static void transpose2D(float *out, const float *in, int32_t width,
                          int32_t height);
 protected:
  int32_t cudaDeviceIdx_;
  size_t blockSize_;
  cudaDeviceProp deviceProp_{};
  cublasHandle_t blasHandle_ = nullptr;
};

}  // namespace TinyTorch
