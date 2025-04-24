/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cassert>
#include <cfloat>
#include <iostream>

#include "TensorImpl_cpu.h"
#include "TensorImpl_cuda.cuh"
#include "TensorImpl_cuda.inc"

namespace TinyTorch {

const char* curandGetErrorString(curandStatus_t status);
const char* cublasGetErrorString(cublasStatus_t status);

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line "         \
                << __LINE__ << ": " << cudaGetErrorString(err) << " (" << err \
                << ")" << std::endl;                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

#define CURAND_CHECK(call)                                               \
  do {                                                                   \
    curandStatus_t err = call;                                           \
    if (err != CURAND_STATUS_SUCCESS) {                                  \
      std::cerr << "CURAND error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << curandGetErrorString(err) << " (" \
                << err << ")" << std::endl;                              \
      abort();                                                           \
    }                                                                    \
  } while (0)

#define CUBLAS_CHECK(call)                                               \
  do {                                                                   \
    cublasStatus_t err = call;                                           \
    if (err != CUBLAS_STATUS_SUCCESS) {                                  \
      std::cerr << "CUBLAS error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << cublasGetErrorString(err) << " (" \
                << err << ")" << std::endl;                              \
      abort();                                                           \
    }                                                                    \
  } while (0)

#define CUDA_KERNEL_CHECK()                                                   \
  do {                                                                        \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                 \
      std::cerr << "CUDA kernel error in file '" << __FILE__ << "' in line "  \
                << __LINE__ << ": " << cudaGetErrorString(err) << " (" << err \
                << ")" << std::endl;                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

static std::random_device _r;
unsigned long RandomGeneratorCUDA::seed_ = _r();
unsigned long RandomGeneratorCUDA::sequence_ = 0;

void* AllocatorCPU::allocatePinned(size_t size) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaMallocHost(&ptr, size));
  return ptr;
}

void AllocatorCPU::deallocatePinned(void* ptr) {
  CUDA_CHECK(cudaFreeHost(ptr));
}

void AllocatorCUDA::allocate(void** ptr, size_t size) {
  CUDA_CHECK(cudaMalloc(ptr, size));
}

void AllocatorCUDA::deallocate(void* ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

TensorOpsCUDA::TensorOpsCUDA(int32_t device, size_t blockSize)
    : cudaDeviceIdx_(device), blockSize_(blockSize) {
  CUDA_CHECK(cudaSetDevice(cudaDeviceIdx_));
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp_, cudaDeviceIdx_));

  if (blockSize_ > deviceProp_.maxThreadsPerBlock) {
    blockSize_ = deviceProp_.maxThreadsPerBlock;
  }

  allocator_.setBaseAllocator(std::make_shared<AllocatorCUDA>());
}

TensorOpsCUDA::~TensorOpsCUDA() {
  allocator_.clear();
  if (blasHandle_) {
    cublasDestroy(blasHandle_);
  }
}

cublasHandle_t TensorOpsCUDA::getCublasHandle() {
  if (blasHandle_ == nullptr) {
    cublasCreate(&blasHandle_);
  }
  return blasHandle_;
}

TensorCudaCtx TensorOpsCUDA::getTensorCtx(const TensorImpl& t) {
  TensorCudaCtx ret{};
  ret.dimCount_ = t.dimCount_;
  ret.elemCount_ = t.elemCount_;
  memcpy(ret.shape_, t.shape_.data(), t.dimCount_ * sizeof(int32_t));
  memcpy(ret.strides_, t.strides_.data(), t.dimCount_ * sizeof(int32_t));
  ret.data_ = t.data_;
  return ret;
}

template <typename OP>
void TensorOpsCUDA::opSingle_(TensorImpl& t) const {
  kSingleOp_<OP>
      <<<getGridSize(t.elemCount_), getBlockSize()>>>(t.data_, t.elemCount_);
  CUDA_KERNEL_CHECK();
}

template <typename OP>
TensorImpl TensorOpsCUDA::opSingle(const TensorImpl& t) const {
  auto result = TensorImpl::shape(t.shape(), t.device_);
  kSingleOp<OP><<<getGridSize(t.elemCount_), getBlockSize()>>>(
      result.data_, t.data_, t.elemCount_);
  CUDA_KERNEL_CHECK();
  return result;
}

template <typename OP>
TensorImpl TensorOpsCUDA::opPair(const TensorImpl& a,
                                 const TensorImpl& b) const {
  auto result = TensorImpl::shape(a.shape(), a.device_);
  kPairOp<OP><<<getGridSize(result.elemCount_), getBlockSize()>>>(
      result.data_, a.data_, b.data_, result.elemCount_);
  CUDA_KERNEL_CHECK();
  return result;
}

template <typename OP>
TensorImpl TensorOpsCUDA::opPair(const TensorImpl& a, float b) const {
  auto result = TensorImpl::shape(a.shape(), a.device_);
  kPairScalarSecondOp<OP><<<getGridSize(a.elemCount_), getBlockSize()>>>(
      result.data_, a.data_, b, a.elemCount_);
  CUDA_KERNEL_CHECK();
  return result;
}

template <typename OP>
TensorImpl TensorOpsCUDA::opPair(float a, const TensorImpl& b) const {
  auto result = TensorImpl::shape(b.shape(), b.device_);
  kPairScalarFirstOp<OP><<<getGridSize(b.elemCount_), getBlockSize()>>>(
      result.data_, a, b.data_, b.elemCount_);
  CUDA_KERNEL_CHECK();
  return result;
}

template <typename OP>
TensorImpl TensorOpsCUDA::opPairScalarFirst(const TensorImpl& a,
                                            const TensorImpl& b) const {
  auto result = TensorImpl::shape(b.shape(), b.device_);
  kPairScalarFirstOp<OP><<<getGridSize(result.elemCount_), getBlockSize()>>>(
      result.data_, a.data_, b.data_, result.elemCount_);
  CUDA_KERNEL_CHECK();
  return result;
}

template <typename OP>
TensorImpl TensorOpsCUDA::opPairScalarSecond(const TensorImpl& a,
                                             const TensorImpl& b) const {
  auto result = TensorImpl::shape(a.shape(), a.device_);
  kPairScalarSecondOp<OP><<<getGridSize(result.elemCount_), getBlockSize()>>>(
      result.data_, a.data_, b.data_, result.elemCount_);
  CUDA_KERNEL_CHECK();
  return result;
}

template <typename OP>
void TensorOpsCUDA::opPair_(TensorImpl& t, float b) const {
  kPairScalarSecondOp_<OP>
      <<<getGridSize(t.elemCount_), getBlockSize()>>>(t.data_, b, t.elemCount_);
  CUDA_KERNEL_CHECK();
}

template <typename OP>
void TensorOpsCUDA::opPair_(TensorImpl& t, const TensorImpl& b) const {
  kPairOp_<OP><<<getGridSize(t.elemCount_), getBlockSize()>>>(t.data_, b.data_,
                                                              t.elemCount_);
  CUDA_KERNEL_CHECK();
}

template <typename OP>
void TensorOpsCUDA::opPairScalarFirst_(TensorImpl& a,
                                       const TensorImpl& b) const {
  auto result = TensorImpl::shape(b.shape_, b.device_);
  kPairScalarFirstOp<OP><<<getGridSize(result.elemCount_), getBlockSize()>>>(
      result.data_, a.data_, b.data_, result.elemCount_);
  CUDA_KERNEL_CHECK();
  a = std::move(result);
}

template <typename OP>
void TensorOpsCUDA::opPairScalarSecond_(TensorImpl& a,
                                        const TensorImpl& b) const {
  kPairScalarSecondOp_<OP><<<getGridSize(a.elemCount_), getBlockSize()>>>(
      a.data_, b.data_, a.elemCount_);
  CUDA_KERNEL_CHECK();
}

template <typename OP>
void TensorOpsCUDA::broadcastImpl(TensorImpl& result, const TensorImpl& a,
                                  const TensorImpl& b) const {
  // fast broadcast with a
  if (b.elemCount_ == result.elemCount_) {
    if (isLeadingOnes(a.shape())) {
      kBroadcastOpFast<OP, true, true>
          <<<getGridSize(result.elemCount_), getBlockSize()>>>(
              result.data_, a.data_, b.data_, a.elemCount_, result.elemCount_);
      CUDA_KERNEL_CHECK();
      return;
    }

    if (isTrailingOnes(a.shape())) {
      kBroadcastOpFast<OP, false, true>
          <<<getGridSize(result.elemCount_), getBlockSize()>>>(
              result.data_, a.data_, b.data_, result.elemCount_ / a.elemCount_,
              result.elemCount_);
      CUDA_KERNEL_CHECK();
      return;
    }
  }

  // fast broadcast with b
  if (a.elemCount_ == result.elemCount_) {
    if (isLeadingOnes(b.shape())) {
      kBroadcastOpFast<OP, true, false>
          <<<getGridSize(result.elemCount_), getBlockSize()>>>(
              result.data_, a.data_, b.data_, b.elemCount_, result.elemCount_);
      CUDA_KERNEL_CHECK();
      return;
    }

    if (isTrailingOnes(b.shape())) {
      kBroadcastOpFast<OP, false, false>
          <<<getGridSize(result.elemCount_), getBlockSize()>>>(
              result.data_, a.data_, b.data_, result.elemCount_ / b.elemCount_,
              result.elemCount_);
      CUDA_KERNEL_CHECK();
      return;
    }
  }

  const auto ctxA = getTensorCtx(a);
  const auto ctxB = getTensorCtx(b);
  const auto ctxC = getTensorCtx(result);
  kBroadcastOpCommon<OP><<<getGridSize(result.elemCount_), getBlockSize()>>>(
      ctxC, ctxA, ctxB, result.elemCount_);
  CUDA_KERNEL_CHECK();
}

template <typename OP>
TensorImpl TensorOpsCUDA::opPairBroadcast(const TensorImpl& a,
                                          const TensorImpl& b) const {
  Shape retShape;
  auto comp = checkShapeCompatible(a.shape(), b.shape(), retShape);
  if (comp == ShapeCompatible_Error) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return {};
  }

  if (comp == ShapeCompatible_SameShape) {
    return opPair<OP>(a, b);
  }

  auto result = TensorImpl::shape(retShape, a.device_);
  broadcastImpl<OP>(result, a, b);
  return result;
}

template <typename OP>
void TensorOpsCUDA::opPairBroadcast_(TensorImpl& a, const TensorImpl& b) const {
  Shape retShape;
  auto comp = checkShapeCompatible(a.shape(), b.shape(), retShape);
  if (comp == ShapeCompatible_Error) {
    error(__FUNCTION__, TensorError_ShapeNotAligned);
    return;
  }
  if (comp == ShapeCompatible_SameShape) {
    opPair_<OP>(a, b);
    return;
  }

  auto result = TensorImpl::shape(retShape, a.device_);
  broadcastImpl<OP>(result, a, b);
  a = std::move(result);
}

template <typename OP>
void TensorOpsCUDA::reduceAllImpl(float* dOutput, const float* dInput,
                                  int32_t n, int32_t m, KernelFunc<OP> kernel) {
  auto blocks = getGridSize(n);

  float* dTmp = nullptr;
  allocate(reinterpret_cast<void**>(&dTmp), m * blocks * sizeof(float));

  kernel<<<m * blocks, getBlockSize()>>>(dTmp, dInput, n, m);
  CUDA_KERNEL_CHECK();
  while (blocks > 1) {
    auto currBlocks = blocks;
    blocks = getGridSize(currBlocks);
    kReduceAll<OP><<<m * blocks, getBlockSize()>>>(dTmp, dTmp, currBlocks, m);
    CUDA_KERNEL_CHECK();
  }
  copyOnDevice(dOutput, dTmp, m * sizeof(float));
  deallocate(dTmp);
}

template <typename OP>
void TensorOpsCUDA::reduceAll(float* dOutput, const float* dInput, int32_t n,
                              int32_t m) {
  reduceAllImpl<OP>(dOutput, dInput, n, m, kReduceAll<OP>);
}

template <typename OP>
void TensorOpsCUDA::reduceAllIdx(float* dOutput, const float* dInput, int32_t n,
                                 int32_t m) {
  reduceAllImpl<OP>(dOutput, dInput, n, m, kReduceAllIdx<OP>);
}

template <typename OP>
void TensorOpsCUDA::reduceAllLastDim(float* dOutput, const float* dInput,
                                     int32_t n, int32_t m) {
  reduceAllImpl<OP>(dOutput, dInput, n, m, kReduceAllLastDim<OP>);
}

template <typename OP>
std::pair<TensorImpl, TensorImpl> TensorOpsCUDA::reduceDim(const TensorImpl& t,
                                                           int32_t dim,
                                                           bool keepDims) {
  if (dim < 0) {
    dim += t.dimCount_;
  }
  if (dim < 0 || dim >= t.dimCount_) {
    error(__FUNCTION__, TensorError_InvalidAxis);
    return {};
  }

  const auto retShape = getReduceShape(t, dim, false);
  auto values = TensorImpl::shape(retShape, t.device_);
  auto indices = TensorImpl::shape(retShape, t.device_);

  if (dim == t.dimCount_ - 1) {
    kReduceLastDim<OP><<<getGridSize(t.elemCount_), getBlockSize()>>>(
        values.data_, indices.data_, t.data_, t.shape_[dim], values.elemCount_);
  } else {
    auto ctxT = getTensorCtx(t);
    auto ctxValues = getTensorCtx(values);
    kReduceDim<OP><<<getGridSize(t.elemCount_), getBlockSize()>>>(
        ctxValues, indices.data_, ctxT, dim, ctxValues.elemCount_);
  }
  CUDA_KERNEL_CHECK();

  if (keepDims) {
    const auto shapeKeepDims = getReduceShape(t, dim, true);
    values.reshape_(shapeKeepDims);
    indices.reshape_(shapeKeepDims);
  }
  return {values, indices};
}

void TensorOpsCUDA::transpose2D(float* out, const float* in, int32_t width,
                                int32_t height) {
  dim3 blockSize(TRANSPOSE_TILE_DIM, TRANSPOSE_TILE_DIM);
  dim3 gridSize((width + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                (height + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);
  kTranspose<<<gridSize, blockSize>>>(out, in, width, height);
  CUDA_KERNEL_CHECK();
}

void TensorOpsCUDA::allocate(void** ptr, size_t size) {
  allocator_.allocate(ptr, size);
}

void TensorOpsCUDA::deallocate(void* ptr) { allocator_.deallocate(ptr); }

void TensorOpsCUDA::copyHostToDevice(void* dst, const void* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void TensorOpsCUDA::copyOnDevice(void* dst, const void* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}

void TensorOpsCUDA::copyDeviceToHost(void* dst, const void* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void TensorOpsCUDA::fillConstant_(float* dst, float val, size_t count) {
  kFillConstant<<<getGridSize(count, 4), getBlockSize()>>>(dst, val, count);
  CUDA_KERNEL_CHECK();
}

void TensorOpsCUDA::fillConstant_(TensorImpl& t, float val) {
  kFillConstant<<<getGridSize(t.elemCount_, 4), getBlockSize()>>>(t.data_, val,
                                                                  t.elemCount_);
  CUDA_KERNEL_CHECK();
}

void TensorOpsCUDA::fillLinSpace_(float* dst, float start, float step,
                                  size_t count) {
  kFillLinSpace<<<getGridSize(count, 4), getBlockSize()>>>(dst, start, step,
                                                           count);
  CUDA_KERNEL_CHECK();
}

void TensorOpsCUDA::fillRandUniform_(TensorImpl& t, float min, float max) {
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  kFillRandUniform<<<getGridSize(t.elemCount_, 4), getBlockSize()>>>(
      t.data_, min, max, seed, seq, t.elemCount_);
  CUDA_KERNEL_CHECK();
}

void TensorOpsCUDA::fillRandNormal_(TensorImpl& t) {
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  kFillRandNormal<<<getGridSize(t.elemCount_, 4), getBlockSize()>>>(
      t.data_, 0.f, 1.f, seed, seq, t.elemCount_);
  CUDA_KERNEL_CHECK();
}

void TensorOpsCUDA::fillRandBernoulli_(TensorImpl& t, float p) {
  auto seed = RandomGeneratorCUDA::getSeed();
  auto seq = RandomGeneratorCUDA::nextSequence();
  kFillRandBernoulli<<<getGridSize(t.elemCount_, 4), getBlockSize()>>>(
      t.data_, p, seed, seq, t.elemCount_);
  CUDA_KERNEL_CHECK();
}

TensorImpl TensorOpsCUDA::add(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return opPairScalarFirst<OpCudaAdd>(a, b);
  }
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaAdd>(a, b);
  }
  return opPairBroadcast<OpCudaAdd>(a, b);
}

TensorImpl TensorOpsCUDA::sub(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return opPairScalarFirst<OpCudaSub>(a, b);
  }
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaSub>(a, b);
  }
  return opPairBroadcast<OpCudaSub>(a, b);
}

TensorImpl TensorOpsCUDA::mul(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return opPairScalarFirst<OpCudaMul>(a, b);
  }
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaMul>(a, b);
  }
  return opPairBroadcast<OpCudaMul>(a, b);
}

TensorImpl TensorOpsCUDA::div(const TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    return opPairScalarFirst<OpCudaDiv>(a, b);
  }
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaDiv>(a, b);
  }
  return opPairBroadcast<OpCudaDiv>(a, b);
}

TensorImpl TensorOpsCUDA::pow(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaPow>(a, b);
  }
  return opPairBroadcast<OpCudaPow>(a, b);
}

TensorImpl TensorOpsCUDA::add(const TensorImpl& a, const float& b) {
  return opPair<OpCudaAdd>(a, b);
}

TensorImpl TensorOpsCUDA::sub(const TensorImpl& a, const float& b) {
  return opPair<OpCudaSub>(a, b);
}

TensorImpl TensorOpsCUDA::mul(const TensorImpl& a, const float& b) {
  return opPair<OpCudaMul>(a, b);
}

TensorImpl TensorOpsCUDA::div(const TensorImpl& a, const float& b) {
  return opPair<OpCudaDiv>(a, b);
}

TensorImpl TensorOpsCUDA::pow(const TensorImpl& a, const float& b) {
  return opPair<OpCudaPow>(a, b);
}

TensorImpl TensorOpsCUDA::add(const float& a, const TensorImpl& b) {
  return opPair<OpCudaAdd>(a, b);
}

TensorImpl TensorOpsCUDA::sub(const float& a, const TensorImpl& b) {
  return opPair<OpCudaSub>(a, b);
}

TensorImpl TensorOpsCUDA::mul(const float& a, const TensorImpl& b) {
  return opPair<OpCudaMul>(a, b);
}

TensorImpl TensorOpsCUDA::div(const float& a, const TensorImpl& b) {
  return opPair<OpCudaDiv>(a, b);
}

void TensorOpsCUDA::add_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    opPairScalarFirst_<OpCudaAdd>(a, b);
    return;
  }
  if (b.dimCount_ == 0) {
    opPairScalarSecond_<OpCudaAdd>(a, b);
    return;
  }
  opPairBroadcast_<OpCudaAdd>(a, b);
}

void TensorOpsCUDA::sub_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    opPairScalarFirst_<OpCudaSub>(a, b);
    return;
  }
  if (b.dimCount_ == 0) {
    opPairScalarSecond_<OpCudaSub>(a, b);
    return;
  }
  opPairBroadcast_<OpCudaSub>(a, b);
}

void TensorOpsCUDA::mul_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    opPairScalarFirst_<OpCudaMul>(a, b);
    return;
  }
  if (b.dimCount_ == 0) {
    opPairScalarSecond_<OpCudaMul>(a, b);
    return;
  }
  opPairBroadcast_<OpCudaMul>(a, b);
}

void TensorOpsCUDA::div_(TensorImpl& a, const TensorImpl& b) {
  if (a.dimCount_ == 0) {
    opPairScalarFirst_<OpCudaDiv>(a, b);
    return;
  }
  if (b.dimCount_ == 0) {
    opPairScalarSecond_<OpCudaDiv>(a, b);
    return;
  }
  opPairBroadcast_<OpCudaDiv>(a, b);
}

void TensorOpsCUDA::add_(TensorImpl& a, const float& b) {
  opPair_<OpCudaAdd>(a, b);
}

void TensorOpsCUDA::sub_(TensorImpl& a, const float& b) {
  opPair_<OpCudaSub>(a, b);
}

void TensorOpsCUDA::mul_(TensorImpl& a, const float& b) {
  opPair_<OpCudaMul>(a, b);
}

void TensorOpsCUDA::div_(TensorImpl& a, const float& b) {
  opPair_<OpCudaDiv>(a, b);
}

TensorImpl TensorOpsCUDA::eq(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaEq>(a, b);
  }
  return opPairBroadcast<OpCudaEq>(a, b);
}

TensorImpl TensorOpsCUDA::ne(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaNe>(a, b);
  }
  return opPairBroadcast<OpCudaNe>(a, b);
}

TensorImpl TensorOpsCUDA::ge(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaGe>(a, b);
  }
  return opPairBroadcast<OpCudaGe>(a, b);
}

TensorImpl TensorOpsCUDA::gt(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaGt>(a, b);
  }
  return opPairBroadcast<OpCudaGt>(a, b);
}

TensorImpl TensorOpsCUDA::le(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaLe>(a, b);
  }
  return opPairBroadcast<OpCudaLe>(a, b);
}

TensorImpl TensorOpsCUDA::lt(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaLt>(a, b);
  }
  return opPairBroadcast<OpCudaLt>(a, b);
}

TensorImpl TensorOpsCUDA::maximum(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaMax>(a, b);
  }
  return opPairBroadcast<OpCudaMax>(a, b);
}

TensorImpl TensorOpsCUDA::minimum(const TensorImpl& a, const TensorImpl& b) {
  if (b.dimCount_ == 0) {
    return opPairScalarSecond<OpCudaMin>(a, b);
  }
  return opPairBroadcast<OpCudaMin>(a, b);
}

TensorImpl TensorOpsCUDA::eq(const TensorImpl& a, const float& b) {
  return opPair<OpCudaEq>(a, b);
}

TensorImpl TensorOpsCUDA::ne(const TensorImpl& a, const float& b) {
  return opPair<OpCudaNe>(a, b);
}

TensorImpl TensorOpsCUDA::ge(const TensorImpl& a, const float& b) {
  return opPair<OpCudaGe>(a, b);
}

TensorImpl TensorOpsCUDA::gt(const TensorImpl& a, const float& b) {
  return opPair<OpCudaGt>(a, b);
}

TensorImpl TensorOpsCUDA::le(const TensorImpl& a, const float& b) {
  return opPair<OpCudaLe>(a, b);
}

TensorImpl TensorOpsCUDA::lt(const TensorImpl& a, const float& b) {
  return opPair<OpCudaLt>(a, b);
}

TensorImpl TensorOpsCUDA::maximum(const TensorImpl& a, const float& b) {
  return opPair<OpCudaMax>(a, b);
}

TensorImpl TensorOpsCUDA::minimum(const TensorImpl& a, const float& b) {
  return opPair<OpCudaMin>(a, b);
}

void TensorOpsCUDA::sin_(TensorImpl& t) { opSingle_<OpCudaSin_>(t); }

void TensorOpsCUDA::cos_(TensorImpl& t) { opSingle_<OpCudaCos_>(t); }

void TensorOpsCUDA::sqrt_(TensorImpl& t) { opSingle_<OpCudaSqrt_>(t); }

void TensorOpsCUDA::tanh_(TensorImpl& t) { opSingle_<OpCudaTanh_>(t); }

void TensorOpsCUDA::exp_(TensorImpl& t) { opSingle_<OpCudaExp_>(t); }

void TensorOpsCUDA::log_(TensorImpl& t) { opSingle_<OpCudaLog_>(t); }

TensorImpl TensorOpsCUDA::sin(const TensorImpl& t) {
  return opSingle<OpCudaSin>(t);
}

TensorImpl TensorOpsCUDA::cos(const TensorImpl& t) {
  return opSingle<OpCudaCos>(t);
}

TensorImpl TensorOpsCUDA::sqrt(const TensorImpl& t) {
  return opSingle<OpCudaSqrt>(t);
}

TensorImpl TensorOpsCUDA::tanh(const TensorImpl& t) {
  return opSingle<OpCudaTanh>(t);
}

TensorImpl TensorOpsCUDA::exp(const TensorImpl& t) {
  return opSingle<OpCudaExp>(t);
}

TensorImpl TensorOpsCUDA::log(const TensorImpl& t) {
  return opSingle<OpCudaLog>(t);
}

void TensorOpsCUDA::clampMin_(TensorImpl& t, float min) {
  opPair_<OpCudaMax>(t, min);
}

void TensorOpsCUDA::clampMax_(TensorImpl& t, float max) {
  opPair_<OpCudaMin>(t, max);
}

void TensorOpsCUDA::clamp_(TensorImpl& t, float min, float max) {
  kClamp_<<<getGridSize(t.elemCount_), getBlockSize()>>>(t.data_, min, max,
                                                         t.elemCount_);
  CUDA_KERNEL_CHECK();
}

TensorImpl TensorOpsCUDA::clampMin(const TensorImpl& t, float min) {
  return opPair<OpCudaMax>(t, min);
}

TensorImpl TensorOpsCUDA::clampMax(const TensorImpl& t, float max) {
  return opPair<OpCudaMin>(t, max);
}

TensorImpl TensorOpsCUDA::clamp(const TensorImpl& t, float min, float max) {
  auto ret = TensorImpl::shape(t.shape_, t.device_);
  kClamp<<<getGridSize(t.elemCount_), getBlockSize()>>>(ret.data_, t.data_, min,
                                                        max, t.elemCount_);
  CUDA_KERNEL_CHECK();
  return ret;
}

TensorImpl TensorOpsCUDA::min(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCudaReduceMin>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCUDA::max(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCudaReduceMax>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCUDA::sum(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCudaReduceSum>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCUDA::mean(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return t;
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCudaReduceSum>(ret.data_, t.data_, t.elemCount_);
  const auto r = 1.f / static_cast<float>(t.elemCount_);
  mul_(ret, r);
  return ret;
}

TensorImpl TensorOpsCUDA::var(const TensorImpl& t, bool unbiased) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  const auto meanVal = mean(t);
  const auto squaredDiff = TensorImpl::shape({t.elemCount_}, t.device_);
  kSquaredDiff<<<getGridSize(t.elemCount_), getBlockSize()>>>(
      squaredDiff.data_, t.data_, meanVal.data_, t.elemCount_);

  auto ret = TensorImpl::scalar(t.device_);
  reduceAll<OpCudaReduceSum>(ret.data_, squaredDiff.data_, t.elemCount_);

  const auto n = static_cast<float>(t.elemCount_);
  auto r = 1.f / n;
  if (unbiased) {
    r *= n / (n - 1.f);
  }
  mul_(ret, r);
  return ret;
}

TensorImpl TensorOpsCUDA::argmin(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAllIdx<OpCudaReduceMin>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

TensorImpl TensorOpsCUDA::argmax(const TensorImpl& t) {
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }
  auto ret = TensorImpl::scalar(t.device_);
  reduceAllIdx<OpCudaReduceMax>(ret.data_, t.data_, t.elemCount_);
  return ret;
}

std::pair<TensorImpl, TensorImpl> TensorOpsCUDA::min(const TensorImpl& t,
                                                     int32_t dim,
                                                     bool keepDims) {
  if (t.dimCount_ == 0) {
    return {t, TensorImpl::scalar(0, t.device_)};
  }
  return reduceDim<OpCudaReduceMin>(t, dim, keepDims);
}



std::pair<TensorImpl, TensorImpl> TensorOpsCUDA::max(const TensorImpl& t,
                                                     int32_t dim,
                                                     bool keepDims) {
  if (t.dimCount_ == 0) {
    return {t, TensorImpl::scalar(0, t.device_)};
  }
  return reduceDim<OpCudaReduceMax>(t, dim, keepDims);
}

TensorImpl TensorOpsCUDA::sum(const TensorImpl& t,
                              const std::vector<int32_t>& dims, bool keepDims) {
  FixedVector<uint8_t> inAxis{};
  for (int32_t d : dims) {
    if (d < 0) {
      d += t.dimCount_;
    }
    if (d < 0 || d >= t.dimCount_) {
      error(__FUNCTION__, TensorError_InvalidAxis);
      return {};
    }
    inAxis.data[d] = 1;
  }
  if (t.dimCount_ == 0) {
    return t;
  }

  auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  if (dims.size() == 1) {
    auto d = dims[0];
    if (d < 0) {
      d += t.dimCount_;
    }

    // first dim
    if (d == 0) {
      const auto dimSize = t.shape_.front();
      const auto tmp = TensorImpl::shape(t.shape_, t.device_);
      transpose2D(tmp.data_, t.data_, ret.elemCount_, dimSize);
      reduceAllLastDim<OpCudaReduceSum>(ret.data_, tmp.data_, dimSize,
                                        ret.elemCount_);
      return ret;
    }

    // last dim
    if (d == t.dimCount_ - 1) {
      const auto dimSize = t.shape_.back();
      reduceAllLastDim<OpCudaReduceSum>(ret.data_, t.data_, dimSize,
                                        ret.elemCount_);
      return ret;
    }
  }

  auto ctxT = getTensorCtx(t);
  fillConstant_(ret, 0);
  kReduceSum<<<getGridSize(t.elemCount_), getBlockSize()>>>(
      ret.data_, ctxT, inAxis, t.elemCount_);
  CUDA_KERNEL_CHECK();
  return ret;
}

TensorImpl TensorOpsCUDA::mean(const TensorImpl& t,
                               const std::vector<int32_t>& dims,
                               bool keepDims) {
  auto ret = sum(t, dims, keepDims);
  if (!ret.empty()) {
    auto reduceSize = (float)t.elemCount_ / (float)ret.elemCount_;
    auto r = 1.f / reduceSize;
    mul_(ret, r);
  }
  return ret;
}

TensorImpl TensorOpsCUDA::var(const TensorImpl& t,
                              const std::vector<int32_t>& dims, bool unbiased,
                              bool keepDims) {
  FixedVector<uint8_t> inAxis{};
  for (int32_t d : dims) {
    if (d < 0) {
      d += t.dimCount_;
    }
    if (d < 0 || d >= t.dimCount_) {
      error(__FUNCTION__, TensorError_InvalidAxis);
      return {};
    }
    inAxis.data[d] = 1;
  }
  if (t.dimCount_ == 0) {
    return TensorImpl::scalar(0, t.device_);
  }

  auto retShape = getReduceShape(t, inAxis, keepDims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  auto meanTensor = mean(t, dims, true);
  fillConstant_(ret, 0);

  auto ctxT = getTensorCtx(t);
  kReduceVar<<<getGridSize(t.elemCount_), getBlockSize()>>>(
      ret.data_, ctxT, meanTensor.data_, inAxis, t.elemCount_);
  CUDA_KERNEL_CHECK();

  auto reduceSize = (float)t.elemCount_ / (float)ret.elemCount_;
  auto r = 1.f / reduceSize;
  if (unbiased) {
    r *= reduceSize / (reduceSize - 1.f);
  }
  mul_(ret, r);
  return ret;
}

TensorImpl TensorOpsCUDA::permute(const TensorImpl& t,
                                  const std::vector<int32_t>& dims) {
  auto retShape = t.shape_;
  reorderIndices(retShape.data(), dims);
  auto ret = TensorImpl::shape(retShape, t.device_);

  auto ctxT = getTensorCtx(t);
  auto ctxRet = getTensorCtx(ret);

  auto* dimsDataPtr = (FixedVector<int32_t>*)dims.data();
  kPermute<<<getGridSize(t.elemCount_), getBlockSize()>>>(
      ctxRet, ctxT, *dimsDataPtr, t.elemCount_);
  CUDA_KERNEL_CHECK();
  return ret;
}

TensorImpl TensorOpsCUDA::transpose2D(const TensorImpl& t) {
  auto ret = TensorImpl::shape({t.shape_[1], t.shape_[0]}, t.device_);
  transpose2D(ret.data_, t.data_, t.shape_[1], t.shape_[0]);
  return ret;
}

TensorImpl TensorOpsCUDA::index(
    const TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];
  Shape retShape = {fistDim};
  for (auto i = len; i < t.dimCount_; i++) {
    retShape.push_back(t.shape_[i]);
  }
  auto retTensor = TensorImpl::shape(retShape, t.device_);

  FixedVector<float*> indicesData{};
  for (int32_t i = 0; i < len; i++) {
    indicesData.data[i] = indices[i].get().data_;
  }
  auto ctxT = getTensorCtx(t);
  kIndex<<<getGridSize(fistDim), getBlockSize()>>>(
      retTensor.data_, ctxT, indicesData, dimStride, len, fistDim);
  CUDA_KERNEL_CHECK();
  return retTensor;
}

void TensorOpsCUDA::indexPut_(
    TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices, float val) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];

  FixedVector<float*> indicesData{};
  for (int32_t i = 0; i < len; i++) {
    indicesData.data[i] = indices[i].get().data_;
  }
  auto ctxT = getTensorCtx(t);
  kIndexPut<<<getGridSize(fistDim), getBlockSize()>>>(
      ctxT, indicesData, dimStride, len, val, fistDim);
  CUDA_KERNEL_CHECK();
}

void TensorOpsCUDA::indexPut_(
    TensorImpl& t,
    const std::vector<std::reference_wrapper<TensorImpl>>& indices,
    const TensorImpl& val) {
  auto len = (int32_t)indices.size();
  auto fistDim = (int32_t)indices[0].get().elemCount_;
  auto dimStride = t.strides_[len - 1];
  assert(val.elemCount_ == dimStride * fistDim);

  FixedVector<float*> indicesData{};
  for (int32_t i = 0; i < len; i++) {
    indicesData.data[i] = indices[i].get().data_;
  }
  auto ctxT = getTensorCtx(t);
  kIndexPut<<<getGridSize(fistDim), getBlockSize()>>>(
      ctxT, indicesData, dimStride, len, val.data_, fistDim);
  CUDA_KERNEL_CHECK();
}

TensorImpl TensorOpsCUDA::im2col(const TensorImpl& t, Size2D kernel,
                                 Size2D stride, Size2D padding) {
  // this: [C, H, W], [N, C, H, W]
  assert(t.dimCount_ == 3 || t.dimCount_ == 4);
  int32_t batch = (t.dimCount_ == 4) ? t.shape_[0] : 1;
  int32_t channels = (t.dimCount_ == 4) ? t.shape_[1] : t.shape_[0];
  int32_t height = (t.dimCount_ == 4) ? t.shape_[2] : t.shape_[1];
  int32_t width = (t.dimCount_ == 4) ? t.shape_[3] : t.shape_[2];
  int32_t outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  int32_t outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  int32_t colH = outH * outW;
  int32_t colW = channels * kernel.h * kernel.w;
  auto retTensor = TensorImpl::shape({batch * colH, colW}, t.device_);

  int32_t imStride = t.strides_[0];
  int totalElements = batch * outH * outW * channels * kernel.h * kernel.w;
  kIm2Col<<<getGridSize(totalElements), getBlockSize()>>>(
      retTensor.data_, t.data_, batch, channels, height, width, outH, outW,
      kernel.h, kernel.w, stride.h, stride.w, padding.h, padding.w, imStride,
      colH, colW);
  CUDA_KERNEL_CHECK();
  return retTensor;
}

TensorImpl TensorOpsCUDA::col2im(const TensorImpl& t, const Shape& shape,
                                 Size2D kernel, Size2D stride, Size2D padding) {
  // shape: [C, H, W], [N, C, H, W]
  assert(shape.size() == 3 || shape.size() == 4);
  int32_t batch = (shape.size() == 4) ? shape[0] : 1;
  int32_t channels = (shape.size() == 4) ? shape[1] : shape[0];
  int32_t height = (shape.size() == 4) ? shape[2] : shape[1];
  int32_t width = (shape.size() == 4) ? shape[3] : shape[2];

  auto outH = (height - kernel.h + 2 * padding.h) / stride.h + 1;
  auto outW = (width - kernel.w + 2 * padding.w) / stride.w + 1;

  // int32_t colH = outH * outW;
  int32_t colW = channels * kernel.h * kernel.w;

  auto retTensor = TensorImpl::zeros(shape, t.device_);

  auto imStride = retTensor.strides_[0];
  int totalElements = batch * channels * height * width;
  kCol2Im<<<getGridSize(totalElements), getBlockSize()>>>(
      retTensor.data_, t.data_, batch, channels, height, width, outH, outW,
      kernel.h, kernel.w, stride.h, stride.w, padding.h, padding.w, imStride,
      colW);
  CUDA_KERNEL_CHECK();
  return retTensor;
}

TensorImpl TensorOpsCUDA::dot(const TensorImpl& a, const TensorImpl& b) {
  auto ret = TensorImpl::scalar(0.f, a.device_);
  auto sharedMemSize = getBlockSize() * sizeof(float);
  kDot<<<getGridSize(a.elemCount_), getBlockSize(), sharedMemSize>>>(
      ret.data_, a.data_, b.data_, a.elemCount_);
  CUDA_KERNEL_CHECK();
  return ret;
}

TensorImpl TensorOpsCUDA::triangle(const TensorImpl& t, int32_t diagonal,
                                    bool lower) {
  auto ret = TensorImpl::shape(t.shape_, t.device_);
  const auto rows = t.shape_[0];
  const auto cols = t.shape_[1];

  dim3 blockSize(WARP_SIZE, WARP_SIZE);
  dim3 gridSize((cols + blockSize.x - 1) / blockSize.x,
                (rows + blockSize.y - 1) / blockSize.y);

  if (lower) {
    kTriangle<true>
        <<<gridSize, blockSize>>>(ret.data_, t.data_, rows, cols, diagonal);
  } else {
    kTriangle<false>
        <<<gridSize, blockSize>>>(ret.data_, t.data_, rows, cols, diagonal);
  }
  CUDA_KERNEL_CHECK();
  return ret;
}

void TensorOpsCUDA::gemm(float* c, const float* a, const float* b, int32_t m,
                         int32_t k, int32_t n, bool transA, bool transB) {
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  int lda = transA ? m : k;
  int ldb = transB ? k : n;
  int ldc = n;

  const float alpha = 1.f;
  const float beta = 0.f;

  CUBLAS_CHECK(cublasSgemm(getCublasHandle(), opB, opA, n, m, k, &alpha, b, ldb,
                           a, lda, &beta, c, ldc));
}

TensorImpl TensorOpsCUDA::upsample_forward(const TensorImpl& a , int32_t scale_factor){
  TensorImpl ret = TensorImpl::shape({a.shape_[0], a.shape_[1], static_cast<int>(a.shape_[2]*scale_factor),
                                            static_cast<int>(a.shape_[3]*scale_factor)}, a.device());
  int32_t N = a.numel();
  int32_t h = a.shape_[2];
  int32_t w = a.shape_[3];
  if (scale_factor == 2 && N >= 256){
    dim3 grid(N / kBlockSize, 1);
    dim3 block(kBlockSize, 1);
    UpsampleNearest2D2XForward<<<grid, block>>>(N, a.data_, h,
                                                w, ret.data_);
  }
  else{}
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
  }
  return ret;
}

TensorImpl TensorOpsCUDA::upsample_backward(const TensorImpl& a , int32_t scale_factor){
  TensorImpl ret = TensorImpl::shape({a.shape_[0], a.shape_[1], static_cast<int>(a.shape_[2]*scale_factor),
                                            static_cast<int>(a.shape_[3]*scale_factor)}, a.device());
  int32_t N = a.numel();
  int32_t h = a.shape_[2];
  int32_t w = a.shape_[3];
  if (scale_factor == 2 && N >= 256){
    dim3 grid(N / kBlockSize, 1);
    dim3 block(kBlockSize, 1);
    UpsampleNearest2D2XBackward<<<grid, block>>>(N, a.data_, h,
                                                 w, ret.data_);
  }
  else
      {}
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
  }
  return ret;
}

const char* curandGetErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown cuRAND error";
}

const char* cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "Unknown cuBLAS error";
}

}  // namespace TinyTorch
