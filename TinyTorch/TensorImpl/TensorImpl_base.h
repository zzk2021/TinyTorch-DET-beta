/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once

#include <vector>

#include "../Allocator.h"
#include "../Logger.h"

namespace TinyTorch {

#define TENSOR_MAX_DIMS 8
#define TENSOR_MEM_ALIGN 16

#ifdef _MSC_VER
#define TINYTORCH_ALIGN(N) __declspec(align(N))
#else
#define TINYTORCH_ALIGN(N) __attribute__((aligned(N)))
#endif

enum class Device { CPU, CUDA };
enum class Dtype {
  float32 = 0,
  bfloat16,
  float16,
  float8_e4m3,
  float8_e5m2
};


struct TINYTORCH_ALIGN(TENSOR_MEM_ALIGN) Size2D {
  Size2D(int32_t n) : h(n), w(n) {}
  Size2D(int32_t h, int32_t w) : h(h), w(w) {}

  int32_t h = 0;
  int32_t w = 0;
};

template <typename T>
struct TINYTORCH_ALIGN(TENSOR_MEM_ALIGN) FixedVector {
  T data[TENSOR_MAX_DIMS]{};
};

typedef std::vector<int32_t> Shape;
typedef std::vector<float> Array1d;
typedef std::vector<std::vector<float>> Array2d;
typedef std::vector<std::vector<std::vector<float>>> Array3d;
typedef std::vector<std::vector<std::vector<std::vector<float>>>> Array4d;

typedef enum TensorError_ {
  TensorError_None = 0,
  TensorError_EmptyTensor,
  TensorError_InvalidShape,
  TensorError_InvalidAxis,
  TensorError_InvalidSections,
  TensorError_InvalidDevice,
  TensorError_ShapeNotAligned,
  TensorError_DeviceNotAligned,
  TensorError_NotSupport,
} TensorError;

typedef enum ShapeCompatible_ {
  ShapeCompatible_Error = 0,
  ShapeCompatible_SameShape,
  ShapeCompatible_Broadcast,
} ShapeCompatible;


#define TENSOR_OPS_DECLARE(_H, _T)                                             \
  /* memory */                                                                 \
  _H void allocate(void** ptr, size_t size) _T;                                \
  _H void deallocate(void* ptr) _T;                                            \
  _H void copyHostToDevice(void* dst, const void* src, size_t count) _T;       \
  _H void copyOnDevice(void* dst, const void* src, size_t count) _T;           \
  _H void convertTypeOnDevice(void* dst, void* src, size_t count, Dtype Ti     \
                   ,Dtype To) _T;                                              \
  _H void copyDeviceToHost(void* dst, const void* src, size_t count) _T;       \
                                                                               \
  /* fill */                                                                   \
  _H void fillConstant_(float* dst, float val, size_t count) _T;               \
  _H void fillConstant_(TensorImpl& t, float val) _T;                          \
  _H void fillLinSpace_(float* dst, float start, float step, size_t count) _T; \
  _H void fillRandUniform_(TensorImpl& t, float min, float max) _T;            \
  _H void fillRandNormal_(TensorImpl& t) _T;                                   \
  _H void fillRandBernoulli_(TensorImpl& t, float p) _T;                       \
                                                                               \
  /* math */                                                                   \
  _H TensorImpl add(const TensorImpl& a, const TensorImpl& b) _T;              \
  _H TensorImpl sub(const TensorImpl& a, const TensorImpl& b) _T;              \
  _H TensorImpl mul(const TensorImpl& a, const TensorImpl& b) _T;              \
  _H TensorImpl div(const TensorImpl& a, const TensorImpl& b) _T;              \
  _H TensorImpl pow(const TensorImpl& a, const TensorImpl& b) _T;              \
                                                                               \
  _H TensorImpl add(const TensorImpl& a, const float& b) _T;                   \
  _H TensorImpl sub(const TensorImpl& a, const float& b) _T;                   \
  _H TensorImpl mul(const TensorImpl& a, const float& b) _T;                   \
  _H TensorImpl div(const TensorImpl& a, const float& b) _T;                   \
  _H TensorImpl pow(const TensorImpl& a, const float& b) _T;                   \
                                                                               \
  _H TensorImpl add(const float& a, const TensorImpl& b) _T;                   \
  _H TensorImpl sub(const float& a, const TensorImpl& b) _T;                   \
  _H TensorImpl mul(const float& a, const TensorImpl& b) _T;                   \
  _H TensorImpl div(const float& a, const TensorImpl& b) _T;                   \
                                                                               \
  _H void add_(TensorImpl& a, const TensorImpl& b) _T;                         \
  _H void sub_(TensorImpl& a, const TensorImpl& b) _T;                         \
  _H void mul_(TensorImpl& a, const TensorImpl& b) _T;                         \
  _H void div_(TensorImpl& a, const TensorImpl& b) _T;                         \
                                                                               \
  _H void add_(TensorImpl& a, const float& b) _T;                              \
  _H void sub_(TensorImpl& a, const float& b) _T;                              \
  _H void mul_(TensorImpl& a, const float& b) _T;                              \
  _H void div_(TensorImpl& a, const float& b) _T;                              \
                                                                               \
  _H TensorImpl eq(const TensorImpl& a, const TensorImpl& b) _T;               \
  _H TensorImpl ne(const TensorImpl& a, const TensorImpl& b) _T;               \
  _H TensorImpl ge(const TensorImpl& a, const TensorImpl& b) _T;               \
  _H TensorImpl gt(const TensorImpl& a, const TensorImpl& b) _T;               \
  _H TensorImpl le(const TensorImpl& a, const TensorImpl& b) _T;               \
  _H TensorImpl lt(const TensorImpl& a, const TensorImpl& b) _T;               \
  _H TensorImpl maximum(const TensorImpl& a, const TensorImpl& b) _T;          \
  _H TensorImpl minimum(const TensorImpl& a, const TensorImpl& b) _T;          \
                                                                               \
  _H TensorImpl eq(const TensorImpl& a, const float& b) _T;                    \
  _H TensorImpl ne(const TensorImpl& a, const float& b) _T;                    \
  _H TensorImpl ge(const TensorImpl& a, const float& b) _T;                    \
  _H TensorImpl gt(const TensorImpl& a, const float& b) _T;                    \
  _H TensorImpl le(const TensorImpl& a, const float& b) _T;                    \
  _H TensorImpl lt(const TensorImpl& a, const float& b) _T;                    \
  _H TensorImpl maximum(const TensorImpl& a, const float& b) _T;               \
  _H TensorImpl minimum(const TensorImpl& a, const float& b) _T;               \
                                                                               \
  _H void sin_(TensorImpl& t) _T;                                              \
  _H void cos_(TensorImpl& t) _T;                                              \
  _H void sqrt_(TensorImpl& t) _T;                                             \
  _H void tanh_(TensorImpl& t) _T;                                             \
  _H void exp_(TensorImpl& t) _T;                                              \
  _H void log_(TensorImpl& t) _T;                                              \
                                                                               \
  _H TensorImpl sin(const TensorImpl& t) _T;                                   \
  _H TensorImpl cos(const TensorImpl& t) _T;                                   \
  _H TensorImpl sqrt(const TensorImpl& t) _T;                                  \
  _H TensorImpl tanh(const TensorImpl& t) _T;                                  \
  _H TensorImpl exp(const TensorImpl& t) _T;                                   \
  _H TensorImpl log(const TensorImpl& t) _T;                                   \
                                                                               \
  _H void clampMin_(TensorImpl& t, float min) _T;                              \
  _H void clampMax_(TensorImpl& t, float max) _T;                              \
  _H void clamp_(TensorImpl& t, float min, float max) _T;                      \
                                                                               \
  _H TensorImpl clampMin(const TensorImpl& t, float min) _T;                   \
  _H TensorImpl clampMax(const TensorImpl& t, float max) _T;                   \
  _H TensorImpl clamp(const TensorImpl& t, float min, float max) _T;           \
                                                                               \
  /* aggregation */                                                            \
  _H TensorImpl min(const TensorImpl& t) _T;                                   \
  _H TensorImpl max(const TensorImpl& t) _T;                                   \
  _H TensorImpl sum(const TensorImpl& t) _T;                                   \
  _H TensorImpl mean(const TensorImpl& t) _T;                                  \
  _H TensorImpl var(const TensorImpl& t, bool unbiased) _T;                    \
  _H TensorImpl argmin(const TensorImpl& t) _T;                                \
  _H TensorImpl argmax(const TensorImpl& t) _T;                                \
                                                                               \
  _H std::pair<TensorImpl, TensorImpl> min(const TensorImpl& t, int32_t dim,   \
                                           bool keepDims) _T;                  \
  _H std::pair<TensorImpl, TensorImpl> max(const TensorImpl& t, int32_t dim,   \
                                           bool keepDims) _T;                  \
                                                                               \
  _H TensorImpl sum(const TensorImpl& t, const std::vector<int32_t>& dims,     \
                    bool keepDims) _T;                                         \
  _H TensorImpl mean(const TensorImpl& t, const std::vector<int32_t>& dims,    \
                     bool keepDims) _T;                                        \
  _H TensorImpl var(const TensorImpl& t, const std::vector<int32_t>& dims,     \
                    bool unbiased, bool keepDims) _T;                          \
                                                                               \
  /* permute */                                                                \
  _H TensorImpl permute(const TensorImpl& t, const std::vector<int32_t>& dims) \
      _T;                                                                      \
  _H TensorImpl transpose2D(const TensorImpl& t) _T;                           \
                                                                               \
  /* index */                                                                  \
  _H TensorImpl index(                                                         \
      const TensorImpl& t,                                                     \
      const std::vector<std::reference_wrapper<TensorImpl>>& indices) _T;      \
  _H void indexPut_(                                                           \
      TensorImpl& t,                                                           \
      const std::vector<std::reference_wrapper<TensorImpl>>& indices,          \
      float val) _T;                                                           \
  _H void indexPut_(                                                           \
      TensorImpl& t,                                                           \
      const std::vector<std::reference_wrapper<TensorImpl>>& indices,          \
      const TensorImpl& val) _T;                                               \
                                                                               \
  _H TensorImpl triangle(const TensorImpl& t, int32_t diagonal, bool lower) _T;\
  /* im2col */                                                                 \
  _H TensorImpl im2col(const TensorImpl& t, Size2D kernel, Size2D stride,      \
                       Size2D padding) _T;                                     \
  /* col2im */                                                                 \
  _H TensorImpl col2im(const TensorImpl& t, const Shape& shape, Size2D kernel, \
                       Size2D stride, Size2D padding) _T;                      \
                                                                               \
  /* dot */                                                                    \
  _H TensorImpl dot(const TensorImpl& a, const TensorImpl& b) _T;              \
                                                                               \
  /* matmul */                                                                 \
  _H TensorImpl flash_attention_(const TensorImpl& Q, const TensorImpl& K,     \
               const TensorImpl& V , int32_t head) _T;                         \
  _H TensorImpl upsample_forward(const TensorImpl& Q, int32_t scale_factor) _T;\
  _H TensorImpl upsample_backward(const TensorImpl& Q, int32_t scale_factor) _T;\
  _H void gemm(float* c, const float* a, const float* b, int32_t m,            \
                         int32_t k, int32_t n, bool transA, bool transBk,      \
                         Dtype Ta = Dtype::float32, Dtype Tc = Dtype::float32) _T;\


class TensorImpl;


class TensorOperations {
 public:
  virtual ~TensorOperations() = default;

  static int32_t indicesToOffset(const Shape& strides, const int32_t* indices);
  static void offsetToIndices(int32_t* indices, const Shape& shape,
                              int32_t offset);
  static void reorderIndices(int32_t* indices,
                             const std::vector<int32_t>& order);
  static bool isLeadingOnes(const Shape& shape);
  static bool isTrailingOnes(const Shape& shape);

  static Shape getReduceShape(const TensorImpl& t, int32_t dim, bool keepDims);
  static Shape getReduceShape(const TensorImpl& t,
                              const FixedVector<uint8_t>& inAxis,
                              bool keepDims);

  static ShapeCompatible checkShapeCompatible(const Shape& t0, const Shape& t1,
                                              Shape& retShape,
                                              int32_t skipLast = 0);
  static bool checkShapeEqual(
       const std::vector<std::reference_wrapper<TensorImpl>>& tensors,
       int32_t exceptDim);
  static void error(const char* where, TensorError error);

  TENSOR_OPS_DECLARE(virtual, = 0)
 protected:
  CachedAllocator allocator_;
};

}  // namespace TinyTorch