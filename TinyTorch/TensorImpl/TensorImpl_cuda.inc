/*
 * TinyTorch
 * @author 	: keith@robot9.me
 *
 */

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "kernel/upsample.inc"
#include "kernel/concat.inc"
#include "kernel/split.inc"
#include "kernel/leakyrelu.inc"
#include "kernel/slice.inc"
#include "kernel/mask.inc"

namespace TinyTorch {

#define WARP_SIZE 32
#define TRANSPOSE_TILE_DIM 32

#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define FETCH_FLOAT3(pointer) (reinterpret_cast<float3*>(&(pointer))[0])
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])
__global__ void convertFloatToHalfKernel(float* src, __half* dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void convertFloatToBf16Kernel(const float* src, __nv_bfloat16* dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

__global__ void convertBf16ToFloatKernel(const __nv_bfloat16* src, float* dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __bfloat162float(src[idx]);  // 使用 CUDA 内置函数转换
    }
}

__global__ void convertHalfToFloatKernel(const __half* src, float* dst, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __half2float(src[idx]);  // 使用 CUDA 内置函数转换
    }
}

#if CUDART_VERSION < 12080
__device__ __half htanh(__half a) {
    __half abs_a = __habs(a);
    __half one = __float2half(1.0f); // 1.0
    __half denominator = __hadd(one, abs_a); // 1 + |a|
    return __hdiv(a, denominator);  // a / (1 + |a|)
}

__device__ __nv_bfloat16 htanh(__nv_bfloat16 a) {
    __nv_bfloat16 abs_a = __habs(a);       //
    __nv_bfloat16 one = __float2bfloat16(1.0f); //  1.0
    __nv_bfloat16 denominator = __hadd(one, abs_a); // 1 + |a|
    return __hdiv(a, denominator);  // a / (1 + |a|)
}
#endif

struct OpCudaAssign {
    __device__ float operator()(const float a, const float b) const {
        return b;
    }
     __device__ float operator()(const half a, const half b) const {
        return b;
    }
     __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
        return b;
    }
};

struct OpCudaAdd {
  __device__ float operator()(const float a, const float b) const {
    return a + b;
  }
  __device__ half operator()(const half a, const half b) const {
    return a + b;
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a + b;
  }
};

struct OpCudaSub {
  __device__ float operator()(const float a, const float b) const {
    return a - b;
  }
  __device__ half operator()(const half a, const half b) const {
    return a - b;
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a - b;
  }
};

struct OpCudaMul {
  __device__ float operator()(const float a, const float b) const {
    return a * b;
  }
  __device__ half operator()(const half a, const half b) const {
    return a * b;
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a * b;
  }
};

struct OpCudaDiv {
  __device__ float operator()(const float a, const float b) const {
    return a / b;
  }
  __device__ half operator()(const half a, const half b) const {
    return a / b;
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a / b;
  }
};

struct OpCudaPow {
  __device__ float operator()(const float a, const float b) const {
    return pow(a, b);
  }
    __device__ float operator()(const half a, const half b) const {
    return hexp(b* hlog2(a));
  }
    __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return hexp(b* hlog2(a));
  }
};

struct OpCudaEq {
  __device__ float operator()(const float a, const float b) const {
    return a == b ? 1.f : 0.f;
  }
  __device__ float operator()(const half a, const half b) const {
    return a == b ? 1.f : 0.f;
  }
  __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a == b ? 1.f : 0.f;
  }
};

struct OpCudaNe {
  __device__ float operator()(const float a, const float b) const {
    return a != b ? 1.f : 0.f;
  }
  __device__ float operator()(const half a, const half b) const {
    return a != b ? 1.f : 0.f;
  }
  __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a != b ? 1.f : 0.f;
  }
};

struct OpCudaLt {
  __device__ float operator()(const float a, const float b) const {
    return a < b ? 1.f : 0.f;
  }
    __device__ float operator()(const half a, const half b) const {
    return a < b ? 1.f : 0.f;
  }
    __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a < b ? 1.f : 0.f;
  }
};

struct OpCudaLe {
  __device__ float operator()(const float a, const float b) const {
    return a <= b ? 1.f : 0.f;
  }
  __device__ float operator()(const half a, const half b) const {
    return a <= b ? 1.f : 0.f;
  }
  __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a <= b ? 1.f : 0.f;
  }
};

struct OpCudaGt {
  __device__ float operator()(const float a, const float b) const {
    return a > b ? 1.f : 0.f;
  }
  __device__ float operator()(const half a, const half b) const {
    return a > b ? 1.f : 0.f;
  }
  __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a > b ? 1.f : 0.f;
  }
};

struct OpCudaGe {
  __device__ float operator()(const float a, const float b) const {
    return a >= b ? 1.f : 0.f;
  }
  __device__ float operator()(const half a, const half b) const {
    return a >= b ? 1.f : 0.f;
  }
  __device__ float operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a >= b ? 1.f : 0.f;
  }
};

struct OpCudaMax {
  __device__ float operator()(const float a, const float b) const {
    return max(a, b);
  }
  __device__ __half operator()(const half a, const half b) const {
    return __hmax(a, b);
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return __hmax(a, b);
  }
};

struct OpCudaMin {
  __device__ float operator()(const float a, const float b) const {
    return min(a, b);
  }
   __device__ __half operator()(const __half a, const __half b) const {
    return __hmin(a, b);
  }
   __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return __hmin(a, b);
  }
};

struct OpCudaSin_ {
  __device__ void operator()(float& a) const { a = sin(a); }
  __device__ void operator()(__half& a) const { a = hsin(a); }
  __device__ void operator()(__nv_bfloat16& a) const { a = hsin(a); }
};


struct OpCudaCos_ {
  __device__ void operator()(float& a) const { a = cos(a); }
  __device__ void operator()(__half& a) const { a = hcos(a); }
  __device__ void operator()(__nv_bfloat16& a) const { a = hcos(a); }
};


struct OpCudaSqrt_ {
  __device__ void operator()(float& a) const { a = sqrt(a); }
  __device__ void operator()(__half& a) const { a = hsqrt(a); }
  __device__ void operator()(__nv_bfloat16& a) const { a = hsqrt(a); }
};


struct OpCudaTanh_ {
  __device__ void operator()(float& a) const { a = tanh(a); }
  __device__ void operator()(__half& a) const { a = htanh(a); }
  __device__ void operator()(__nv_bfloat16& a) const { a = htanh(a); }
};


struct OpCudaExp_ {
  __device__ void operator()(float& a) const { a = exp(a); }
  __device__ void operator()(__half& a) const { a = hexp(a); }
  __device__ void operator()(__nv_bfloat16& a) const { a = hexp(a); }
};


struct OpCudaLog_ {
  __device__ void operator()(float& a) const { a = log(a); }
  __device__ void operator()(__half& a) const { a = hlog(a); }
  __device__ void operator()(__nv_bfloat16& a) const { a = hlog(a); }
};


struct OpCudaSin {
  __device__ float operator()(const float a) const { return sin(a); }
  __device__ __half operator()(const __half a) const { return hsin(a); }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a) const { return hsin(a); }
};


struct OpCudaCos {
  __device__ float operator()(const float a) const { return cos(a); }
  __device__ __half operator()(const __half a) const { return hcos(a); }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a) const { return hcos(a); }
};


struct OpCudaSqrt {
  __device__ float operator()(const float a) const { return sqrt(a); }
  __device__ __half operator()(const __half a) const { return hsqrt(a); }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a) const { return hsqrt(a); }
};


struct OpCudaTanh {
  __device__ float operator()(const float a) const { return tanh(a); }
  __device__ __half operator()(const __half a) const { return htanh(a); }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a) const { return htanh(a); }
};


struct OpCudaExp {
  __device__ float operator()(const float a) const { return exp(a); }
  __device__ __half operator()(const __half a) const { return hexp(a); }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a) const { return hexp(a); }
};


struct OpCudaLog {
  __device__ float operator()(const float a) const { return log(a); }
  __device__ __half operator()(const __half a) const { return hlog(a); }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a) const { return hlog(a); }
};

__device__ int32_t cuGetReduceSrcIndex(const int32_t* __restrict__ retShape,
                                       const int32_t* __restrict__ srcStrides,
                                       int32_t idx, int32_t dim,
                                       int32_t retDimCount) {
  int32_t outIndex = idx;
  int32_t inIndex = 0;
#pragma unroll
  for (int32_t d = retDimCount - 1; d >= 0; d--) {
    int32_t coord = outIndex % retShape[d];
    outIndex /= retShape[d];
    inIndex += coord * srcStrides[d < dim ? d : d + 1];
  }
  return inIndex;
}

__device__ int32_t cuGetReduceDstIndex(const int32_t* __restrict__ shape,
                                       const int32_t* __restrict__ strides,
                                       int32_t idx, int32_t dim,
                                       int32_t dimCount) {
  int32_t retIdx = 0;
  int32_t stride = 1;
#pragma unroll
  for (int32_t d = dimCount - 1; d >= 0; d--) {
    if (d != dim) {
      retIdx += (idx / strides[d] % shape[d]) * stride;
      stride *= shape[d];
    }
  }
  return retIdx;
}

__device__ int32_t cuGetReduceDstIndex(const int32_t* __restrict__ shape,
                                       const int32_t* __restrict__ strides,
                                       const uint8_t* __restrict__ inAxis,
                                       int32_t idx, int32_t dimCount) {
  int32_t retIdx = 0;
  int32_t stride = 1;
#pragma unroll
  for (int32_t d = dimCount - 1; d >= 0; d--) {
    if (0 == inAxis[d]) {
      retIdx += (idx / strides[d] % shape[d]) * stride;
      stride *= shape[d];
    }
  }
  return retIdx;
}

__device__ __forceinline__ void cuGetSubIndices(
    int32_t* __restrict__ subIndices, const int32_t* __restrict__ shape,
    const float* const* __restrict__ indices, int32_t idx, int len) {
#pragma unroll
  for (int32_t i = 0; i < len; i++) {
    auto ind = (int32_t)((float*)indices[i])[idx];
    subIndices[i] = ind >= 0 ? ind : ind + shape[i];
  }
}

template<typename T>
__global__ void kFillConstant(T* __restrict__ t, const T val,
                              const size_t n) {
}

template<>
__global__ void kFillConstant(float* __restrict__ t, const float val,
                              const size_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index + 3 < n) {
    FETCH_FLOAT4(t[index]) = make_float4(val, val, val, val);
  } else {
    if (index < n) t[index] = val;
    if (index + 1 < n) t[index + 1] = val;
    if (index + 2 < n) t[index + 2] = val;
  }
}

template<>
__global__ void kFillConstant(__half* __restrict__ t, const __half val, const size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vec_index = index * 2;

    if (vec_index + 1 < n) {
        *((__half2*)(t + vec_index)) = make_half2(val, val);
    } else {
        if (vec_index < n) t[vec_index] = val;
        if (vec_index + 1 < n) t[vec_index + 1] = val;
    }
}

template<>
__global__ void kFillConstant(__nv_bfloat16* __restrict__ t,
                               const __nv_bfloat16 val,
                               const size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t vec_index = index * 2;

    if (vec_index + 1 < n) {
        *((__nv_bfloat162*)(t + vec_index)) = make_bfloat162(val, val);
    } else {
        if (vec_index < n) t[vec_index] = val;
        if (vec_index + 1 < n) t[vec_index + 1] = val;
    }
}

__global__ void kFillLinSpace(float* __restrict__ dst, const float start,
                              const float step, const size_t n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const auto base = start + static_cast<float>(index) * step;
  if (index + 3 < n) {
    FETCH_FLOAT4(dst[index]) =
        make_float4(base, base + step, base + 2 * step, base + 3 * step);
  } else {
    if (index < n) dst[index] = base;
    if (index + 1 < n) dst[index + 1] = base + step;
    if (index + 2 < n) dst[index + 2] = base + 2 * step;
  }
}

__global__ void kFillRandUniform(float* __restrict__ t, const float minVal,
                                 const float maxVal, const unsigned long seed,
                                 const unsigned long seq, const int n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);
    const auto range = maxVal - minVal;

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x * range + minVal, rand.y * range + minVal,
                      rand.z * range + minVal, rand.w * range + minVal);
    } else {
      if (index < n) t[index] = rand.x * range + minVal;
      if (index + 1 < n) t[index + 1] = rand.y * range + minVal;
      if (index + 2 < n) t[index + 2] = rand.z * range + minVal;
    }
  }
}

__global__ void kFillRandNormal(float* __restrict__ t, const float mean,
                                const float stddev, const unsigned long seed,
                                const unsigned long seq, const int n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_normal4(&state);

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x * stddev + mean, rand.y * stddev + mean,
                      rand.z * stddev + mean, rand.w * stddev + mean);
    } else {
      if (index < n) t[index] = rand.x * stddev + mean;
      if (index + 1 < n) t[index + 1] = rand.y * stddev + mean;
      if (index + 2 < n) t[index + 2] = rand.z * stddev + mean;
    }
  }
}


__global__ void kFillRandBernoulli(float* __restrict__ t, const float p,
                                   const unsigned long seed,
                                   const unsigned long seq, const int n) {
  const auto index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index < n) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, seq, index, &state);
    const auto rand = curand_uniform4(&state);

    if (index + 3 < n) {
      FETCH_FLOAT4(t[index]) =
          make_float4(rand.x < p ? 1.f : 0.f, rand.y < p ? 1.f : 0.f,
                      rand.z < p ? 1.f : 0.f, rand.w < p ? 1.f : 0.f);
    } else {
      if (index < n) t[index] = rand.x < p ? 1.f : 0.f;
      if (index + 1 < n) t[index + 1] = rand.y < p ? 1.f : 0.f;
      if (index + 2 < n) t[index + 2] = rand.z < p ? 1.f : 0.f;
    }
  }
}

template <typename OP, typename T = float>
__global__ void kSingleOp_(T* __restrict__ t, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    opFunc(t[index]);
  }
}

template <typename OP, typename T = float>
__global__ void kSingleOp(T* __restrict__ ret, const T* __restrict__ t,
                          const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    ret[index] = opFunc(t[index]);
  }
}

template <typename OP, typename T = float>
__global__ void kPairOp(T* __restrict__ c, const T* __restrict__ a,
                        const T* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a[index], b[index]);
  }
}



template <typename OP, typename T = float>
__global__ void kPairScalarFirstOp(T* __restrict__ c, const T a,
                                   const T* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a, b[index]);
  }
}

template <typename OP, typename T = float>
__global__ void kPairScalarFirstOp(T* __restrict__ c,
                                   const T* __restrict__ a,
                                   const T* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    c[index] = opFunc(a[0], b[index]);
  }
}

template <typename OP, typename T = float>
__global__ void kPairScalarSecondOp(T* __restrict__ c,
                                    const T* __restrict__ a, const T b,
                                    const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    c[index] = opFunc(a[index], b);
  }
}

template <typename OP, typename T = float>
__global__ void kPairScalarSecondOp(T* __restrict__ c,
                                    const T* __restrict__ a,
                                    const T* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    c[index] = opFunc(a[index], b[0]);
  }
}

template <typename OP , typename T = float>
__global__ void kPairOp_(T* __restrict__ a, const T* __restrict__ b,
                         const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    a[index] = opFunc(a[index], b[index]);
  }
}

template <typename OP, typename T = float>
__global__ void kPairScalarSecondOp_(T* __restrict__ a, const T b,
                                     const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    a[index] = opFunc(a[index], b);
  }
}

template <typename OP, typename T = float>
__global__ void kPairScalarSecondOp_(T* __restrict__ a,
                                     const T* __restrict__ b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    a[index] = opFunc(a[index], b[0]);
  }
}

__global__ void kClamp(float* __restrict__ ret, const float* __restrict__ t,
                       const float minVal, const float maxVal, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    ret[index] = max(minVal, min(t[index], maxVal));
  }
}

__global__ void kClamp_(float* __restrict__ t, const float minVal,
                        const float maxVal, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    t[index] = max(minVal, min(t[index], maxVal));
  }
}

template <typename OP, bool Leading, bool First, typename T = float>
__global__ void kBroadcastOpFast(T* __restrict__ ret,
                                 const T* __restrict__ a,
                                 const T* __restrict__ b,
                                 const int32_t stride, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;
  if (index < n) {
    if (First) {
      ret[index] =
          opFunc(a[Leading ? index % stride : index / stride], b[index]);
    } else {
      ret[index] =
          opFunc(a[index], b[Leading ? index % stride : index / stride]);
    }
  }
}

__device__ int32_t cuIndicesToOffset(const int32_t* __restrict__ strides,
                                     const int32_t* __restrict__ indices,
                                     const int32_t dimCount) {
  int32_t offset = 0;
#pragma unroll
  for (int32_t i = 0; i < dimCount; i++) {
    offset += indices[i] * strides[i];
  }
  return offset;
}

__device__ void cuOffsetToIndices(int32_t* __restrict__ indices,
                                  const int32_t* __restrict__ shape,
                                  const int32_t index, const int32_t dimCount) {
  int32_t offset = index;
#pragma unroll
  for (int32_t i = dimCount - 1; i >= 0; i--) {
    indices[i] = offset % shape[i];
    offset /= shape[i];
  }
}

template <typename OP, typename T = float >
__global__ void kBroadcastOpCommon(TensorCudaCtx<T> c, const TensorCudaCtx<T> a,
                                   const TensorCudaCtx<T> b, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP opFunc;

  if (index < n) {
    int32_t cIndices[TENSOR_MAX_DIMS];
    int32_t aIndices[TENSOR_MAX_DIMS] = {};
    int32_t bIndices[TENSOR_MAX_DIMS] = {};

    cuOffsetToIndices(cIndices, c.shape_, static_cast<int32_t>(index),
                      c.dimCount_);

#pragma unroll
    for (auto j = 0; j < c.dimCount_; j++) {
      if (j >= c.dimCount_ - a.dimCount_) {
        const int32_t aIndex = j - (c.dimCount_ - a.dimCount_);
        aIndices[aIndex] = (a.shape_[aIndex] != 1) ? cIndices[j] : 0;
      }
      if (j >= c.dimCount_ - b.dimCount_) {
        const int32_t bIndex = j - (c.dimCount_ - b.dimCount_);
        bIndices[bIndex] = (b.shape_[bIndex] != 1) ? cIndices[j] : 0;
      }
    }
    const auto aIdx = cuIndicesToOffset(a.strides_, aIndices, a.dimCount_);
    const auto bIdx = cuIndicesToOffset(b.strides_, bIndices, b.dimCount_);
    c.data_[index] = opFunc(a.data_[aIdx], b.data_[bIdx]);
  }
}


struct OpCudaReduceMax {
  __device__ float operator()(const float a, const float b) const {
    return max(a, b);
  }
  __device__ half operator()(const half a, const half b) const {
    return __hmax(a, b);
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return __hmax(a, b);
  }

  template <typename T = float>
  static __device__ T defaultVal() {
        if constexpr (std::is_same_v<T, float>) {
            return -FLT_MAX;
        } else if constexpr (std::is_same_v<T, half>) {
            return -CUDART_MAX_NORMAL_FP16;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return -CUDART_MAX_NORMAL_BF16;
        }
    }
};

struct OpCudaReduceMin {
  __device__ float operator()(const float a, const float b) const {
    return min(a, b);
  }
  __device__ half operator()(const half a, const half b) const {
    return __hmin(a, b);
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return __hmin(a, b);
  }

  template <typename T = float>
  static __device__ T defaultVal() {
        if constexpr (std::is_same_v<T, float>) {
            return FLT_MAX;
        } else if constexpr (std::is_same_v<T, half>) {
            return CUDART_MAX_NORMAL_FP16;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return CUDART_MAX_NORMAL_BF16;
        }
    }
};

struct OpCudaReduceSum {

  __device__ float operator()(const float a, const float b) const {
    return a + b;
  }
  __device__ half operator()(const half a, const half b) const {
    return a + b;
  }
  __device__ __nv_bfloat16 operator()(const __nv_bfloat16 a, const __nv_bfloat16 b) const {
    return a + b;
  }

    template <typename T = float>
    static __device__ T defaultVal() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.f;
        } else if constexpr (std::is_same_v<T, half>) {
            return CUDART_NEG_ZERO_FP16;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            return CUDART_NEG_ZERO_BF16;
        }
    }

};

template <typename OP, typename T = float>
__device__ __forceinline__ float cuWarpReduce(T val) {
  const OP op;
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
  val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
  return val;
}

template <typename OP, typename T = float>
__device__ __forceinline__ int cuWarpReduceIdx(T val, int idx) {
  const OP op;

  T otherVal = __shfl_down_sync(0xFFFFFFFF, val, 16);
  int32_t otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 16);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 8);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 8);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 4);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 4);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 2);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 2);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  otherVal = __shfl_down_sync(0xFFFFFFFF, val, 1);
  otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, 1);
  if (op(otherVal, val) == otherVal) {
    val = otherVal;
    idx = otherIdx;
  }

  return idx;
}

template <typename OP, typename T = float>
__global__ void kReduceAll(T* __restrict__ output,
                           const T* __restrict__ input, const int n,
                           const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[WARP_SIZE];
  T val = OP::template defaultVal<T>();

  if (index < n) {
    val = input[index];
  }
  val = cuWarpReduce<OP, T>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x]
                                                : OP::defaultVal();
    val = cuWarpReduce<OP, T>(val);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = val;
  }
}

template <typename OP, typename T  = float>
__global__ void kReduceAllFirstDim(T* __restrict__ output,
                                   const T* __restrict__ input, const int n,
                                   const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[WARP_SIZE];
  float val = OP::template defaultVal<T>();

  const auto segDim = gridDim.x * blockDim.x / m;
  const auto segIdx = index / segDim;
  const auto segTid = index % segDim;
  if (segTid < n) {
    val = input[segIdx + segTid * m];
  }
  val = cuWarpReduce<OP, T>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x]
                                                : OP::template  defaultVal<T>();
    val = cuWarpReduce<OP, T>(val);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = val;
  }
}

template <typename OP, typename T>
__global__ void kReduceAllLastDim(T* __restrict__ output,
                                  const T* __restrict__ input, const int n,
                                  const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared[WARP_SIZE];
  float val = OP::template defaultVal<T>();

  const auto segDim = gridDim.x * blockDim.x / m;
  const auto segIdx = index / segDim;
  const auto segTid = index % segDim;
  if (segTid < n) {
    val = input[segTid + segIdx * n];
  }
  val = cuWarpReduce<OP, T>(val);

  if (threadIdx.x % warpSize == 0) {
    shared[threadIdx.x / warpSize] = val;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[threadIdx.x]
                                                : OP::template defaultVal<T>();
    val = cuWarpReduce<OP, T>(val);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = val;
  }
}

template <typename OP, typename T =float>
__global__ void kReduceAllIdx(T* output, const T* __restrict__ input,
                              const int n, const int m) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T sharedVal[WARP_SIZE];
  __shared__ int sharedIdx[WARP_SIZE];

  float val = OP::template defaultVal<T>();
  int idx = -1;

  if (index < n) {
    val = input[index];
    idx = static_cast<int>(index);
  }
  idx = cuWarpReduceIdx<OP, T>(val, idx);

  if (threadIdx.x % warpSize == 0) {
    sharedVal[threadIdx.x / warpSize] = val;
    sharedIdx[threadIdx.x / warpSize] = idx;
  }
  __syncthreads();

  if (threadIdx.x < warpSize) {
    val = (threadIdx.x < blockDim.x / warpSize) ? sharedVal[threadIdx.x]
                                                : OP::template defaultVal<T>();
    idx = (threadIdx.x < blockDim.x / warpSize) ? sharedIdx[threadIdx.x] : -1;

    idx = cuWarpReduceIdx<OP, T>(val, idx);
  }

  if (threadIdx.x == 0) {
    output[blockIdx.x] = static_cast<float>(idx);
  }
}
template <typename T =float>
__global__ void kSquaredDiff(T* __restrict__ output,
                             const T* __restrict__ input,
                             const T* __restrict__ mean, const int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    const T diff = input[index] - *mean;
    output[index] = diff * diff;
  }
}

template <typename OP, typename T = float>
__global__ void kReduceLastDim(T* values, float * indices, const T* t,
                               int32_t dimSize, int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP op;
  if (index < n) {
    auto targetVal = OP::template defaultVal<T>();
    int32_t targetIdx = 0;
    int32_t srcIdx = index * dimSize;
#pragma unroll
    for (int32_t j = 0; j < dimSize; j++) {
      auto val = t[srcIdx++];
      if (op(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    values[index] = targetVal;
    indices[index] = targetIdx;
  }
}

template <typename OP, typename T =float>
__global__ void kReduceDim(TensorCudaCtx<T> values, float* indices,
                           const TensorCudaCtx<T> t, int32_t dim, int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  const OP op;

  const auto dimSize = t.shape_[dim];
  const auto stride = t.strides_[dim];

  if (index < n) {
    auto targetVal = OP::template defaultVal<T>();
    int32_t targetIdx = 0;
    int32_t srcIdx = cuGetReduceSrcIndex(values.shape_, t.strides_, index, dim,
                                         values.dimCount_);
#pragma unroll
    for (int32_t j = 0; j < dimSize; j++) {
      auto val = t.data_[srcIdx];
      srcIdx += stride;
      if (op(val, targetVal) == val) {
        targetVal = val;
        targetIdx = j;
      }
    }
    values.data_[index] = targetVal;
    indices[index] = targetIdx;
  }
}

template <typename T = float>
__global__ void kReduceSum(T* const retPtr, const TensorCudaCtx<T> t,
                           const FixedVector<uint8_t> inAxis, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t retIdx = cuGetReduceDstIndex(t.shape_, t.strides_, inAxis.data,
                                         index, t.dimCount_);
    atomicAdd(&retPtr[retIdx], t.data_[index]);
  }
}
template <typename T = float>
__global__ void kReduceVar(float* retPtr, const TensorCudaCtx<T> t,
                           const float* meanValues,
                           const FixedVector<uint8_t> inAxis, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t retIdx = cuGetReduceDstIndex(t.shape_, t.strides_, inAxis.data,
                                         index, t.dimCount_);
    float diff = t.data_[index] - meanValues[retIdx];
    atomicAdd(&retPtr[retIdx], diff * diff);
  }
}
template <typename T = float>
__global__ void kPermute(const TensorCudaCtx<T> ret, const TensorCudaCtx<T> t,
                         const FixedVector<int32_t> dims, int n) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    unsigned int srcIndex = 0;
    auto offset = index;
#pragma unroll
    for (int32_t d = 0; d < t.dimCount_; d++) {
      srcIndex += (offset / ret.strides_[d]) * t.strides_[dims.data[d]];
      offset %= ret.strides_[d];
    }
    ret.data_[index] = t.data_[srcIndex];
  }
}

template <typename T = float>
__global__ void kTranspose(T* __restrict__ out,
                           const T* __restrict__ in, const int width,
                           const int height) {
  __shared__ float tile[TRANSPOSE_TILE_DIM]
                       [TRANSPOSE_TILE_DIM + 1];  // +1 to avoid bank conflicts

  auto x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
  auto y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

  if (x < width && y < height) {
    tile[threadIdx.y][threadIdx.x] = in[y * width + x];
  }
  __syncthreads();

  x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
  y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

  if (x < height && y < width) {
    out[y * height + x] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename T>
__global__ void kIndex(float* retData, const TensorCudaCtx<T> t,
                       const FixedVector<float*> indices, int dimStride,
                       int len, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t.shape_, indices.data, index, len);
    int32_t dataIdx = cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
    memcpy(&retData[dimStride * index], &t.data_[dataIdx],
           dimStride * sizeof(float));
  }
}
template <typename T>
__global__ void kIndexPut(TensorCudaCtx<T> t, FixedVector<float*> indices,
                          int dimStride, int len, float val, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t.shape_, indices.data, index, len);
    int32_t dataIdx = cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
#pragma unroll
    for (int32_t i = 0; i < dimStride; i++) {
      t.data_[dataIdx + i] = val;
    }
  }
}

template<typename T >
__global__ void kIndexPut(TensorCudaCtx<T> t, FixedVector<float*> indices,
                          int dimStride, int len, float* val, int n) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    int32_t subIndices[TENSOR_MAX_DIMS];
    cuGetSubIndices(subIndices, t.shape_, indices.data, index, len);
    int32_t dataIdx = cuIndicesToOffset(t.strides_, subIndices, t.dimCount_);
    memcpy(&t.data_[dataIdx], &val[dimStride * index],
           dimStride * sizeof(float));
  }
}

template<typename T = float>
__global__ void kIm2Col(T * ret, const T * t, int32_t batch,
                        int32_t channels, int32_t height, int32_t width,
                        int32_t outH, int32_t outW, int32_t kernelH,
                        int32_t kernelW, int32_t strideH, int32_t strideW,
                        int32_t paddingH, int32_t paddingW, int32_t imStride,
                        int32_t colH, int32_t colW) {
  auto index = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t totalElements = batch * outH * outW * channels * kernelH * kernelW;

  if (index < totalElements) {
    int32_t kw = index % kernelW;
    int32_t kh = (index / kernelW) % kernelH;
    int32_t c = (index / (kernelW * kernelH)) % channels;
    int32_t w = (index / (kernelW * kernelH * channels)) % outW;
    int32_t h = (index / (kernelW * kernelH * channels * outW)) % outH;
    int32_t n = (index / (kernelW * kernelH * channels * outW * outH)) % batch;

    int32_t colIdx = (n * outH + h) * outW + w;
    int32_t imRow = h * strideH + kh - paddingH;
    int32_t imCol = w * strideW + kw - paddingW;
    int32_t colWIdx = c * kernelH * kernelW + kh * kernelW + kw;

    T value = static_cast<T>(0.0f);
    if (imRow >= 0 && imRow < height && imCol >= 0 && imCol < width) {
      int32_t imgIdx = imCol + width * (imRow + height * c);
      value = t[n * imStride + imgIdx];
    }
    ret[colIdx * colW + colWIdx] = value;
  }
}

template<typename T>
__global__ void kIm2Col1D(T* col_data,
                         const T* im_data,
                         int32_t batch,
                         int32_t channels,
                         int32_t length,
                         int32_t outLength,
                         int32_t kernel_size,
                         int32_t stride,
                         int32_t padding,
                         int32_t im_stride,
                         int32_t colH,
                         int32_t colW) {
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t totalElements = batch * outLength * channels * kernel_size;

    if (index < totalElements) {
        const int32_t kl = index % kernel_size;
        const int32_t c = (index / kernel_size) % channels;
        const int32_t l = (index / (kernel_size * channels)) % outLength;
        const int32_t n = index / (kernel_size * channels * outLength);

        const int32_t imPos = l * stride + kl - padding;

        const int32_t colIdx = n * outLength + l;
        const int32_t colWIdx = c * kernel_size + kl;

        T value = static_cast<T>(0);
        if (imPos >= 0 && imPos < length) {

            const int32_t imOffset = n * (channels * im_stride)
                                   + c * im_stride
                                   + imPos;
            value = im_data[imOffset];
        }

        col_data[colIdx * colW + colWIdx] = value;
    }
}


template<typename T = float>
__global__ void kCol2Im(T* ret, const T* t, int32_t batch,
                        int32_t channels, int32_t height, int32_t width,
                        int32_t outH, int32_t outW, int32_t kernelH,
                        int32_t kernelW, int32_t strideH, int32_t strideW,
                        int32_t paddingH, int32_t paddingW, int32_t imStride,
                        int32_t colW) {
  auto index = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t totalElements = batch * channels * height * width;

  if (index < totalElements) {
    int32_t w = index % width;
    int32_t h = (index / width) % height;
    int32_t c = (index / (width * height)) % channels;
    int32_t n = index / (width * height * channels);

    T value = 0.0f;
    for (int32_t kh = 0; kh < kernelH; ++kh) {
      for (int32_t kw = 0; kw < kernelW; ++kw) {
        int32_t imRow = h + paddingH - kh;
        int32_t imCol = w + paddingW - kw;

        if (imRow % strideH == 0 && imCol % strideW == 0) {
          int32_t outRow = imRow / strideH;
          int32_t outCol = imCol / strideW;

          if (outRow >= 0 && outRow < outH && outCol >= 0 && outCol < outW) {
            int32_t colIdx = (n * outH + outRow) * outW + outCol;
            int32_t colWIdx = c * kernelH * kernelW + kh * kernelW + kw;
            value += t[colIdx * colW + colWIdx];
          }
        }
      }
    }

    ret[index] = value;
  }
}
template<typename T>
__global__ void kCol2Im1D(T* grad_input,
                         const T* grad_col,
                         int32_t batch,
                         int32_t channels,
                         int32_t length,
                         int32_t outLength,
                         int32_t kernel_size,
                         int32_t stride,
                         int32_t padding,
                         int32_t colW) {
  const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t totalElements = batch * channels * length;

  if (index < totalElements) {
    const int32_t l = index % length;         // 输入长度位置
    const int32_t c = (index / length) % channels; // 通道索引
    const int32_t n = index / (length * channels); // 批次索引

    T value = 0.0f;

    for (int32_t kl = 0; kl < kernel_size; ++kl) {

      const int32_t imPos = l + padding - kl;

      if (imPos % stride == 0) {
        const int32_t outPos = imPos / stride;

        if (outPos >= 0 && outPos < outLength) {

          const int32_t colIdx = n * outLength + outPos;
          const int32_t colWIdx = c * kernel_size + kl;

          value += grad_col[colIdx * colW + colWIdx];
        }
      }
    }

    grad_input[index] = value;
  }
}
__global__ void kDot(float* ret, const float* a, const float* b, int n) {
  extern __shared__ float sharedData[];

  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto threadId = threadIdx.x;

  float temp = 0.f;
  while (tid < n) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  sharedData[threadId] = temp;
  __syncthreads();

  for (int32_t stride = (int32_t)blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadId < stride) {
      sharedData[threadId] += sharedData[threadId + stride];
    }
    __syncthreads();
  }

  if (threadId == 0) {
    atomicAdd(ret, sharedData[0]);
  }
}

}  // namespace TinyTorch

template <bool LOWER>
 __global__ void kTriangle(float* ret, const float* t, const int32_t rows,
                           const int32_t cols, const int32_t diagonal) {
  auto i = static_cast<int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  auto j = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);

  if (i < rows && j < cols) {
    const auto index = i * cols + j;
    if ((LOWER && j <= i + diagonal) || (!LOWER && j >= i + diagonal)) {
      ret[index] = t[index];
    } else {
      ret[index] = 0.0f;
    }
  }
}

