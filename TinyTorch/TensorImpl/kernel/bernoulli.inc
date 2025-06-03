#pragma once
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<typename T>
__global__ void kFillRandBernoulli(T* __restrict__ t, const float p,
                                   const unsigned long seed,
                                   const unsigned long seq, const int n){
}


template <>
__global__ void kFillRandBernoulli<float>(float* __restrict__ t, const float p,
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

template <>
__global__ void kFillRandBernoulli<__half>(__half* __restrict__ t, const float p,
                                       const unsigned long seed,
                                       const unsigned long seq, const int n) {
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index >= n) return;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, seq, index, &state);
  const float4 rand = curand_uniform4(&state);
  const __half val0 = __float2half(rand.x < p ? 1.0f : 0.0f);
  const __half val1 = __float2half(rand.y < p ? 1.0f : 0.0f);
  const __half val2 = __float2half(rand.z < p ? 1.0f : 0.0f);
  const __half val3 = __float2half(rand.w < p ? 1.0f : 0.0f);

  const int remain = n - index;

  if (remain >= 4) {
    half2* ptr2 = reinterpret_cast<half2*>(t + index);
    ptr2[0] = __halves2half2(val0, val1);
    ptr2[1] = __halves2half2(val2, val3);
  } else {
    if (remain > 0) t[index] = val0;
    if (remain > 1) t[index + 1] = val1;
    if (remain > 2) t[index + 2] = val2;
  }
}


template <>
__global__ void kFillRandBernoulli<__nv_bfloat16>(__nv_bfloat16* __restrict__ t, const float p,
                                       const unsigned long seed,
                                       const unsigned long seq, const int n) {
  const int index = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (index >= n) return;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, seq, index, &state);
  const float4 rand = curand_uniform4(&state);
  const __nv_bfloat16 val0 = rand.x < p ? 1.0f : 0.0f;
  const __nv_bfloat16 val1 = rand.y < p ? 1.0f : 0.0f;
  const __nv_bfloat16 val2 = rand.z < p ? 1.0f : 0.0f;
  const __nv_bfloat16 val3 = rand.w < p ? 1.0f : 0.0f;

  const int remain = n - index;

  if (remain >= 4) {
    __nv_bfloat162* ptr2 = reinterpret_cast<__nv_bfloat162*>(t + index);
    ptr2[0] = __halves2bfloat162(val0, val1);
    ptr2[1] = __halves2bfloat162(val2, val3);
  } else {
    if (remain > 0) t[index] = val0;
    if (remain > 1) t[index + 1] = val1;
    if (remain > 2) t[index + 2] = val2;
  }
}