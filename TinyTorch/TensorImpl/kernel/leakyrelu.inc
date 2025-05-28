#pragma once

template <int VEC_SIZE = 4>
__global__ void leaky_relu_opt_kernel(const float* input, float* output, float ALPHA, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * VEC_SIZE;
    if (idx + VEC_SIZE > n) return;

    float4 in = *reinterpret_cast<const float4*>(input + idx);
    float4 out;

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        float val = (&in.x)[i];
        (&out.x)[i] = val > 0 ? val : ALPHA * val;
    }

    *reinterpret_cast<float4*>(output + idx) = out;
}

template <typename T>
__global__ void leaky_relu_kernel(const T* input, T* output, bool* mask, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = input[idx];
        bool condition = (x > static_cast<T>(0));
        output[idx] = condition ? x : static_cast<T>(alpha) * x;
        mask[idx] = condition;
    }
}

template <>
__global__ void leaky_relu_kernel<half>(const half* input, half* output, bool* mask, float alpha, int n) {
     const half alpha_half = __float2half(alpha);
     const half zero = __float2half(0.0f);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half x = input[idx];
         bool condition = x > zero;
        output[idx] = condition ? x : __hmul(alpha_half, x);
         mask[idx] = condition;
    }
}

template <>
__global__ void leaky_relu_kernel<float>(const float* input, float* output, bool* mask, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        bool condition = x > 0;
        output[idx] = condition ? x : alpha * x;
         mask[idx] = condition;
    }
}

template <>
__global__ void leaky_relu_kernel<__nv_bfloat16>(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    bool* mask,
    float alpha,
    int n
) {

     const __nv_bfloat16 alpha_bf16 = __float2bfloat16(alpha);
     const __nv_bfloat16 zero_bf16 = __float2bfloat16(0.0f);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __nv_bfloat16 x = input[idx];
        bool condition = x > zero_bf16;
        output[idx] = condition ? x : __hmul(alpha_bf16, x);
        mask[idx] = condition;
    }
}

template <typename T>
__global__ void leaky_relu_backward(const T* input, T* output, const bool* mask, float alpha, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float mask_float = mask[idx] ? 1.0f : alpha;
        output[idx] = input[idx] * static_cast<T>(mask_float);
    }
}

template <>
__global__ void leaky_relu_backward<half>(const half* input, half* output, const bool* mask, float alpha, int n) {
     const half alpha_half = __float2half(alpha);
     const half ones = __float2half(1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half mask_float = mask[idx] ? ones : alpha_half;
        output[idx] = input[idx] * mask_float;
    }
}

template <>
__global__ void leaky_relu_backward<float>(const float* input, float* output, const bool* mask, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float mask_float = mask[idx] ? 1.0f : alpha;
        output[idx] = input[idx] * mask_float;
    }
}

template <>
__global__ void leaky_relu_backward<__nv_bfloat16>(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    const bool* mask,
    float alpha,
    int n
) {
     const __nv_bfloat16 alpha_half = __float2bfloat16(alpha);
     const __nv_bfloat16 ones = __float2bfloat16(1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __nv_bfloat16 mask_float = mask[idx] ? ones : alpha_half;
        output[idx] = input[idx] * mask_float;
    }
}