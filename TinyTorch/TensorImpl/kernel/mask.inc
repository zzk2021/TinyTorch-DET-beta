#pragma once

__global__ void computePrefixSumKernel(const float* mask, int* prefixSum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        prefixSum[idx] = mask[idx] ? 1.0f : 0.0f;
    }
}

__global__ void gatherElementsKernel(const float* input, const int* prefixSum, float* out_indices, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int prev_prefix = (idx > 0) ? prefixSum[idx-1] : 0;
    if (prefixSum[idx] != prev_prefix) {
        int pos = prefixSum[idx] - 1;
        if (pos >= 0) {
            output[pos] = input[idx];
            out_indices[pos] = static_cast<float>(idx);
        }
    }
}

__global__ void scatterElementsKernel(float* original, const int* prefixSum, const float* processed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && prefixSum[idx] != prefixSum[idx - (idx>0)]) {
        int pos = prefixSum[idx] - 1;
        original[idx] = processed[pos];
    }
}

__global__ void scatterGradKernel(
        const float* grad_output,
        const float* indices,
        float* grad_input,
        int num_valid
    ) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < num_valid) {
            int src_pos = tid;          // grad_output中的位置
            int dst_pos = static_cast<int>(indices[tid]); // grad_input中的目标位置
            grad_input[dst_pos] = grad_output[src_pos];
        }
    };