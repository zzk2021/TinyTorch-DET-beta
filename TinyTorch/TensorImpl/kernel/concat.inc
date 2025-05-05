#pragma once

template <typename T, int32_t NumDims = 4>
__global__ void ppl_cukernel_concat(
    const T* a_data,
    const T* b_data,
    T* output_data,
    const int32_t* a_strides,   // Precomputed strides for tensor a
    const int32_t* b_strides,   // Precomputed strides for tensor b
    int32_t* output_dims, // Shape of output tensor [dim0_size, dim1_size, ...]
    int32_t concat_dim,         // Dimension to concatenate along
    int a_dim_size,         // Size of tensor a along concat_dim
    int32_t total_elems  // Total elements in output
    )
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < total_elems; idx += blockDim.x * gridDim.x) {
        // Convert linear index to multidimensional coordinates
        int32_t remaining = idx;
        int32_t coord[NumDims];
        #pragma unroll
        for (int i = NumDims - 1; i >= 0; --i) {
            coord[i] = remaining % output_dims[i];
            remaining /= output_dims[i];
        }

        // Determine if the element comes from a or b
        int32_t a_flat_idx = 0;
        int32_t b_flat_idx = 0;
        bool from_a = (coord[concat_dim] < a_dim_size);

        // Calculate the index for the source tensor
        #pragma unroll
        for (int i = 0; i < NumDims; ++i) {
            if (from_a) {
                a_flat_idx += coord[i] * a_strides[i];
            } else {
                int32_t adjusted_coord = (i == concat_dim) ? (coord[i] - a_dim_size) : coord[i];
                b_flat_idx += adjusted_coord * b_strides[i];
            }
        }

        output_data[idx] = from_a ? a_data[a_flat_idx] : b_data[b_flat_idx];
    }
}