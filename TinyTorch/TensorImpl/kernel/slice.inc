#pragma once

template <int NDIMS>
__global__ void from_slice_kernel(
    const float* __restrict__ a_data,
    const int32_t* __restrict__ a_strides,
    const int* __restrict__ starts,
    const int32_t* __restrict__ new_strides, // [N1*N2*N3, N2*N3, N3, 1]
    const int32_t* __restrict__ new_dims,     // [N0, N1, N2, N3]
    float* __restrict__ result_data,
    int total_elements
){
}

template <>
__global__ void from_slice_kernel<1>(
    const float* __restrict__ a_data,
    const int32_t* __restrict__ a_strides,
    const int* __restrict__ starts,
    const int32_t* __restrict__ new_strides,  // [1]
    const int32_t* __restrict__ new_dims,      // [N0]
    float* __restrict__ result_data,
    int total_elements
) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= total_elements) return;
  int dim0_idx = (dst_idx / new_strides[0]) % new_dims[0];
  int src_offset = (starts[0] + dim0_idx) * a_strides[0];
  result_data[dst_idx] = a_data[src_offset];
}

template <>
__global__ void from_slice_kernel<2>(
    const float* __restrict__ a_data,
    const int32_t* __restrict__ a_strides,
    const int* __restrict__ starts,
    const int32_t* __restrict__ new_strides, //
    const int32_t* __restrict__ new_dims,     // [N0, N1]
    float* __restrict__ result_data,
    int total_elements
) {
  int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= total_elements) return;
  int dim0_idx = (dst_idx / new_strides[0]) % new_dims[0];
  int dim1_idx = dst_idx % new_dims[1];
  int src_offset = (starts[0] + dim0_idx) * a_strides[0] + (starts[1] + dim1_idx) * a_strides[1];
  result_data[dst_idx] = a_data[src_offset];
}

template <>
__global__ void from_slice_kernel<4>(
    const float* __restrict__ a_data,
    const int32_t* __restrict__ a_strides,
    const int* __restrict__ starts,
    const int32_t* __restrict__ new_strides, // [N1*N2*N3, N2*N3, N3, 1]
    const int32_t* __restrict__ new_dims,     // [N0, N1, N2, N3]
    float* __restrict__ result_data,
    int total_elements
) {
    int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_idx >= total_elements) return;

    int dim0_idx = (dst_idx / new_strides[0]) % new_dims[0];
    int dim1_idx = (dst_idx / new_strides[1]) % new_dims[1];
    int dim2_idx = (dst_idx / new_strides[2]) % new_dims[2];
    int dim3_idx = dst_idx % new_dims[3];

    int src_offset =
        (starts[0] + dim0_idx) * a_strides[0] +
        (starts[1] + dim1_idx) * a_strides[1] +
        (starts[2] + dim2_idx) * a_strides[2] +
        (starts[3] + dim3_idx) * a_strides[3];

    result_data[dst_idx] = a_data[src_offset];
}

template <>
__global__ void from_slice_kernel<5>(
    const float* __restrict__ a_data,
    const int32_t* __restrict__ a_strides,     // [5]
    const int* __restrict__ starts,            // [5]
    const int32_t* __restrict__ new_strides,   // [5], e.g. [D*H*W*C, H*W*C, W*C, C, 1]
    const int32_t* __restrict__ new_dims,      // [5], e.g. [B, C, D, H, W]
    float* __restrict__ result_data,
    int total_elements
) {
    int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_idx >= total_elements) return;

    int idx = dst_idx;
    int dim4 = idx % new_dims[4]; idx /= new_dims[4]; // W
    int dim3 = idx % new_dims[3]; idx /= new_dims[3]; // H
    int dim2 = idx % new_dims[2]; idx /= new_dims[2]; // D
    int dim1 = idx % new_dims[1]; idx /= new_dims[1]; // C
    int dim0 = idx;                                   // B

    int src_offset =
        (starts[0] + dim0) * a_strides[0] +
        (starts[1] + dim1) * a_strides[1] +
        (starts[2] + dim2) * a_strides[2] +
        (starts[3] + dim3) * a_strides[3] +
        (starts[4] + dim4) * a_strides[4];

    result_data[dst_idx] = a_data[src_offset];
}

template <int NDIMS>
__global__ void from_slice_kernel_backward(
    float* __restrict__ a_data,                 //
    const int32_t* __restrict__ a_strides,      //
    const int* __restrict__ starts,             //
    const int32_t* __restrict__ new_strides,    //
    const int32_t* __restrict__ new_dims,       //
    const float* __restrict__ source_data,      //
    int total_elements                          //
) {
}

template <>
__global__ void from_slice_kernel_backward<1>(
    float* __restrict__ a_data,                 //
    const int32_t* __restrict__ a_strides,      //
    const int* __restrict__ starts,             //
    const int32_t* __restrict__ new_strides,    //
    const int32_t* __restrict__ new_dims,       //
    const float* __restrict__ source_data,      //
    int total_elements                          //
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;
  int dim0_idx = (idx / new_strides[0]) % new_dims[0];
  int offset = (starts[0] + dim0_idx) * a_strides[0] ;
  a_data[offset] += source_data[idx];
}


template <>
__global__ void from_slice_kernel_backward<2>(
    float* __restrict__ a_data,                 //
    const int32_t* __restrict__ a_strides,      //
    const int* __restrict__ starts,             //
    const int32_t* __restrict__ new_strides,    //
    const int32_t* __restrict__ new_dims,       //
    const float* __restrict__ source_data,      //
    int total_elements                          //
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  int dim0_idx = (idx / new_strides[0]) % new_dims[0];
  int dim1_idx = idx % new_dims[1];

  int offset = (starts[0] + dim0_idx) * a_strides[0] + (starts[1] + dim1_idx) * a_strides[1];
  a_data[offset] += source_data[idx];
}


template <>
__global__ void from_slice_kernel_backward<4>(
    float* __restrict__ a_data,                 //
    const int32_t* __restrict__ a_strides,      //
    const int* __restrict__ starts,             //
    const int32_t* __restrict__ new_strides,    //
    const int32_t* __restrict__ new_dims,       //
    const float* __restrict__ source_data,      //
    int total_elements                          //
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int dim0_idx = (idx / new_strides[0]) % new_dims[0];
    int dim1_idx = (idx / new_strides[1]) % new_dims[1];
    int dim2_idx = (idx / new_strides[2]) % new_dims[2];
    int dim3_idx = idx % new_dims[3];

    int offset =
        (starts[0] + dim0_idx) * a_strides[0] +
        (starts[1] + dim1_idx) * a_strides[1] +
        (starts[2] + dim2_idx) * a_strides[2] +
        (starts[3] + dim3_idx) * a_strides[3];

    a_data[offset] += source_data[idx];
}

template <>
__global__ void from_slice_kernel_backward<5>(
    float* __restrict__ a_data,                 //
    const int32_t* __restrict__ a_strides,      // [5]  strides
    const int* __restrict__ starts,             // [5]
    const int32_t* __restrict__ new_strides,    // [5]  strides
    const int32_t* __restrict__ new_dims,       // [5]
    const float* __restrict__ source_data,      //
    int total_elements                          //
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int dim4 = idx % new_dims[4]; idx /= new_dims[4]; // W
    int dim3 = idx % new_dims[3]; idx /= new_dims[3]; // H
    int dim2 = idx % new_dims[2]; idx /= new_dims[2]; // D
    int dim1 = idx % new_dims[1]; idx /= new_dims[1]; // C
    int dim0 = idx;                                   // B

    int offset =
        (starts[0] + dim0) * a_strides[0] +
        (starts[1] + dim1) * a_strides[1] +
        (starts[2] + dim2) * a_strides[2] +
        (starts[3] + dim3) * a_strides[3] +
        (starts[4] + dim4) * a_strides[4];

    a_data[offset] += source_data[idx];
}