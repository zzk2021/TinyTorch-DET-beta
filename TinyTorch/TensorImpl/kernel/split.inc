
#pragma once

template <typename T>
__global__ void ppl_cukernel_split(
    const T* input_data,
    T* output0,
    T* output1,
    const int32_t* input_shape,
    int32_t split_size0,
    int32_t split_size1,
    int32_t dim,
    int32_t N)
{
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  int32_t flat_idx = idx;
  int32_t coords[4];
  for (int32_t i = input_shape[0] - 1; i >= 0; --i) {
    coords[i] = flat_idx % input_shape[i];
    flat_idx /= input_shape[i];
  }

  int32_t pos = coords[dim];
  if (pos < split_size0) {
    coords[dim] = pos;


    int32_t out_idx = 0;
    for (int32_t i = 0; i < input_shape[0]; ++i) {
      out_idx = coords[i] + out_idx * input_shape[i];
    }
    output0[out_idx] = input_data[idx];

  } else if (split_size1 > 0 && pos < split_size0 + split_size1) {
    coords[dim] = pos - split_size0;

    int32_t out_idx = 0;
    for (int32_t i = 0; i < input_shape[0]; ++i) {
      out_idx = coords[i] + out_idx * input_shape[i];
    }
    output1[out_idx] = input_data[idx];
  }
}

template <typename T>
__global__ void ppl_cukernel_concat_backward(
    const T* input_data,
    T* output0,
    T* output1,
    const int32_t* input_shape,
    int32_t split_size0,
    int32_t dim,
    int32_t N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;


    int remaining = tid;
    int coord[4] = {0};
    int strides[4] = {1, 1, 1, 1};

    for (int i = 3; i > 0; --i) {
        strides[i-1] = strides[i] * input_shape[i];
    }

    for (int i = 0; i < 4; ++i) {
        coord[i] = remaining / strides[i];
        remaining %= strides[i];
    }

    const int dim_coord = coord[dim];

    if (dim_coord < split_size0) {
        output0[tid] = input_data[tid];
    } else {
        output1[tid] = input_data[tid];
    }
}