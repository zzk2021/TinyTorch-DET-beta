
#pragma once
#define kBlockSize 256
// CUDA: grid stride looping
#define CUDA_1D_KERNEL_LOOP(i, n)                                                                 \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

namespace TinyTorch {
template<typename T = float>
struct alignas(2 * sizeof(T)) Pack2X {
  T x;
  T y;
};

template<typename T = float>
__global__ void UpsampleNearest2D2XForward(const int32_t in_elem_cnt, const T* in_dptr,
                                           const int32_t in_height, const int32_t in_width,
                                           T* out_dptr) {
  const int32_t in_hw_size = in_width * in_height;
  CUDA_1D_KERNEL_LOOP(index, in_elem_cnt) {
    const T in_value = in_dptr[index];
    const int32_t nc_idx = index / in_hw_size;
    const int32_t hw_off = index - nc_idx * in_hw_size;
    const int32_t h = hw_off / in_width;
    const int32_t w = hw_off - h * in_width;
    Pack2X<T> out_value{in_value, in_value};
    Pack2X<T>* out_pack_dptr = reinterpret_cast<Pack2X<T>*>(out_dptr);
    out_pack_dptr[nc_idx * in_hw_size * 2 + h * 2 * in_width + w] = out_value;
    out_pack_dptr[nc_idx * in_hw_size * 2 + (h * 2 + 1) * in_width + w] = out_value;
  }
}

template<typename T = float>
__global__ void UpsampleNearest2D2XBackward(const int32_t in_elem_cnt, const T* dy_dptr,
                                            const int32_t dx_height, const int32_t dx_width,
                                            T* dx_dptr) {
  const int32_t dx_hw_size = dx_height * dx_width;
  CUDA_1D_KERNEL_LOOP(index, in_elem_cnt) {
    T dx_value = 0.0;
    const int32_t nc_idx = index / dx_hw_size;
    const int32_t dx_hw_off = index - nc_idx * dx_hw_size;
    const int32_t dx_h = dx_hw_off / dx_width;
    const int32_t dx_w = dx_hw_off - dx_h * dx_width;
    const Pack2X<T>* dy_pack_dptr = reinterpret_cast<const Pack2X<T>*>(dy_dptr);
    const Pack2X<T> dy_pack_value1 =
        dy_pack_dptr[nc_idx * dx_hw_size * 2 + dx_h * 2 * dx_width + dx_w];
    const Pack2X<T> dy_pack_value2 =
        dy_pack_dptr[nc_idx * dx_hw_size * 2 + (dx_h * 2 + 1) * dx_width + dx_w];
    dx_value += dy_pack_value1.x;
    dx_value += dy_pack_value1.y;
    dx_value += dy_pack_value2.x;
    dx_value += dy_pack_value2.y;
    dx_dptr[index] = dx_value;
  }
}
}