#pragma once
namespace TinyTorch {
__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {

    int tx = threadIdx.x;

    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  //

    extern __shared__ float sram[];
    int tile_size = Bc * d;  //
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        for (int x = 0; x < d; x++) {

            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];


            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];

        }
        __syncthreads();


        for (int i = 0; i < Tr; i++)  {


            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];

            }

            float row_m_prev = m[lm_offset + (Br * i) + tx];


            float row_l_prev = l[lm_offset + (Br * i) + tx];


            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];

                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;


                if (sum > row_m)
                    row_m = sum;
            }

            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);

                row_l += S[(Bc * tx) + y];
            }


            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];

                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));

            }
            m[lm_offset + (Br * i) + tx] = row_m_new;

            l[lm_offset + (Br * i) + tx] = row_l_new;

        }
        __syncthreads();
    }
}
}