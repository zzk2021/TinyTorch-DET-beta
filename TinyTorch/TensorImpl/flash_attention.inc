#pragma once

namespace TinyTorch {


__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    // 获取当前线程在块内的索引
    int tx = threadIdx.x;
    // 获取当前块在网格中的索引（batch和head索引）
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // 计算Q,K,V,O,l,m在全局内存中的偏移量 - 每个batch和head都不同
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // l和m的偏移量

    // 在共享内存中为Q,K,V,S定义空间
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // Qi, Kj, Vj的大小
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    // 外循环：遍历所有的K和V块
    for (int j = 0; j < Tc; j++) {

        // 将Kj, Vj加载到共享内存
        for (int x = 0; x < d; x++) {
            // Bc个线程，每个线程负责K的一列，注意转置之后，该矩阵列优先
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            // tx*d: 当前线程在Kj中的起始位置
            // x: 当前线程负责的列中的偏移量
            // qkv_offset: 当前batch和head的起始位置
            // tile_size * j: 当前K块的起始位置
            // (tx * d) + x: 当前K块内的具体位置

            // Bc个线程，每个线程负责V的一行，注意该矩阵行优先
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
            // tx*d: 当前线程在Vj中的起始位置
            // x: 当前线程负责的行中的偏移量
            // qkv_offset: 当前batch和head的起始位置
            // tile_size * j: 当前V块的起始位置
            // (tx * d) + x: 当前V块内的具体位置
        }
        __syncthreads();  // 确保内部循环可以使用正确的Kj, Vj

        // 内循环：遍历所有的Q块
        for (int i = 0; i < Tr; i++)  {

            // 将Qi加载到共享内存，将l和m加载到寄存器
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
                // tx*d: 当前线程在Qi中的起始位置
                // x: 当前线程负责的行中的偏移量
                // qkv_offset: 当前batch和head的起始位置
                // tile_size * i: 当前Q块的起始位置
                // (tx * d) + x: 当前Q块内的具体位置
            }

            float row_m_prev = m[lm_offset + (Br * i) + tx];
            // lm_offset: 当前batch和head的m起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行

            float row_l_prev = l[lm_offset + (Br * i) + tx];
            // lm_offset: 当前batch和head的l起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行

            // 计算S = QK^T，并找出每行的最大值row_m
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                    // Qi[(tx * d) + x]: 当前Q行的第x个元素
                    // Kj[(y * d) + x]: 当前K列的第x个元素
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                // Bc * tx: 当前线程在S中的起始位置
                // y: 当前列的偏移量

                if (sum > row_m)
                    row_m = sum;
            }

            // 计算P = exp(S - row_m)，并求每行的和row_l
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                // Bc * tx: 当前线程在S中的起始位置
                // y: 当前列的偏移量
                row_l += S[(Bc * tx) + y];
            }

            // 计算新的m和l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // 计算并写入O，更新l和m
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                    // S[(Bc * tx) + y]: 当前S行的第y个元素
                    // Vj[(y * d) + x]: 当前V行的第x个元素
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
                // qkv_offset: 当前batch和head的O起始位置
                // tile_size * i: 当前O块的起始位置
                // (tx * d) + x: 当前O块内的具体位置
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            // lm_offset: 当前batch和head的m起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行
            l[lm_offset + (Br * i) + tx] = row_l_new;
            // lm_offset: 当前batch和head的l起始位置
            // Br * i: 当前Q块的起始行
            // tx: 当前线程对应的行
        }
        __syncthreads();  // 防止线程在内部循环中使用错误的Kj, Vj
    }
}
}