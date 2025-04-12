
#include "Torch.h"

using namespace TinyTorch;

// Training settings
struct TrainArgs {
  // input batch size for training (default: 64)
  int32_t batchSize = 64;

  // input batch size for testing (default: 1000)
  int32_t testBatchSize = 1000;

  // number of epochs to train (default: 1)
  int32_t epochs = 1;

  // learning rate (default: 1.0)
  float lr = 1.f;

  // Learning rate step gamma (default: 0.7)
  float gamma = 0.7f;

  // disables CUDA training
  bool noCuda = false;

  // quickly check a single pass
  bool dryRun = false;

  // random seed (default: 1)
  unsigned long seed = 1;

  // how many batches to wait before logging training status
  int32_t logInterval = 10;

  // For Saving the current Model
  bool saveModel = false;
};

Tensor forward(Tensor Q, Tensor K, Tensor V) {
    // TODO: 动态确定Bc, Br
  Timer timer;
  timer.start();

  TrainArgs args;

  manualSeed(args.seed);
    auto useCuda = (!args.noCuda) && Tensor::deviceAvailable(Device::CUDA);
    Device device = useCuda ? Device::CUDA : Device::CPU;

    const int Bc = 32; const int Br = 32;

    // 获取输入张量的维度
    const int B = Q.shape()[2]; const int nh = Q.shape()[1];
    const int N = Q.shape()[2]; const int d = Q.shape()[3];

    // 计算块的数量
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // 在GPU内存中初始化O, l, m
    auto O = Tensor::zeros(Q.shape());
    auto l = Tensor::zeros({B, nh, N});
    auto m = Tensor::zeros({B, nh, N});
    TensorImpl::fill(m.data(), -INFINITY);
    l = l.to(device); m = m.to(device);

    // 计算每个块需要的共享内存大小
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    /*
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    // 设置网格和块的维度
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // 每个块Bc个线程

    // 启动kernel
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
     */
    return O;
}