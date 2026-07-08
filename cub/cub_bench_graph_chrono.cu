#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#include "../utils.h"

__global__ void fill_kernel(float* p, size_t n, float v) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

struct SizeCfg {
  const char* name;
  int N, C, H, W;
  size_t elems; // elems = N*C*H*W
};

int main() {
  using clock = std::chrono::high_resolution_clock;
  using d_ms  = std::chrono::duration<double, std::milli>;

  const int iters_list[3] = {100, 1000, 10000};
  const SizeCfg sizes[3] = {
    {"Small ",  1, 32,  32,  32,  size_t(1)*32*32*32},
    {"Medium",  8, 64, 112, 112,  size_t(8)*64*112*112},
    {"Big   ", 16, 64, 224, 224,  size_t(16)*64*224*224},
  };

  // CSV header (stdout only)
  printf("MODE,Size,N,C,H,W,Elems,iters,graph_create_ms,launch_ms,total_ms,launch_us/iter,total_us/iter,cpu_idle_us/iter\n");

  for (const auto& cfg : sizes) {
    const size_t N = cfg.elems;

    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // device buffers
    float *d_in=nullptr, *d_out=nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));

    {
      int threads = 256;
      int blocks  = int((N + threads - 1) / threads);
      // init d_in = 1.0f
      fill_kernel<<<blocks, threads, 0, s>>>(d_in, N, 1.0f);
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaStreamSynchronize(s));
    }

    // temp storage query
    void*  d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, (int)N, s);

    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));

    // warmup CUB before capture
    for (int i=0; i<20; ++i) {
      cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, (int)N, s);
    }
    CHECK_CUDA(cudaStreamSynchronize(s));

    // ################# graph creation #################
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;

    const auto g0 = clock::now();
    CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));

    // capture only reduce
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, (int)N, s);

    CHECK_CUDA(cudaStreamEndCapture(s, &g));
    CHECK_CUDA(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CHECK_CUDA(cudaGraphDestroy(g));
    const auto g1 = clock::now();

    const double graph_create_ms = d_ms(g1 - g0).count();

    // warmup graph
    for (int i=0; i<20; ++i) CHECK_CUDA(cudaGraphLaunch(ge, s));
    CHECK_CUDA(cudaStreamSynchronize(s));

    for (int iters : iters_list) {
      CHECK_CUDA(cudaStreamSynchronize(s));

      const auto t0 = clock::now();
      for (int i=0; i<iters; ++i) {
        CHECK_CUDA(cudaGraphLaunch(ge, s));
      }
      // before sync = launch time
      const auto t1 = clock::now();
      CHECK_CUDA(cudaStreamSynchronize(s));
      // after sync = total time
      const auto t2 = clock::now();

      const double launch_ms = d_ms(t1 - t0).count();
      const double total_ms  = d_ms(t2 - t0).count();

      const double launch_us_per_iter   = (launch_ms * 1000.0) / iters;
      const double total_us_per_iter    = (total_ms  * 1000.0) / iters;
      const double cpu_idle_us_per_iter = ((total_ms - launch_ms) * 1000.0) / iters;

      // Validation
      float h_out = 0.0f;
      CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
      const float expected = float(N);
      if (fabsf(h_out - expected) > 1e-2f) {
        fprintf(stderr, "Validation FAIL for %s iters=%d: got %.1f expected %.1f\n",
                cfg.name, iters, h_out, expected);
      }

      // CSV row
      printf("Graph,%s,%d,%d,%d,%d,%zu,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
             cfg.name, cfg.N, cfg.C, cfg.H, cfg.W, N, iters,
             graph_create_ms,
             launch_ms, total_ms,
             launch_us_per_iter, total_us_per_iter, cpu_idle_us_per_iter);
    }

    // cleanup
    CHECK_CUDA(cudaGraphExecDestroy(ge));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaStreamDestroy(s));
  }

  return 0;
}