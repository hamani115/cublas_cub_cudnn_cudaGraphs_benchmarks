#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUBLAS
#include "../utils.h"

__global__ void fill_kernel(float* p, size_t n, float v) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

struct SizeCfg {
  const char* name;
  int N, C, H, W;
  size_t L; // vector length = N*C*H*W
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

  // CSV header
  printf("MODE,Size,N,C,H,W,VecLen,iters,graph_create_ms,launch_ms,total_ms,launch_us/iter,total_us/iter,cpu_idle_us/iter\n");

  for (const auto& cfg : sizes) {
    const int n = (int)cfg.L;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    // device buffers
    float *dx=nullptr, *dy=nullptr, *d_alpha=nullptr;
    CHECK_CUDA(cudaMalloc(&dx, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dy, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(float)));

    {
      int threads = 256;
      int blocks  = (n + threads - 1) / threads;
      fill_kernel<<<blocks, threads, 0, stream>>>(dx, cfg.L, 1.0f);
      CHECK_CUDA(cudaGetLastError());

      float h_alpha = 2.5f;
      CHECK_CUDA(cudaMemcpyAsync(d_alpha, &h_alpha, sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // warmup
    {
      int threads = 256;
      int blocks  = (n + threads - 1) / threads;
      fill_kernel<<<blocks, threads, 0, stream>>>(dy, cfg.L, 3.0f);
      CHECK_CUDA(cudaGetLastError());

      for (int i=0; i<20; ++i) {
        CHECK_CUBLAS(cublasSaxpy(handle, n, d_alpha, dx, 1, dy, 1));
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    for (int iters : iters_list) {
      // Reset dy to 3.0
      {
        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        fill_kernel<<<blocks, threads, 0, stream>>>(dy, cfg.L, 3.0f);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));
      }

      const auto t0 = clock::now();
      for (int i=0; i<iters; ++i) {
        CHECK_CUBLAS(cublasSaxpy(handle, n, d_alpha, dx, 1, dy, 1));
      }

      // before sync = launch time
      const auto t1 = clock::now();
      CHECK_CUDA(cudaStreamSynchronize(stream));
      // after sync = total time
      const auto t2 = clock::now();

      const double launch_ms = d_ms(t1 - t0).count();
      const double total_ms  = d_ms(t2 - t0).count();

      const double launch_us_per_iter   = (launch_ms * 1000.0) / iters;
      const double total_us_per_iter    = (total_ms  * 1000.0) / iters;
      const double cpu_idle_us_per_iter = ((total_ms - launch_ms) * 1000.0) / iters;

      // Validation: dy[0] should be 3 + iters*(alpha*1) = 3 + 2.5*iters
      float y0 = 0.0f;
      CHECK_CUDA(cudaMemcpy(&y0, dy, sizeof(float), cudaMemcpyDeviceToHost));
      const float expected = 3.0f + 2.5f * float(iters);
      if (fabsf(y0 - expected) > 1e-2f) {
        fprintf(stderr, "Validation FAIL %s iters=%d: got %.3f expected %.3f\n",
                cfg.name, iters, y0, expected);
      }

      // CSV row
      printf("NoGraph,%s,%d,%d,%d,%d,%zu,%d,0.0,%.6f,%.6f,%.6f,%.6f,%.6f\n",
             cfg.name, cfg.N, cfg.C, cfg.H, cfg.W, cfg.L, iters,
             launch_ms, total_ms,
             launch_us_per_iter, total_us_per_iter, cpu_idle_us_per_iter);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
  }

  return 0;
}