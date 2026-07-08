#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDNN
#include "../utils.h"

__global__ void fill_kernel(float* p, size_t n, float v) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

struct SizeCfg {
  const char* name;
  int N, C, H, W;
  int K, R, S;
};

int main() {
  using clock = std::chrono::high_resolution_clock;
  using d_ms  = std::chrono::duration<double, std::milli>;

  const int iters_list[3] = {100, 1000, 10000};
  const SizeCfg sizes[3] = {
    {"Small ",  1, 32,  32,  32,  32, 3, 3},
    {"Medium",  8, 64, 112, 112, 128, 3, 3},
    {"Big   ", 16, 64, 224, 224, 128, 3, 3},
  };


  printf("MODE,Size,N,C,H,W,K,R,S,iters,graph_create_ms,launch_ms,total_ms,launch_us/iter,total_us/iter,cpu_idle_us/iter\n");

  // conv params
  const int padH=1, padW=1, strideH=1, strideW=1, dilH=1, dilW=1;
  const float alpha = 1.0f, beta = 0.0f;

  for (const auto& cfg : sizes) {
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&wDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, cfg.N, cfg.C, cfg.H, cfg.W));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, cfg.K, cfg.C, cfg.R, cfg.S));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
        padH, padW, strideH, strideW, dilH, dilW,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int oN,oK,oH,oW;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &oN,&oK,&oH,&oW));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, oN,oK,oH,oW));

    // device buffers
    const size_t x_elems = size_t(cfg.N) * cfg.C * cfg.H * cfg.W;
    const size_t w_elems = size_t(cfg.K) * cfg.C * cfg.R * cfg.S;
    const size_t y_elems = size_t(oN) * oK * oH * oW;

    float *d_x=nullptr, *d_w=nullptr, *d_y=nullptr;
    CHECK_CUDA(cudaMalloc(&d_x, x_elems*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_w, w_elems*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, y_elems*sizeof(float)));

    // device input weights from host -> device
    {
      int threads = 256;
      int blocksX = int((x_elems + threads - 1) / threads);
      int blocksY = int((y_elems + threads - 1) / threads);
      fill_kernel<<<blocksX, threads, 0, stream>>>(d_x, x_elems, 1.0f);
      fill_kernel<<<blocksY, threads, 0, stream>>>(d_y, y_elems, 0.0f);
      CHECK_CUDA(cudaGetLastError());

      std::vector<float> h_w(w_elems, 0.0f);
      const int center = (cfg.R/2)*cfg.S + (cfg.S/2);
      for (int k=0; k<cfg.K; ++k) {
        for (int c=0; c<cfg.C; ++c) {
          h_w[(k*cfg.C + c)*cfg.R*cfg.S + center] = 1.0f;
        }
      }
      CHECK_CUDA(cudaMemcpyAsync(d_w, h_w.data(), w_elems*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // algo
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t ws_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &ws_bytes));
    void* d_ws = nullptr;
    if (ws_bytes) CHECK_CUDA(cudaMalloc(&d_ws, ws_bytes));

    // warmup
    for (int i=0; i<20; ++i) {
      CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w, convDesc,
                                       algo, d_ws, ws_bytes, &beta, yDesc, d_y));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // action
    for (int iters : iters_list) {
      CHECK_CUDA(cudaStreamSynchronize(stream));
      const auto t0 = clock::now();

      for (int i=0; i<iters; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, xDesc, d_x, wDesc, d_w, convDesc,
                                         algo, d_ws, ws_bytes, &beta, yDesc, d_y));
      }

      // before sync = launch time
      const auto t1 = clock::now();
      CHECK_CUDA(cudaStreamSynchronize(stream));
      // after sync = total time
      const auto t2 = clock::now();

      const double launch_ms = d_ms(t1 - t0).count();
      const double total_ms  = d_ms(t2 - t0).count();
      const double launch_us_per_iter = (launch_ms * 1000.0) / iters;
      const double total_us_per_iter  = (total_ms  * 1000.0) / iters;
      const double cpu_idle_us_per_iter = ((total_ms - launch_ms) * 1000.0) / iters;

      printf("NoGraph,%s,%d,%d,%d,%d,%d,%d,%d,%d,0.0,%.6f,%.6f,%.6f,%.6f,%.6f\n",
             cfg.name, cfg.N, cfg.C, cfg.H, cfg.W, cfg.K, cfg.R, cfg.S, iters,
             launch_ms, total_ms, launch_us_per_iter, total_us_per_iter, cpu_idle_us_per_iter);
    }

    // cleanup
    if (d_ws) cudaFree(d_ws);
    cudaFree(d_y); cudaFree(d_w); cudaFree(d_x);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle);
    cudaStreamDestroy(stream);
  }

  return 0;
}