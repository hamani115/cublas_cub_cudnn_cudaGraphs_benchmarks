#ifndef UTILS
#define UTILS

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do {                                   \
  cudaError_t error = (x);                                   \
  if (error != cudaSuccess) {                                \
    fprintf(stderr, "CUDA error %s at %s:%d\n",              \
            cudaGetErrorString(error), __FILE__, __LINE__);  \
    std::exit(1);                                            \
  }                                                          \
} while (0)

#ifdef CUDNN
#include <cudnn.h>
#define CHECK_CUDNN(x) do {                                  \
  cudnnStatus_t status = (x);                                \
  if (status != CUDNN_STATUS_SUCCESS) {                      \
    fprintf(stderr, "cuDNN error %s at %s:%d\n",             \
            cudnnGetErrorString(status), __FILE__, __LINE__);\
    std::exit(1);                                            \
  }                                                          \
} while (0)
#endif

#ifdef CUBLAS
#include <cublas_v2.h>
#define CHECK_CUBLAS(x) do {                                 \
  cublasStatus_t status = (x);                               \
  if (status != CUBLAS_STATUS_SUCCESS) {                     \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n",            \
            int(status), __FILE__, __LINE__);                \
    std::exit(1);                                            \
  }                                                          \
} while (0)
#endif

#endif