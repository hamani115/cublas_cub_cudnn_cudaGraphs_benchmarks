
# COMPILE
# cuBLAS
/usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cublas/cublas_saxpy_bench_no_graph_chrono.cu -lcublas -o cublas/cublas_saxpy_bench_no_graph_chrono.out
/usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cublas/cublas_saxpy_bench_graph_chrono.cu -lcublas -o cublas/cublas_saxpy_bench_graph_chrono.out

# CUB
/usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cub/cub_bench_no_graph_chrono.cu -o cub/cub_bench_no_graph_chrono.out
/usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cub/cub_bench_graph_chrono.cu -o cub/cub_bench_graph_chrono.out

#cuDNN
# /usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cudnn/cudnn_bench_no_graph_chrono.cu -I$SPACK_ROOT/opt/spack/linux-zen2/cudnn-9.8.0.87-12-5mdht7ecf2jk33xka46agfa6hyei2odr/include/ -L$LD_LIBRARY_PATH/ -lcudnn -o cudnn_bench_no_graph_chrono.out
# /usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cudnn/cudnn_bench_graph_chrono.cu -I$SPACK_ROOT/opt/spack/linux-zen2/cudnn-9.8.0.87-12-5mdht7ecf2jk33xka46agfa6hyei2odr/include/ -L$LD_LIBRARY_PATH/ -lcudnn -o cudnn_bench_graph_chrono.out

# source /data/software/packages/spack/setup-env.sh
# spack load cudnn

# /usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cudnn/cudnn_bench_no_graph_chrono.cu \
#     -I$SPACK_ROOT/opt/spack/linux-zen2/cudnn-9.8.0.87-12-5mdht7ecf2jk33xka46agfa6hyei2odr/include/ \
#     -L$LD_LIBRARY_PATH/ \
#     -lcudnn -o cudnn/cudnn_bench_no_graph_chrono.out

# /usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cudnn/cudnn_bench_graph_chrono.cu \
#     -I$SPACK_ROOT/opt/spack/linux-zen2/cudnn-9.8.0.87-12-5mdht7ecf2jk33xka46agfa6hyei2odr/include/ \
#     -L$LD_LIBRARY_PATH/ \
#     -lcudnn -o cudnn/cudnn_bench_graph_chrono.out

/usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cudnn/cudnn_conv_bench_no_graph_chrono.cu \
    -I$SPACK_ROOT/opt/spack/linux-zen2/cudnn-9.8.0.87-12-5mdht7ecf2jk33xka46agfa6hyei2odr/include/ \
    -L$LD_LIBRARY_PATH/ \
    -lcudnn -o cudnn/cudnn_conv_bench_no_graph_chrono.out

/usr/local/cuda/bin/nvcc -O2 -arch=sm_80 cudnn/cudnn_conv_bench_graph_chrono.cu \
    -I$SPACK_ROOT/opt/spack/linux-zen2/cudnn-9.8.0.87-12-5mdht7ecf2jk33xka46agfa6hyei2odr/include/ \
    -L$LD_LIBRARY_PATH/ \
    -lcudnn -o cudnn/cudnn_conv_bench_graph_chrono.out