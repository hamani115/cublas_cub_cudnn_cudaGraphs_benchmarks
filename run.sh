# RUN
# cuBLAS 
./cublas/cublas_saxpy_bench_no_graph_chrono.out > ./cublas/cublas_no_graph.csv
./cublas/cublas_saxpy_bench_graph_chrono.out > ./cublas/cublas_graph.csv
# CUB
./cub/cub_bench_no_graph_chrono.out > ./cub/cub_no_graph.csv
./cub/cub_bench_graph_chrono.out > ./cub/cub_graph.csv
# cuDNN
./cudnn/cudnn_conv_bench_no_graph_chrono.out > ./cudnn/cudnn_no_graph.csv
./cudnn/cudnn_conv_bench_graph_chrono.out > ./cudnn/cudnn_graph.csv