# CUDA Graphs Library Benchmarks

Small benchmark repo for comparing ML CUDA library calls with and without CUDA Graphs. The goal is to measure how much CUDA Graphs reduce repeated launch overhead compared to normal CUDA library execution.

The repo currently includes benchmarks for:

* cuBLAS
* CUB
* cuDNN

Each benchmark has a normal version and a CUDA Graph version.

## Requirements

* NVIDIA GPU
* CUDA Toolkit
* cuBLAS
* CUB
* cuDNN
* Python 3 for plotting

Clone repo:

```bash
git clone https://github.com/hamani115/cublas_cub_cudnn_cudaGraphs_benchmarks
cd cublas_cub_cudnn_cudaGraphs_benchmarks
```

Install Python packages:

```bash
# Linux commands
pip -m venv my_venv
source my_venv/bin/activate

pip install -r requirements.txt
```

## Build

```bash
bash build.sh
```

Note: you may need to edit `build.sh` to match your GPU architecture, CUDA path, or cuDNN installation path.

## Run

```bash
bash run.sh
```

This generates CSV result files inside the benchmark folders.

## Plot Results

```bash
bash plot.sh
```

Plots will be saved in the `plots/` directory.

## Repository Structure

```text
cublas/   cuBLAS benchmark files
cub/      CUB benchmark files
cudnn/    cuDNN benchmark files
build.sh  Compile benchmarks
run.sh    Run benchmarks and save CSV files
plot.sh   Generate plots
```
