#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

#SBATCH --job-name=587ProjectTiledCUDA
#SBATCH --partition=spgpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=eecs587f22_class
#SBATCH --output=outputs/slurm-output.log

ALGO="tiled_cuda"

nvcc -o bin/matrixmult_CUDA.o matrixmult.cu
./bin/matrixmult_CUDA.o 256 f1 > outputs/cuda/$ALGO\_256_f1.txt
./bin/matrixmult_CUDA.o 2048 f1 > outputs/cuda/$ALGO\_2048_f1.txt
./bin/matrixmult_CUDA.o 256 f2 > outputs/cuda/$ALGO\_256_f2.txt
./bin/matrixmult_CUDA.o 2048 f2 > outputs/cuda/$ALGO\_2048_f2.txt
