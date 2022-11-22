#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

#SBATCH --job-name=587ProjectNaiveCUDA
#SBATCH --partition=spgpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=eecs587f22_class
#SBATCH --output=outputs/slurm-output.log

nvcc -o matrixmult_CUDA.o matrixmult.cu
./matrixmult_CUDA.o > outputs/cuda/out_naive_matrixmult.txt