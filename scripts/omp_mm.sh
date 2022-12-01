#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

#SBATCH --job-name=587hw3omp
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f22_class
#SBATCH --output=outputs/slurm-output.log

module load openmpi
> outputs/omp/omp_mm.txt
g++ -O3 -o bin/matrixmult.o -fopenmp matrixmult.cpp

ALGO="omp_naive"
OMP_NUM_THREADS="32" ./bin/matrixmult.o 256 f1 >> outputs/omp/$ALGO\_256_f1_$OMP_NUM_THREADS.txt
OMP_NUM_THREADS="32" ./bin/matrixmult.o 2048 f1 >> outputs/omp/$ALGO\_2048_f1_$OMP_NUM_THREADS.txt
OMP_NUM_THREADS="32" ./bin/matrixmult.o 256 f2 >> outputs/omp/$ALGO\_256_f2_$OMP_NUM_THREADS.txt
OMP_NUM_THREADS="32" ./bin/matrixmult.o 2048 f2 >> outputs/omp/$ALGO\_2048_f2_$OMP_NUM_THREADS.txt
OMP_NUM_THREADS="4" ./bin/matrixmult.o 2048 f2 >> outputs/omp/$ALGO\_2048_f2_$OMP_NUM_THREADS.txt
OMP_NUM_THREADS="8" ./bin/matrixmult.o 2048 f2  >> outputs/omp/$ALGO\_2048_f2_$OMP_NUM_THREADS.txt
OMP_NUM_THREADS="16" ./bin/matrixmult.o 2048 f2  >> outputs/omp/$ALGO\_2048_f2_$OMP_NUM_THREADS.txt
