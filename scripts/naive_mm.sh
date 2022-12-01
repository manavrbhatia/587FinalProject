#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)
#SBATCH --job-name=587hw3omp
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f22_class
#SBATCH --output=outputs/slurm-output.log

module load gcc
g++ -o bin/matrixmult.o matrixmult.cpp
#./bin/matrixmult.o 256 f1 > outputs/strassens_cpu_output256f1.txt
#./bin/matrixmult.o 256 f2 > outputs/strassens_cpu_output256f2.txt
./bin/matrixmult.o 2048 f1 > outputs/strassens_cpu_output2048f1.txt
#./bin/matrixmult.o 2048 f2 > outputs/strassens_cpu_output2048f2.txt
