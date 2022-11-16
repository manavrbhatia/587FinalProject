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

> outputs/omp/omp_mm.txt
OMP_NUM_THREADS="32" ./matrixmult.o >> outputs/omp/omp_mm.txt
