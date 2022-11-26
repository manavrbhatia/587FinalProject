#!/bin/bash
#SBATCH --job-name=EECS587ProjectMPI
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem-per-cpu=1g
#SBATCH --time=00:05:00
#SBATCH --account=eecs587f22_class
#SBATCH --partition=standard
#SBATCH --output=outputs/slurm-output.log

# -np command sets number of processors to run program with. If you specify more processors
# using -np command than you have requisitioned using the #SBATCH --ntasks-per-node command
# your program will not run

# --bind-to core sets up mpi environment and specifies which hardware to bind MPI processes to
# you don't need to mess with this

module load openmpi
module load gcc

echo "Running Fox Algorithm" > outputs/mpi_output.txt
mpic++ -o bin/mpi_fox.o fox_mm.cpp
mpirun -np 36 --bind-to core:overload-allowed ./bin/mpi_fox.o  >> outputs/mpi_output.txt
