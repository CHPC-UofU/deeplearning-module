#!/bin/bash
# SLURM script to construct the container.
#SBATCH -p notchpeak-shared
#SBATCH -A chpc
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --mail-user=brett.milash@utah.edu
#SBATCH --mail-type=ALL
#SBATCH -o build_container.out

date +'Starting at %R.'
ml singularity
make -f Makefile.deeplearning
date +'Finished at %R.'
