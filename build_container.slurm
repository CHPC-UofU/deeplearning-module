#!/bin/bash
# SLURM script to construct the container.
#SBATCH -p notchpeak-gpu-guest
#SBATCH -A owner-gpu-guest
#SBATCH --gres=gpu
#SBATCH --mem=50G
#SBATCH --time=0-01:00:00
#SBATCH --mail-user=brett.milash@utah.edu
#SBATCH --mail-type=ALL

date +'Starting at %c.'
ml apptainer
make -f Makefile.deeplearning
exitcode=$?
date +'Finished at %c.'
exit $exitcode
