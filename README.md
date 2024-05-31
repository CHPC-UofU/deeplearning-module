# deeplearning-module

This repo contains files to construct the container for the CHPC deeplearning
module.

The container is based on docker://tensorflow/tensorflow:latest-gpu-jupyter .

Version 2023.3 included sklearn and pandas. 2024.1 is missing those,
so need to add those and add checks for those and other required
modules prior to publishing.
Also need to add JAX and FLAX, pillow, seaborn and matplotlib

beginner.py
build_container.slurm
check_modules.ipynb
check_modules.py
get_versions.py
Makefile.deeplearning
Singularity.deeplearning
test_tensorflow_installation.ipynb
test_tensorflow_installation.py
torch_confirm_cuda.ipynb
