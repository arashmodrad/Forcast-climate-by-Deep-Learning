#!/bin/bash

#Account and Email Information
#SBATCH -A amodaresirad
#SBATCH --mail-type=end
#SBATCH --mail-user=arashmodaresirad@u.boisestate.edu

#SBATCH -J Final        # job name
#SBATCH -o outputs/results_final.o%j # output and error file name (%j expands to jobID)
#SBATCH -e outputs/errors_final.e%j
#SBATCH -n 1               # Run one process
#SBATCH --cpus-per-task=28 # allow job to multithread across all cores
#SBATCH -t 5-00:00:00      # run time (d-hh:mm:ss)
ulimit -v unlimited
ulimit -s unlimited
ulimit -u 1000

module load cuda10.0/toolkit/10.0.130 # loading cuda libraries/drivers 
module load python/intel/3.7          # loading python environment




# Variables
# ---------
env_out_file=outputs/environment.out


# Set up the environment
# ----------------------

# Modules
module load python/intel/3.7          # loading python environment
module load anaconda

# Conda
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate base
conda create -n Arash-project python=3.7 -c conda-forge -y -q > $env_out_file
echo "(Arash-project) environment created" >> $env_out_file
conda activate Arash-project
conda install -c conda-forge mamba -y -q >> $env_out_file
echo "mamba installed" >> $env_out_file
mamba env update -n Arash-project -f environment.yml --prune -q >> $env_out_file
conda activate Arash-project
echo "(Arash-project) environment updated" >> $env_out_file
echo "Running scripts..."

# Development environment packages
python3 ./setup.py -q develop
echo "(autosegment) installed development environment" >> $env_out_file


# Run the scripts
# ---------------

# Main python training
python3 Model_proj_final.py


# Cleanup
# -------

# Uninstall the conda environment
conda activate base
# conda env remove -n Arash-project -y -q >> $env_out_file