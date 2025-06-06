#!/bin/bash

#SBATCH --job-name=optimize_neuro
#SBATCH --output=output_%j.log        # Standard output and error log
#SBATCH --error=error_%j.log
#SBATCH --ntasks=100                    # Total number of MPI processes
#SBATCH --ntasks-per-node=48
#SBATCH --time=1-72:00:00               # Walltime (HH:MM:SS)
#SBATCH --partition=big               # Replace with your cluster's partition/queue
#SBATCH --mail-type=END,FAIL          # Optional: email on job end/fail
#SBATCH --mail-user=bankowski@zib.de  # Optional: your email


cd /data/numerik/people/abankowski/neuro_param_estimation
# Activate your virtualenv if using one
source ./bin/activate

cd /data/numerik/people/abankowski/neuro_param_estimation/codes

# Run the Python script with mpirun
mpirun -np $SLURM_NTASKS python opt_DE_fixed_params.py