#!/bin/bash -l

#SBATCH -J BASE-2-cluster
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --ntasks 1
#SBATCH --time=01-06:00:00
#SBATCH -p batch
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-use=jeff.didier@uni.lu

# Safeguard for NOT running this launcher on access/login nodes
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }

# Set language module and environment path to load
language_to_load=lang/Python/3.8.6-GCCcore-10.2.0
environment_to_load=./ML_env_${ULHPC_CLUSTER}/bin/activate

# load modules
module purge || print_error_and_exit # No 'module' command
module load ${language_to_load}

#activate python environment
source ${environment_to_load}

LOG_DIR="$(pwd)/logs/job_${SLURM_JOB_ID}"
mkdir -p ${LOG_DIR}

#srun: starts job
echo "Launching job for script $1"
srun -w $(scontrol show hostname ${SLURM_NODELIST} | head -n 1) --output=${LOG_DIR}/code-%j-execution.out --exclusive -N 1 -n 1 -c 16 -u python $1
