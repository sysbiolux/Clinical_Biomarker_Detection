#!/bin/bash -l

#SBATCH -J long-parallel-job
#SBATCH -N 2
#SBATCH -c 8
#SBATCH --ntasks 32
#SBATCH --time=10-00:00:00
#SBATCH --qos=long
#SBATCH -p batch
#SBATCH --mail-type=END,FAIL
##SBATCH --mail-use=john.doe@example.com  # complete and uncomment if desired

# Safeguard for NOT running this launcher on access/login nodes
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }

# Set language module and environment path to load, adapt if necessary
language_to_load=lang/Python/3.8.6-GCCcore-10.2.0
environment_to_load=./ML_env_${ULHPC_CLUSTER}/bin/activate

# load modules
module purge || print_error_and_exit # No 'module' command
module load ${language_to_load}

#activate python environment
source ${environment_to_load}

profile=job_${SLURM_JOB_ID}

echo "Creating profile_${profile}"
ipython profile create ${profile}

# Number of tasks - 1 controller task - 1 python task
export NB_WORKERS=$((${SLURM_NTASKS}-2))

# Set and create log directory
LOG_DIR="$(pwd)/logs/job_${SLURM_JOB_ID}"
mkdir -p ${LOG_DIR}

#srun: runs ipcontroller -- forces to start on first node
srun -w $(scontrol show hostname ${SLURM_NODELIST} | head -n 1) --output=${LOG_DIR}/ipcontroller-%j-workers.out --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} ipcontroller --ip="*" --nodb --profile=${profile} &
sleep 10

#srun: starts job
echo "Launching job for script $1"
srun -w $(scontrol show hostname ${SLURM_NODELIST} | tail -n 1) --output=${LOG_DIR}/code-%j-execution.out --exclusive -N 1 -n 1 -c ${SLURM_CPUS_PER_TASK} -u python $1 -p ${profile}  &

#srun: runs ipengine on each available core -- controller location first node
srun --output=${LOG_DIR}/ipengine-%j-workers.out --exclusive -N ${SLURM_NNODES} -n ${NB_WORKERS} -c ${SLURM_CPUS_PER_TASK} ipengine --profile=${profile} --location=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1) &
wait
