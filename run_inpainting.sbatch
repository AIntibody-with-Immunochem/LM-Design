#!/bin/bash
#SBATCH --job-name=LMDesign         # short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=56G           # memory per cpu-core 
#SBATCH --time=05:00:00             # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:A100:1           # number of GPUs per node
#SBATCH --partition=gpu             # requested partition
#SBATCH --account=p_2024001_hpccom_immunochem     # billing account

# ------------------------------------------------------------------------------
# Print some info
# ------------------------------------------------------------------------------
echo "job_id             : $SLURM_JOB_ID"
echo "jobname            : $SLURM_JOB_NAME"
echo "number of mpi tasks: $SLURM_NTASKS tasks"
echo "OMP_NUM_THREADS    : $OMP_NUM_THREADS"

# ------------------------------------------------------------------------------
# Activate Conda Environment
# ------------------------------------------------------------------------------
module load devel/python/Anaconda3-2022.05  # Load Anaconda if required
source $(conda info --base)/etc/profile.d/conda.sh  # Ensure `conda activate` works

# Activate your IgDesign Conda environment
conda activate /gpfs/projects/immunochem/p_2024001_hpccom_immunochem/.conda/envs/LMDesign

# **Force the Job to Use the Conda Python**
export PATH=/gpfs/projects/immunochem/p_2024001_hpccom_immunochem/.conda/envs/LMDesign/bin:$PATH
export CONDA_PREFIX=/gpfs/projects/immunochem/p_2024001_hpccom_immunochem/.conda/envs/LMDesign
export CONDA_DEFAULT_ENV=antifold
export PYTHONNOUSERSITE=1  # Prevent interference from user-wide packages

# Verify Python Path to ensure Conda's Python is used
echo "Using Python from: $(which python)"
python --version

# ------------------------------------------------------------------------------
# Run the AntiFold inference
# ------------------------------------------------------------------------------
# Go to the IgDesign directory
cd /gpfs/projects/immunochem/p_2024001_hpccom_immunochem/garmstro/LM-Design

# Start the inference
pwd
date
echo "Starting LM-Design (Inpainting) inference... "
python inpainting.py
wait
echo "Inference complete"
date


