#!/bin/bash

#SBATCH --job-name=pytorch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=5:00:00
#SBATCH --partition=a100-gpu,beta-gpu
#SBATCH --output=slurm_log/ablation/img_input/dDIP/200/run-%j.log
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

# Path to the package directory
PACKAGE_PATH=/nas/longleaf/home/cctsai/.local/lib/python3.9/site-packages

# Add the package path to Python's sys.path
export PYTHONPATH="$PYTHONPATH:$PACKAGE_PATH"

module load cuda/11.8
module load python/3.9.6

cd /nas/longleaf/home/cctsai/proj_dip/

python submission_ablation.py --file /work/users/c/c/cctsai/data/BCP_sample/17_T1w_MPR_NORM_3.npy --factor 2 --model_folder ablation/img_input/dDIP --input_img --kspace_mse --kspace_boundary
