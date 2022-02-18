#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 23:00:00
#SBATCH --mem 10G

#SBATCH --gres=gpu:1

#SBATCH --job-name sartorileo_job
#SBATCH --output output_%j.txt
#SBATCH --error errors_%j.txt
#SBATCH --mail-user sartorileo@dei.unipd.it

# setup my personal/shared project folder
MY_DIR=/nfsd/hda/sartorileo

# make my folder the current directory
cd $MY_DIR

# setup python3 environment for machine learning computations 
source /nfsd/opt/anaconda3/anaconda3.sh
conda activate /nfsd/signet/hda_env_gpu

# run code
srun python /nfsd/hda/sartorileo/src_leonardo/kws_engine/batchInputTest.py

# deactivate environment at the end of the job
conda deactivate
