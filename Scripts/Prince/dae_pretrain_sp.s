#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=dae2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dlc423@nyu.edu
#SBATCH --output=slurm_%j.out

module load python3/intel/3.6.3 cudnn/9.0v7.0.5 cuda/9.0.176
source ~/pyenv/py3.6.3/bin/activate

export BASE_DIR=~/Shared/LeProject/DAE
export NOISE_TYPE=sp
export NOISE_PERC=.05
export VALID=False


python $BASE_DIR/pretrain.py --perc_noise=$NOISE_PERC --corr_type=$NOISE_TYPE --valid=$VALID
