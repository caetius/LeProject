#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=48:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=dae1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dlc423@nyu.edu
#SBATCH --output=slurm_%j.out

module load python-3.6 cuda-9.0
source ~/pyenv/py3.6.3/bin/activate

export BASE_DIR=~/Shared/LeProject/DAE
export NOISE_TYPE=gauss
export NOISE_PERC=.05
export VALID=False

python $BASE_DIR/pretrain.py --perc_noise=$NOISE_PERC --corr_type=$NOISE_TYPE --valid=$VALID
