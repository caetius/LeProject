#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 3
#SBATCH -c 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=name_of_run
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_email
#SBATCH --output=slurm_%j.out

module load python3/intel/3.6.3 cudnn/9.0v7.0.5 cuda/9.0.176
source ~/pyenv/py3.6.3/bin/activate (your_env)

export BASE_DIR=~/Shared/LeProject/Split_Brain (your_project_root_dir)
export BATCH=64
export FOLDER=weights_$BATCH
export IMG_SPACE1=rgb
export IMG_SPACE2=lab
export IMG_SPACE3=lab_distort

export LRD=0.5
export MODEL=simple

srun -N 1 -n 1 python $BASE_DIR/split_brain_pretrain.py --model_type=$MODEL --weights_folder=$FOLDER --verbose=False --wandb_on=True --batch_size=$BATCH --lr_decay=$LRD --image_space=$IMG_SPACE1 &
srun -N 1 -n 1 python $BASE_DIR/split_brain_pretrain.py --model_type=$MODEL --weights_folder=$FOLDER --verbose=False --wandb_on=True --batch_size=$BATCH --lr_decay=$LRD --image_space=$IMG_SPACE2 &
srun -N 1 -n 1 python $BASE_DIR/split_brain_pretrain.py --model_type=$MODEL --weights_folder=$FOLDER --verbose=False --wandb_on=True --batch_size=$BATCH --lr_decay=$LRD --image_space=$IMG_SPACE3
wait
