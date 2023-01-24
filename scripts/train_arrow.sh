#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:tesla:1
#SBATCH --exclude=falcon1,falcon2,falcon3

module load cuda/10.1.243
module load cudnn/7.6.5.32-10.1
module load broadwell/gcc/9.2.0

module load any/python/3.8.3-conda

conda activate ml_project2

cd /gpfs/space/home/lepson/ml/

python scripts/train_arrow.py