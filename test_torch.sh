#!/bin/bash
#SBATCH --job-name=tinyImageNet_test
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-user=c.a.voinea@student.rug.nl
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1

module purge
module load PyTorch/1.10.0-fosscuda-2020b jbigkit

source /data/$USER/.envs/deep_torch/bin/activate

cd Deep_Learning_Lab2/Compact-Transformers

./dist_train.sh 1 -c configs/tiny_imagenet.yml --model cct_14_7x2_224 --pretrain --epochs 2 /data/$USER/deepl_data/tiny-imagenet-200/train/


deactivate