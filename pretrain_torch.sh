#!/bin/bash
#SBATCH --job-name=flowers_pretrain
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c.a.voinea@student.rug.nl
<<<<<<< HEAD
#SBATCH --partition=gpu
=======
#SBATCH --partition=gpus
>>>>>>> 393d7647a791dae206ebe3be705ee863fd29d905
#SBATCH --gres=gpu:v100:1

module purge
module load PyTorch/1.10.0-fosscuda-2020b jbigkit

source /data/$USER/.envs/deep_torch/bin/activate

cd Compact-Transformers

./dist_train.sh 1 -c configs/flowers.yml --pretrain --epochs 100 --log-wandb /data/$USER/deepl_data/flowers_dataset/

deactivate