#!/bin/bash
#SBATCH --job-name=flowers_pretrain
#SBATCH --time=07:30:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=c.a.voinea@student.rug.nl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

module purge
module load PyTorch/1.10.0-fosscuda-2020b jbigkit

source /data/$USER/.envs/deep_torch/bin/activate

cd Compact-Transformers

./dist_train.sh 1 -c configs/flowers.yml --weight-decay 0 --pretrain --wandb-name pretrain_wd_0 --output/data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --drop 0.2 --pretrain --wandb-name pretrain_drop_0.2 /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --opt adagrad --pretrain --wandb-name pretrain_opt_adagrad /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --opt adahessian --pretrain --wandb-name pretrain_opt_adahessian /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --no-aug --pretrain --wandb-name pretrain_no_aug /data/$USER/deepl_data/flowers_dataset/

deactivate