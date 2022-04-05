#!/bin/bash
#SBATCH --job-name=flowers_train
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

./dist_train.sh 1 -c configs/flowers.yml --weight-decay 0 --wandb-name train_wd_0 /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --drop 0.2 --wandb-name train_drop_0.2 /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --opt adagrad --wandb-name train_opt_adagrad /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --opt rmsprop --wandb-name train_opt_rmsprop /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --no-aug --wandb-name train_no_aug /data/$USER/deepl_data/flowers_dataset/

deactivate
