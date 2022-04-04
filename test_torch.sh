#!/bin/bash
#SBATCH --job-name=flowers_test
#SBATCH --time=00:20:00
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --mail-type=END
#SBATCH --mail-user=c.a.voinea@student.rug.nl
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1

module purge
module load PyTorch/1.10.0-fosscuda-2020b jbigkit

source /data/$USER/.envs/deep_torch/bin/activate

cd Compact-Transformers

./dist_train.sh 1 -c configs/flowers.yml --warmup-epochs 0 --cooldown-epochs 0 --weight-decay 0 --pretrain --epochs 2 --log-wandb --wandb-name test_wd_0 /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --warmup-epochs 0 --cooldown-epochs 0 --drop 0.2 --pretrain --epochs 2 --log-wandb --wandb-name test_drop_0.2 /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --warmup-epochs 0 --cooldown-epochs 0 --opt adagrad --pretrain --epochs 2 --log-wandb --wandb-name test_opt_adagrad /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --warmup-epochs 0 --cooldown-epochs 0 --opt adahessian --pretrain --epochs 2 --log-wandb --wandb-name test_opt_adahessian /data/$USER/deepl_data/flowers_dataset/

./dist_train.sh 1 -c configs/flowers.yml --warmup-epochs 0 --cooldown-epochs 0 --no-aug --pretrain --epochs 2 --log-wandb --wandb-name test_no_aug /data/$USER/deepl_data/flowers_dataset/


deactivate