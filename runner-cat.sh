#!/bin/bash
echo "dataset: $1"
model_list=('categorical/deepfm' 'categorical/fm' 'categorical/attitemknn')
for model in "${model_list[@]}"
do
    echo "model: $model"
    sbatch --job-name=emo-rec --mail-user=andreas.peintner@uibk.ac.at --time=300:00:00 --mem=64G --gres=gpu:1 ~/jobs/single-node-gpu.job "conda run -n elliot python run_sep.py --data $1 --model $model"
    sleep 1
done
