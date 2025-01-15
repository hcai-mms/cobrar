#!/bin/bash
echo "dataset: $1"
#model_list=('simple' 'bprmf' 'neumf' 'multivae' 'lightgcn' 'bivae')
#model_list=('simple' 'bprmf' 'neumf' 'multivae' 'lightgcn' 'bivae' 'featureitemknn' 'multimodal/bm3' 'multimodal/clcrec' 'multimodal/freedom' 'multimodal/grcn' 'multimodal/lattice' 'multimodal/mmgcn' 'multimodal/vbpr' 'multimodal/lightgcn_m')
model_list=('featureitemknn' 'bivaem' 'multimodal/bm3' 'multimodal/clcrec' 'multimodal/freedom' 'multimodal/grcn' 'multimodal/lattice' 'multimodal/mmgcn' 'multimodal/vbpr' 'multimodal/lightgcn_m')
for model in "${model_list[@]}"
do
    echo "model: $model"
    sbatch --job-name=emo-rec --mail-user=andreas.peintner@uibk.ac.at --time=300:00:00 --mem=128G --gres=gpu:1 ~/jobs/single-node-gpu.job "conda run -n elliot python run_sep.py --data $1 --model $model"
    sleep 1
done
