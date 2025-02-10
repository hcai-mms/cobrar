#!/bin/bash
#dataset_list=('emma' 'emma_audio' 'emma_textual' 'emma_visual' 'onion' 'onion_audio' 'onion_textual' 'onion_visual')
dataset_list=('onion_textual_emotion')
#model_list=('simple' 'bprmf' 'neumf' 'multivae' 'lightgcn' 'bivae')
#model_list=('simple' 'bprmf' 'neumf' 'multivae' 'lightgcn' 'bivae' 'bivaem' 'featureitemknn' 'multimodal/bm3' 'multimodal/clcrec' 'multimodal/freedom' 'multimodal/grcn' 'multimodal/lattice' 'multimodal/mmgcn' 'multimodal/vbpr' 'multimodal/lightgcn_m')
model_list=('featureitemknn' 'bivaem' 'multimodal/bm3' 'multimodal/clcrec' 'multimodal/freedom' 'multimodal/grcn' 'multimodal/lattice' 'multimodal/mmgcn' 'multimodal/vbpr' 'multimodal/lightgcn_m')
#model_list=('multimodal/lightgcn_m')
for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
            echo "model: $model"
            echo "dataset: $dataset"
            sbatch --job-name=emo-rec --mail-user=andreas.peintner@uibk.ac.at --time=300:00:00 --mem=200G --gres=gpu:1 ~/jobs/single-node-gpu.job "conda run -n elliot python run_sep.py --data $dataset --model $model"
            sleep 1
    done
done
