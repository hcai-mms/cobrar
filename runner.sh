#!/bin/bash

# Check if dataset argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "dataset options: emma, onion"
    exit 1
fi

model_list=('featureitemknn' 'lightgcnm')

emma_dataset_list=(
    'emma_emotion' 'emma_audio' 'emma_textual' 'emma_visual' 'emma_textual_emotion'
    'emma_audio_emotion' 'emma_visual_emotion'
    'emma_audio_textual' 'emma_audio_visual' 'emma_textual_visual'
    'emma_audio_textual_emotion' 'emma_audio_visual_emotion' 'emma_textual_visual_emotion'
    'emma_audio_textual_visual' 'emma_audio_textual_visual_emotion')

onion_dataset_list=(
    'onion_emotion' 'onion_audio' 'onion_textual' 'onion_visual' 'onion_textual_emotion'
    'onion_audio_emotion' 'onion_visual_emotion'
    'onion_audio_textual' 'onion_audio_visual' 'onion_textual_visual'
    'onion_audio_textual_emotion' 'onion_audio_visual_emotion' 'onion_textual_visual_emotion'
    'onion_audio_textual_visual' 'onion_audio_textual_visual_emotion'
)

dataset=$1
if [ $dataset == "emma" ]; then
    dataset_list=("${emma_dataset_list[@]}")
elif [ $dataset == "onion" ]; then
    dataset_list=("${onion_dataset_list[@]}")
else
    echo "Invalid dataset"
    exit 1
fi

#dataset_list=('emma_textual_emotion')

for dataset in "${dataset_list[@]}"; do
    for model in "${model_list[@]}"; do
            echo "model: $model"
            echo "dataset: $dataset"
            sbatch --job-name=emo-rec --mail-user=andreas.peintner@uibk.ac.at --time=300:00:00 --mem=128G --gres=gpu:1 ~/jobs/single-node-gpu.job "conda run -n elliot python run_sep.py --data $dataset --model $model"
            sleep 0.5
    done
done
