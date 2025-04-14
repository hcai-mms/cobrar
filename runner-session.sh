#!/bin/bash

# Check if dataset argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "dataset options: session_emma, session_onion"
    exit 1
fi

emma_dataset_list=('session_emma_emotion' 'session_emma_audio_textual_visual' 'session_emma_audio_textual_visual_emotion')

onion_dataset_list=('session_onion_emotion' 'session_onion_audio_textual_visual' 'session_onion_audio_textual_visual_emotion')

model_list=('featureitemknn' 'lightgcnm')
#model_list=('lightgcnm')

dataset=$1
if [ $dataset == "session_emma" ]; then
    dataset_list=("${emma_dataset_list[@]}")
elif [ $dataset == "session_onion" ]; then
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
            sbatch --job-name=emo-rec --mail-user=andreas.peintner@uibk.ac.at --time=300:00:00 --mem=128G --gres=gpu:1 ~/jobs/single-node-gpu.job "conda run -n elliot python run_sep_session.py --data $dataset --model $model"
            sleep 0.5
    done
done
