#!/bin/bash

# Check if dataset argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    echo "dataset options: emma, onion, session_emma, session_onion"
    exit 1
fi

emma_model_list=(
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''emotion'\''_n_layers=2_normalize=True_aggregation=concat_loads='\''EmotionAttribute'\''.tsv'
  'FeatureItemKNN_nn=70_sim=cosine_msf=0$05_bin=False_modalites='\''emotion'\''_aggregation=concat_loads='\''EmotionAttribute'\''.tsv'
  'LightGCN_seed=123_e=200_bs=128_lr=0$0005_factors=64_l_w=1e-05_n_layers=2_normalize=True.tsv'
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''-'\''emotion'\''_n_layers=1_normalize=True_aggregation=concat_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''-'\''EmotionAttribute'\''.tsv'
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''_n_layers=1_normalize=True_aggregation=mean_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''.tsv'
  'FeatureItemKNN_nn=70_sim=cosine_msf=0$05_bin=False_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''-'\''emotion'\''_aggregation=concat_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''-'\''EmotionAttribute'\''.tsv'
  'FeatureItemKNN_nn=70_sim=cosine_msf=0$05_bin=False_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''_aggregation=ensemble_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''.tsv'
  'ItemKNN_nn=70_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv'
)

onion_model_list=(
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''-'\''emotion'\''_n_layers=2_normalize=True_aggregation=mean_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''-'\''EmotionAttribute'\''.tsv'
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''_n_layers=2_normalize=True_aggregation=mean_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''.tsv'
  'FeatureItemKNN_nn=20_sim=cosine_msf=0$15_bin=False_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''_aggregation=ensemble_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''.tsv'
  'FeatureItemKNN_nn=20_sim=cosine_msf=0$15_bin=False_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''-'\''emotion'\''_aggregation=concat_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''-'\''EmotionAttribute'\''.tsv'
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''emotion'\''_n_layers=3_normalize=True_aggregation=sum_loads='\''EmotionAttribute'\''.tsv'
  'LightGCN_seed=123_e=200_bs=128_lr=0$0005_factors=64_l_w=1e-05_n_layers=3_normalize=True.tsv'
  'FeatureItemKNN_nn=20_sim=cosine_msf=0$05_bin=False_modalites='\''emotion'\''_aggregation=concat_loads='\''EmotionAttribute'\''.tsv'
  'ItemKNN_nn=20_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv'
)

session_emma_model_list=(
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''emotion'\''_n_layers=2_normalize=True_aggregation=concat_loads='\''EmotionAttribute'\''.tsv'
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''_n_layers=2_normalize=True_aggregation=concat_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''.tsv'
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''-'\''emotion'\''_n_layers=2_normalize=True_aggregation=concat_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''-'\''EmotionAttribute'\''.tsv'
  'FeatureItemKNN_nn=100_sim=cosine_msf=0$05_bin=False_modalites='\''emotion'\''_aggregation=ensemble_loads='\''EmotionAttribute'\''.tsv'
  'LightGCN_seed=123_e=200_bs=128_lr=0$0005_factors=64_l_w=1e-05_n_layers=3_normalize=True.tsv'
  'FeatureItemKNN_nn=100_sim=cosine_msf=0$05_bin=False_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''-'\''emotion'\''_aggregation=concat_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''-'\''EmotionAttribute'\''.tsv'
  'ItemKNN_nn=100_sim=cosine_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv'
  'FeatureItemKNN_nn=70_sim=cosine_msf=0$05_bin=False_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''_aggregation=ensemble_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''.tsv'
)

session_onion_model_list=(
  'LightGCN_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_n_layers=2_normalize=True.tsv'
  'LightGCNM_seed=123_e=200_bs=128_lr=0$001_factors=64_l_w=1e-05_modalites='\''audio'\''-'\''textual'\''-'\''visual'\''_n_layers=3_normalize=True_aggregation=mean_loads='\''AudioAttribute'\''-'\''TextualAttribute'\''-'\''VisualAttribute'\''.tsv'
  'ItemKNN_nn=20_sim=dot_imp=standard_bin=False_shrink=0_norm=True_asymalpha=_tvalpha=_tvbeta=_rweights=.tsv'
)

dataset=$1
if [ $dataset == "emma" ]; then
    model_list=("${emma_model_list[@]}")
elif [ $dataset == "onion" ]; then
    model_list=("${onion_model_list[@]}")
elif [ $dataset == "session_emma" ]; then
    model_list=("${session_emma_model_list[@]}")
elif [ $dataset == "session_onion" ]; then
    model_list=("${session_onion_model_list[@]}")
else
    echo "Invalid dataset"
    exit 1
fi


for i in "${!model_list[@]}"; do
  model_list[$i]="./results/${dataset}/recs/${model_list[$i]}"
done

for model in "${model_list[@]}"; do
  echo "model path: $model"
  echo "dataset: $dataset"
  sbatch --job-name=emo-rec --mail-user=andreas.peintner@uibk.ac.at --time=300:00:00 --mem=128G --gres=gpu:1 ~/jobs/single-node-gpu.job "conda run -n elliot python run_eval.py --dataset $dataset --path $model"
  sleep 0.5
done
