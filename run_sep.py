from pyexpat import model
from elliot import dataset
from elliot.run import run_experiment
import argparse
from utils import save_yaml, load_yaml, merge_dicts

import wandb
import time

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--data', type=str, default='onion_audio_emotion')
parser.add_argument('--model', type=str, default='featureitemknn')
args = parser.parse_args()

print(f"Running experiment with data config: {args.data} and model config: {args.model}")

dataset_defaults_config = load_yaml("config_files/datasets/dataset_defaults.yml")
model_defaults_config = load_yaml("config_files/models/model_defaults.yml")

dataset_config = load_yaml(f"config_files/datasets/{args.data}.yml")
model_config = load_yaml(f"config_files/models/{args.model}.yml")

config = merge_dicts(dataset_defaults_config, dataset_config, model_config)

# add model defaults & modalities and loaders
model_keys = config["experiment"]["models"].keys()
for key in model_keys:
    config["experiment"]["models"][key] = {**model_defaults_config, **config["experiment"]["models"][key]}
    config["experiment"]["models"][key]["modalities"] = dataset_config['modalities']
    config["experiment"]["models"][key]["loaders"] = dataset_config['loaders']

# drop all non experiment keys
config = {key: config[key] for key in config.keys() & {'experiment'}}

save_yaml(config, "config_files/tmp.yml")

wandb.login()
run_experiment("config_files/tmp.yml", dataset_name=args.data, model_name=args.model)