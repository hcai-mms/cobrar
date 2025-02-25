from pyexpat import model
from elliot.run import run_wandb_experiment
import argparse
from utils import merge_yaml, save_yaml

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--data', type=str, default='emma')
parser.add_argument('--model', type=str, default='featureitemknn')
args = parser.parse_args()

print(f"Running experiment with data config: {args.data} and model config: {args.model}")

dataset_config = f"config_files/datasets/{args.data}.yml"
model_config = f"config_files/models/{args.model}.yml"

merged_config = merge_yaml(dataset_config, model_config)

# add model defaults
model_defaults = merged_config.get("model_defaults", {})
model_keys = merged_config["experiment"]["models"].keys()
for key in model_keys:
    merged_config["experiment"]["models"][key] = {**model_defaults, **merged_config["experiment"]["models"][key]}

save_yaml(merged_config, "config_files/tmp.yml")

run_wandb_experiment(
    config_path="config_files/tmp.yml",
    actual_config=merged_config,
    dataset_name=args.data,
    model_name=args.model
)