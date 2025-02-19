from elliot.run import run_experiment
import argparse
from utils import merge_yaml, save_yaml

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--config', type=str, default='cobrar_onion')
args = parser.parse_args()

print(f"Running experiment with config: {args.config}")

base_config = 'config_files/DEFAULT.yml'
extended_config = f"config_files/{args.config}.yml"
merged_config = merge_yaml(base_config, extended_config)

save_yaml(merged_config, "config_files/merged.yml")

run_experiment("config_files/merged.yml")