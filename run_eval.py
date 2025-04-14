from elliot.run import run_experiment
import argparse
from utils import merge_dicts, save_yaml, load_yaml

parser = argparse.ArgumentParser(description="Run evaluation.")
parser.add_argument('--config', type=str, default='eval')
parser.add_argument('--dataset', type=str, default="onion")
parser.add_argument('--path', type=str, default="./results/onion/recs/LightGCNM.tsv")
args = parser.parse_args()

if "session" in args.dataset:
    dataset_defaults_config = load_yaml("config_files/datasets/session/dataset_session_defaults.yml")
else:
    dataset_defaults_config = load_yaml("config_files/datasets/dataset_defaults.yml")

eval_config = load_yaml(f"config_files/{args.config}.yml")

config = merge_dicts(dataset_defaults_config, eval_config)

config['experiment']['models']['ProxyRecommender']['path'] = args.path
config['experiment']['dataset'] = args.dataset

save_yaml(config, "config_files/tmp.yml")
run_experiment(f"config_files/tmp.yml")