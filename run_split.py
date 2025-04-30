import os
import shutil

from elliot.run import run_experiment

import argparse

parser = argparse.ArgumentParser(description="Run training and evaluation.")
parser.add_argument('--data', type=str, default='ml1m')
args = parser.parse_args()

ata_base_path = f'../dataset/movielens1m_elliot/movielens_1m/'

if not (os.path.exists(f'{data_base_path}/{args.data}/train.tsv') and os.path.exists(f'{data_base_path}/{args.data}/val.tsv') and os.path.exists(f'{data_base_path}/{args.data}/test.tsv')):
    run_experiment(f"config_files/split_{args.data}.yml")
    shutil.move(f'{data_base_path}/{args.data}_splits/0/test.tsv', f'{data_base_path}/{args.data}/test.tsv')
    shutil.move(f'{data_base_path}/{args.data}_splits/0/0/train.tsv', f'{data_base_path}/{args.data}/train.tsv')
    shutil.move(f'{data_base_path}/{args.data}_splits/0/0/val.tsv', f'{data_base_path}/{args.data}/val.tsv')
    shutil.rmtree(f'{data_base_path}/{args.data}_splits/')
