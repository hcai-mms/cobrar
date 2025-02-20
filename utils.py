import yaml
import os

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def merge_dicts(*dicts):
    if len(dicts) == 0:
        return {}
    if len(dicts) == 1:
        return dicts[0]

    result = dicts[0]
    for d in dicts[1:]:
        result = deep_merge(result, d)
    return result