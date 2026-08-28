"""Shared script utilities: config loading."""
import os

import yaml


def load_config(name_or_path):
    """name: 'model_nano' | 'voc' | 'coco' or a path to yaml."""
    if os.path.isfile(name_or_path):
        path = name_or_path
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo, "configs", f"{name_or_path}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config(*cfgs):
    """Shallow-merge dicts, later wins."""
    out = {}
    for c in cfgs:
        out.update({k: v for k, v in c.items()})
    return out