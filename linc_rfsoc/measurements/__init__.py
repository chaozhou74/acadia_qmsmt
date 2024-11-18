import yaml
from linc_rfsoc.helpers.yaml_editor import update_yaml, load_yaml

CONFIG_FILE_PATH = "temp_config.yaml"


def load_config():
    config = load_yaml(CONFIG_FILE_PATH)
    return config


def update_config(new_config:dict):
    update_yaml(CONFIG_FILE_PATH, new_config)