from acadia_qmsmt.helpers.yaml_editor import update_yaml, load_yaml

CONFIG_FILE_PATH = "/home/chao/github/acadia_qmsmt/acadia_qmsmt/measurements/temp_config.yaml"


def load_config():
    """
    Shortcut function for loading the YAML configuration file defined in `CONFIG_FILE_PATH` and applying the handler
    functions to turn it into a dict of python objects.

    This provides a single point of access to the YAML config file used across all experiments in the `measurements`
    folder.

    :return: A dictionary containing the loaded configuration, with the YAML file path included under the key
     'yaml_path'.
    """

    config = load_yaml(CONFIG_FILE_PATH)
    # attach the yaml path to the config dict for easier access when writing
    config["yaml_path"] = CONFIG_FILE_PATH
    return config