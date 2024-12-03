import fnmatch
import operator
from functools import reduce
import ruamel.yaml as yaml
import numpy as np
from attr.setters import convert

from acadia_qmsmt.helpers.yaml_routine import PARAMETER_HANDLERS


def apply_type_handlers(config: dict, handlers: dict) -> dict:
    """
    Recursively applies type handlers to matched paths in the configuration dict.

    :param config: The nested configuration dict.
    :param handlers: A dictionary mapping wildcard path patterns (as strings) to handler functions.

    :return: The modified configuration dictionary with transformations applied.
    """

    def _recursive_apply(data, path):
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = path + (key,)
                data[key] = _recursive_apply(value, new_path)
        else:
            # Match the path using the wildcard handlers
            for pattern, handler in handlers.items():
                if fnmatch.fnmatch(".".join(path), pattern):
                    return handler(data)
        return data

    return _recursive_apply(config, ())


def load_yaml(yaml_path: str):
    """
    Load the yaml file and process it with handlers defined in yaml_routine.PARAMETER_HANDLERS

    :param yaml_path: path to the yaml config file

    :return:
    """
    yml = yaml.YAML(typ='safe', pure=True)
    with open(yaml_path, 'r') as file:
        config = yml.load(file)
    return apply_type_handlers(config, PARAMETER_HANDLERS)


def to_yaml_friendly(value):
    """convert possible numpy type to native python types"""
    if type(value) == str:
        return value
    if type(value) == dict:
        converted = {}
        for k_, v_ in value.items():
            vv_ = to_yaml_friendly(v_)
            converted[k_] = vv_
        return converted

    try:
        if len(value) > 0:
            # convert np.array to list
            try: # not just np.ndarray, any class that has this method
                converted = value.tolist()
            except AttributeError:
                converted = list(value)
            # convert each element
            for i, d in enumerate(converted):
                converted[i] = to_yaml_friendly(d)
            return converted

    except TypeError:
        if isinstance(value, np.generic): # Covert all NumPy scalar types
            return value.item()
        else:
            return value


def update_yaml(yaml_path: str, new_param_dict: dict, keep_format=True, verbose=False):
    """
    update a yaml config file with updated parameters, and keep the original format. 

    :param yaml_path: path to the yaml file to be updated
    :param new_param_dict: dictionary that contains the updated parameters. For nested parameters, the key needs be the
        key of each layer jointed with '.'
   :param keep_format: When True, format the new parameters with the dtype of the original ones
   :param verbose: When True, print details of the update

    :Example:
        >>> old_config = {"config":{"relax_delay" : 100}} #to update relax_delay to 20, we do:
        >>> update_yaml(yaml_path, {"config.relax_delay": 20})

    :return:
    """

    def get_by_path(root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    def set_by_path(root, items, value, keep_format):
        """Set a value in a nested object in root by item sequence."""
        if keep_format:
            try:
                data_type = type(get_by_path(root, items)).__base__
                new_data = data_type(value)
            except Exception as e:
                new_data = value
        else:
            new_data = value
        get_by_path(root, items[:-1])[items[-1]] = new_data

    config, ind, bsi = yaml.util.load_yaml_guess_indent(open(yaml_path))

    for s, val in new_param_dict.items():
        set_by_path(config, s.split("."), to_yaml_friendly(val), keep_format)

    new_yaml = yaml.YAML()
    new_yaml.default_flow_style = None
    new_yaml.indent(mapping=ind, sequence=ind, offset=bsi)

    with open(yaml_path, 'w') as fp:
        new_yaml.dump(config, fp)

    if verbose:
        print(f"YAML file {yaml_path} updated with {new_param_dict}")


if __name__ == "__main__":
    temp_yaml = "../measurements/temp_config.yaml"
    new_param_dict = {"q_stimulus.nco_config.nco_frequency": f"{3e9:6e}"}
    update_yaml(temp_yaml, new_param_dict, keep_format=True)