import fnmatch
import operator
from functools import reduce
import ruamel.yaml as yaml

from linc_rfsoc.helpers.yaml_rountine import PARAMETER_HANDLERS


def apply_handlers(config: dict, handlers: dict) -> dict:
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
    return apply_handlers(config, PARAMETER_HANDLERS)

def to_yaml_friendly(v):
    """convert possible numpy type to native python types"""
    if type(v) == str:
        vv = v
        return vv
    if type(v) == dict:
        vv = {}
        for k_, v_ in v.items():
            vv_ = to_yaml_friendly(v_)
            vv[k_] = vv_
        return vv
    try:
        if len(v) > 0:
            # convert np.array to list
            try:
                vv = v.tolist()
            except AttributeError:
                vv = v
            # convert each element
            for i, d in enumerate(vv):
                vv[i] = to_yaml_friendly(d)
            return vv
    except TypeError:
        vv = float(v)
        return vv


def update_yaml(yaml_path:str, new_param_dict: dict):
    """
    update a yaml config file with updated parameters, and keep the original format. 

    :param yaml_path: path to the yaml file to be updated
    :param new_param_dict: dictionary that contains the updated parameters. For nested parameters, the key needs be the
        key of each layer jointed with '.'

    :Example:
        >>> old_config = {"config":{"relax_delay" : 100}} #to update relax_delay to 20, we do:
        >>> update_yaml(yaml_path, {"config.relax_delay": 20})

    :return:
    """

    def get_by_path(root, items):
        """Access a nested object in root by item sequence."""
        return reduce(operator.getitem, items, root)

    def set_by_path(root, items, value):
        """Set a value in a nested object in root by item sequence."""
        data_type = type(get_by_path(root, items))
        get_by_path(root, items[:-1])[items[-1]] = data_type(value)

    config, ind, bsi = yaml.util.load_yaml_guess_indent(open(yaml_path))
    for s, val in new_param_dict.items():
        set_by_path(config, s.split("."), to_yaml_friendly(val))

    new_yaml = yaml.YAML()
    new_yaml.default_flow_style = None
    new_yaml.indent(mapping=ind, sequence=ind, offset=bsi)

    with open(yaml_path, 'w') as fp:
        new_yaml.dump(config, fp)
