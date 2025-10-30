import importlib.util
import inspect
import sys
from pathlib import Path
import logging
import os
from contextlib import contextmanager

logger = logging.getLogger("qmsmt_runtime_loader")

"""
This module provides helper functions for loading the runtime object of a saved experiment directly from a data folder.

Compared to the existing `runtime.load()`, this provides two additional features:

1. The runtime class is inferred based on the files saved in the data folder (see `get_saved_runtime_class`). 
   This means we can simply call `load_runtime_from_data_dir(data_directory)` 
   without having to manually find and import the runtime class first, then do `Runtime.load(data_folder_path)`. 

2. For qmsmt runtimes, this allows choosing whether to use the `acadia_qmsmt.py` module saved in the data folder 
   (default behavior, realized by temporally changing cwd to data folder) or the global one in the Python environment.
"""

DEFAULT_RUNTIME_MODULE = "runtime.py"  # module that contains the runtime class in the data folder

@contextmanager
def change_dir(path):
    """
    Context manager for changing the current working directory to the given path.
    Useful for temporarily changing the current working directory to the data folder,
    so saved acadia_qmsmt can be used.
    """
    prev_dir = os.getcwd()
    try:
        os.chdir(path)
        logger.debug(f"Switching curren working directory to {path}")
        yield
    finally:
        os.chdir(prev_dir)
        logger.debug(f"Switching current working directory back to {prev_dir}")


def _get_classes_in_module(module_path: str):
    """
    Given a path to a Python module file, return a dictionary of class names and class objects.

    :param module_path: Path to a .py file
    :return: dict of {class_name: class_obj}
    """
    module_path = Path(module_path)
    if not module_path.exists() or module_path.suffix != '.py':
        raise ValueError(f"Invalid Python module path: {module_path}")

    # Use a consistent module name for loading the runtime to avoid buildup
    module_name = "loaded_runtime_module"
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Load the module from the given path
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Get all classes defined in this module
    classes = {
        name: cls for name, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    }

    return classes


def get_saved_runtime_class(data_directory: str, use_saved_qmsmt: bool = True):
    """
    Get the runtime class in `DEFAULT_RUNTIME_MODULE`(runtime.py) in the local data folder.

    Currently, the runtime class is identified by assuming there is only one class definition in the local
    runtime module, but this can be improved later if we have more complicated cases.

    :param data_directory: data directory
    """
    data_path = Path(data_directory)

    runtime_module = data_path / DEFAULT_RUNTIME_MODULE

    if use_saved_qmsmt:
        with change_dir(data_path):
            runtime_classes = _get_classes_in_module(runtime_module)
    else:
        runtime_classes = _get_classes_in_module(runtime_module)

    if len(runtime_classes) == 0:
        raise ImportError(f"Can't find runtime class in {runtime_module}")

    elif len(runtime_classes) > 1:
        raise ImportError(f"Can't determine which runtime class to use in {runtime_module}, "
                          f"found {list(runtime_classes.keys())}")

    return list(runtime_classes.values())[0]


def load_runtime_from_data_dir(data_directory: str, use_saved_qmsmt: bool = True):
    """
    Load and return the saved runtime object from a given saved data directory.

    :param data_directory: Path to a local data folder
    :param use_saved_qmsmt: If True, temporarily switch current working directory to `data_directory` before loading the
        runtime class, so the `acadia_qmsmt` saved in the data folder will be used to create the runtime class.
    """
    # Get the saved runtime class
    runtime_cls = get_saved_runtime_class(data_directory, use_saved_qmsmt)

    # Initialize and return the runtime instance
    return runtime_cls.load(data_directory)


if __name__ == "__main__":
    # example usage:
    path = "/home/chao/Data/test_gui/AmpSweep/250414_183827"
    rt = load_runtime_from_data_dir(path)
    print(rt.data)
