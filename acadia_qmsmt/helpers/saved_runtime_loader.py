import importlib.util
import inspect
import sys
from pathlib import Path
import logging
logger = logging.getLogger("qmsmt_runtime_loader")

"""
This module provides helper functions for loading the runtime object of a saved experiment directly from a data folder.

Compared to the existing `runtime.load()`, this provides two additional features:

1. The runtime class is inferred based on the files saved in the data folder (see `get_saved_runtime_class`). 
   This means we can simply call `load_runtime_from_data_dir(data_directory)` 
   without having to manually find and import the runtime class first, then do `Runtime.load(data_folder_path)`. 

2. For qmsmt runtimes, this allows choosing whether to use the `acadia_qmsmt.py` module saved in the data folder 
   (default behavior) or the global one in the Python environment. In `load_runtime_from_data_dir`, this is achieved 
   by optionally loading the saved `acadia_qmsmt` module first, and then instantiating the runtime using the class 
   defined in the saved `runtime.py`.
"""

DEFAULT_RUNTIME_MODULE = "runtime.py"  # module that contains the runtime class in the data folder
QMSMT_MODULE_NAME = "acadia_qmsmt"  # the saved acadia_qmsmt module name used for imports

def _get_classes_in_module(module_path: str):
    """
    Given a path to a Python module file, return a dictionary of class names and class objects.

    :param module_path: Path to a .py file
    :return: dict of {class_name: class_obj}
    """
    module_path = Path(module_path)
    if not module_path.exists() or module_path.suffix != '.py':
        raise ValueError(f"Invalid Python module path: {module_path}")

    # Create a unique module name to avoid clashes
    module_name = module_path.stem + "_loaded"

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


def get_saved_runtime_class(data_directory: str):
    """
    Get the runtime class in `DEFAULT_RUNTIME_MODULE`(runtime.py) in the local data folder.

    Currently the runtime class is identified by assuming there is only one class defination in the local
    runtime module, but this can be improved later if we have more complicated cases.

    :param data_directory: data directory
    """
    data_path = Path(data_directory)

    runtime_module = data_path / DEFAULT_RUNTIME_MODULE
    runtime_classes = _get_classes_in_module(runtime_module)

    if len(runtime_classes) == 0:
        raise ImportError(f"Can't find runtime class in {runtime_module}")

    elif len(runtime_classes) > 1:
        raise ImportError(f"Can't determine which runtime class to use in {runtime_module}, "
                          f"found {list(runtime_classes.keys())}")

    return list(runtime_classes.values())[0]


def import_saved_acadia_qmsmt(module_path: str):
    """
    Import the acadia_qmsmt module from a specific path,
    and inject it into sys.modules so future imports use this version.
    Without calling this first, the global acadia_qmsmt module in the python environment will be used.

    :param module_path: Path the acadia_qmsmt.py file in the local data folder
    """
    spec = importlib.util.spec_from_file_location(QMSMT_MODULE_NAME, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[QMSMT_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def load_runtime_from_data_dir(data_directory: str, use_saved_qmsmt:bool = True):
    """
    Load and return the saved runtime object from a given saved data directory.

    :param data_directory: Path a local data folder
    :param use_saved_qmsmt: If True, attempt to use the `acadia_qmsmt` module saved in the data folder.
                            If False or not found, fall back to the global version in the Python environment.
    """
    data_path = Path(data_directory)

    if use_saved_qmsmt:
        # Load the acadia_qmsmt copy from the data folder, if it exists
        local_qmsmt = data_path / (QMSMT_MODULE_NAME + ".py")
        if local_qmsmt.exists():
            logger.info(f"Importing local acadia_qmsmt from: {local_qmsmt}")
            import_saved_acadia_qmsmt(local_qmsmt)
        else:
            logger.warning("Warning: No local copy of acadia_qmsmt found. Using global version.")

    # Get the saved runtime class
    runtime_cls = get_saved_runtime_class(data_directory)

    # Initialize and return the runtime instance
    return runtime_cls.load(data_directory)



if __name__ == "__main__":
    # example usage:
    path = "/home/chao/Data/test_gui/AmpSweep_250413_172312"
    rt = load_runtime_from_data_dir(path, True)
    print(rt)
