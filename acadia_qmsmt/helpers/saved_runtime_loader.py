import importlib.util
import inspect
import sys
import types
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
   (default behavior) or the global one in the Python environment.
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


def get_saved_runtime_class(data_directory: str):
    """
    Get the runtime class in `DEFAULT_RUNTIME_MODULE`(runtime.py) in the local data folder.

    Currently the runtime class is identified by assuming there is only one class definition in the local
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


def insert_saved_qmsmt_module(data_directory: str):
    """
    Load the saved `acadia_qmsmt.py` file as the `acadia_qmsmt.qmsmt` submodule
    so the rest of the local `acadia_qmsmt` package remains available, but the `qmsmt` 
    submodule will be replaced by the saved copy.

    :param data_directory: data directory that contains the acadia_qmsmt.py
    """

    # Remove the old qmsmt if it exists
    if f"{QMSMT_MODULE_NAME}" in sys.modules:
        del sys.modules[f"{QMSMT_MODULE_NAME}"]

    data_path = Path(data_directory)
    local_qmsmt_path = data_path / (QMSMT_MODULE_NAME + ".py")
    if not local_qmsmt_path.exists():
        logger.warning("Warning: No local copy of acadia_qmsmt found. Will try using global version.")
        return

    # Load the saved acadia_qmsmt.py as the acadia_qmsmt.qmsmt submodule
    qmsmt_module = types.ModuleType(f"{QMSMT_MODULE_NAME}.qmsmt")
    exec(open(local_qmsmt_path).read(), qmsmt_module.__dict__)
    sys.modules[f"{QMSMT_MODULE_NAME}.qmsmt"] = qmsmt_module

    logger.info(f"Insrted saved {QMSMT_MODULE_NAME} from {data_directory} to global {QMSMT_MODULE_NAME}.qmsmt")


def load_runtime_from_data_dir(data_directory: str, use_saved_qmsmt:bool = True):
    """
    Load and return the saved runtime object from a given saved data directory.

    :param data_directory: Path a local data folder
    :param use_saved_qmsmt: If True, attempt to replace the local `acadia_qmsmt.qmsmt` module by the saved
                            `acadia_qmsmt.py` n the data folder. So direct re-deploy will use the qmsmt classes that
                            were used at the original runtime.
                            If False or not found, fall back to the global version in the Python environment.
    """
    if use_saved_qmsmt:
        insert_saved_qmsmt_module(data_directory)
    else:
        # Reset sys.modules to ensure global version is used, in case we have loaded the saved one
        # in the same console earlier
        if QMSMT_MODULE_NAME in sys.modules:
            del sys.modules[QMSMT_MODULE_NAME]
        # Re-import the global module
        import acadia_qmsmt

    # Get the saved runtime class
    runtime_cls = get_saved_runtime_class(data_directory)

    # Initialize and return the runtime instance
    return runtime_cls.load(data_directory)



if __name__ == "__main__":
    # example usage:
    path = "/home/chao/Data/test_gui/AmpSweep/250414_183827"
    rt = load_runtime_from_data_dir(path)
    print(rt.data)
