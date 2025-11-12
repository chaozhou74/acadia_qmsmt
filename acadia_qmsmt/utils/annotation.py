from typing import Dict, Callable, Union
from acadia import Runtime


# === Method annotation tags for runtime introspection ===
PLOT_NAME_TAG = "plot_name"  # Tag for identifying a plot method and naming the plot (used in GUI dropdowns, etc.)
AXS_SHAPE_TAG = "axs_shape"  # Tag specifying the shape of preallocated axes for GUI plot layout
BUTTON_NAME_TAG = "button_name"  # Tag for identifying an update button method and assigning its display name
DISABLE_TAG = "disable"  # If set to True, prevents the method from being registered or shown in the GUI
DATA_PROCESS_TAG = "is_data_processor"  # Tag for marking the data processing method that populates attributes for plotting
CUSTOMIZATION_METHOD_TAG = "is_customizer"  # Tag for methods that dynamically create/enable/disable annotated methods (e.g. plots or buttons)


# ------------------------- runtime annotation methods --------------------------------------------
def get_registered_methods(runtime_obj:Runtime, identifier) -> Dict[str, str]:
    """
    Get all registered methods that has the identifier attribute from a runtime object,
    the method in runtime class should be annotated with `@annotate_method(identifier=tagged_name)`
    return as a dict of {tagged_name: method_name}

    Note: this returns the method name in a runtime class, rather than the bounded method.
    """
    methods = {}
    for attr in dir(runtime_obj):
        method = getattr(runtime_obj, attr)
        tagged_name = getattr(method, identifier, None)
        disabled = getattr(method, DISABLE_TAG, False)
        if callable(method) and (tagged_name is not None) and (not disabled):
            methods[tagged_name] = attr
    methods = dict(sorted(methods.items()))  # sort by tagged name alphabetically
    return methods


def get_registered_plot_methods(runtime_obj:Runtime) -> Dict[str, str]:
    """
    Get all registered plot methods from a runtime object,that are annotated
    with `@annotate_method(plot_name=...)`
    return as a dict of {plot_name: plot_method_name}

    """
    plots = get_registered_methods(runtime_obj, PLOT_NAME_TAG)
    return plots


def get_registered_button_methods(runtime_obj:Runtime) -> Dict[str, str]:
    """
    Get all registered methods for clickable buttons from a runtime object, that are annotated
    with `@annotate_method(button_name=...)`
    return as a dict of {button_name: button_method_name}
    """
    buttons = get_registered_methods(runtime_obj, BUTTON_NAME_TAG)
    return buttons



def get_singular_registered_methods(runtime_obj:Runtime, identifier) -> Union[None, str]:
    """
    Get all registered method that has the annotation `identifier` = True
    raise error of find multiple instances,
    return None if not found
    """
    methods = []
    for attr in dir(runtime_obj):
        method = getattr(runtime_obj, attr)
        identifier_true = getattr(method, identifier, False)
        if callable(method) and identifier_true:
            methods.append(attr)
    if len(methods) > 1:
        raise AttributeError(
            f"Found multiple methods with annotation `{identifier}=True` in {runtime_obj}"
            f"Please define a single wrapper method that calls all necessary functions"
        )
    elif len(methods) == 0:
        return None
    else:
        return methods[0]


def get_data_process_method(runtime_obj:Runtime) -> str:
    """
    Get the registered runtime method that processes the current data.
    The data processing method should get all the necessary data ready for plotting,
    and store in class attributes, and return the current completed iterations

    The method is identified with the tag `DATA_PROCESS_TAG = True`

    :param runtime_obj:
    :return: Name of the data processing method
    """
    process_method = get_singular_registered_methods(runtime_obj, DATA_PROCESS_TAG)
    if process_method is None:
        raise AttributeError(
            f"No data processor method found in {runtime_obj}. "
            f"Make sure one method is decorated with `@annotate_method({DATA_PROCESS_TAG}=True)`."
        )

    return process_method


def get_registered_customizer(runtime_obj:Runtime):
    """
    Get the registered runtime method that performs dynamic generation/disabling/enabling of annotated methods.
    """
    return get_singular_registered_methods(runtime_obj, CUSTOMIZATION_METHOD_TAG)



def set_method_annotation(bound_ethod:Callable, **annotations):
    """
    add annotations to a bound method by adding to its original function
    """
    func = getattr(bound_ethod, "__func__")
    for k, v in annotations.items():
        setattr(func, k, v)


