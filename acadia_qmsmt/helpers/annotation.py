from typing import Dict

PLOT_NAME_TAG = "plot_name"
BUTTON_NAME_TAG = "button_name"
DISABLE_TAG = "disabled"
AXS_SHAPE_TAG = "axs_shape"
DATA_PROCESS_TAG = "is_data_processor"


# ------------------------- runtime annotation methods --------------------------------------------
def get_registered_methods(runtime_obj, identifier) -> Dict[str, str]:
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


def get_registered_plot_methods(runtime_obj) -> Dict[str, str]:
    """
    Get all registered plot methods from a runtime object,that are annotated
    with `@annotate_method(plot_name=...)`
    return as a dict of {plot_name: plot_method_name}

    """
    plots = get_registered_methods(runtime_obj, PLOT_NAME_TAG)
    return plots


def get_registered_button_methods(runtime_obj) -> Dict[str, str]:
    """
    Get all registered methods for clickable buttons from a runtime object, that are annotated
    with `@annotate_method(button_name=...)`
    return as a dict of {button_name: button_method_name}
    """
    buttons = get_registered_methods(runtime_obj, BUTTON_NAME_TAG)
    return buttons


def get_data_process_method(runtime_obj) -> str:
    """
    Get the registered runtime method that processes the current data.
    The data processing method should get all the necessary data ready for plotting,
    and store in class attributes, and return the current completed iterations

    The method is identified with the tag `DATA_PROCESS_TAG = True`

    :param runtime_obj:
    :return: Name of the data processing method
    """
    process_methods = []
    for attr in dir(runtime_obj):
        method = getattr(runtime_obj, attr)
        is_data_processor = getattr(method, DATA_PROCESS_TAG, False)
        if callable(method) and is_data_processor:
            process_methods.append(attr)
    if len(process_methods) > 1:
        raise AttributeError(
            f"Multiple data processor methods found: {process_methods}. "
            f"Please define a single wrapper method that calls all necessary processors, "
            f"and decorate with `@annotate_method({DATA_PROCESS_TAG}=True)`."
        )
    elif len(process_methods) == 0:
        raise AttributeError(
            f"No data processor method found in {runtime_obj}. "
            f"Make sure one method is decorated with `@annotate_method({DATA_PROCESS_TAG}=True)`."
        )

    return process_methods[0]
