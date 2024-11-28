import numbers
import re
from typing import Union
import numpy as np

"""
type conversion functions
"""

def to_float(input):
    return float(input)

def to_complex(input):
    """
    Convert input of format "a+bj", "(a, b pi)", "(a, b deg)" to complex numbers
    :param input:
    :return:
    """
    # strip the spaces
    input_str = str(input).replace(" ", "")

    # Case 1: "a+bj" or variations like "a", "bj", "a+bj"
    try:
        return complex(input_str)
    except ValueError:
        pass

    # Case 2: "(a, b pi)"
    match_pi = re.match(r"\(\s*([\d\.\-eE]+)\s*,\s*([\d\.\-eE]+)pi\s*\)", input_str)
    if match_pi:
        amp = float(match_pi.group(1))
        angle = float(match_pi.group(2)) * np.pi  # Convert angle in pi to radians
        return np.exp(1j*angle) * amp

    # Case 3: "(a, b deg)"
    match_deg = re.match(r"\(\s*([\d\.\-eE]+)\s*,\s*([\d\.\-eE]+)deg\s*\)", input_str)
    if match_deg:
        amp = float(match_deg.group(1))
        angle = float(match_deg.group(2))/180*np.pi  # Convert angle in degrees to radians
        return np.exp(1j*angle) * amp


def load_kernel(input:Union[str, list, float, complex]):
    """
    Based on the type of input, convert it to a numpy array.
    - If input is a number, load as np.array([input]).
    - If input is a list, load as np.array(input).
    - If input is a string for a path of a .npy file, load the array from the file.

    :param input: Any number, list, or string representing a file path to a .npy file.
    :return: A numpy array representation of the input.
    """
    if isinstance(input, numbers.Number):  # Check if input is a number
        return np.array([input])
    elif isinstance(input, list):  # Check if input is a list
        return np.array(input)
    elif isinstance(input, str):  # Check if input is a string
        if input.endswith('.npy'):  # Ensure it's a path to a .npy file
            try:
                return np.load(input)
            except Exception as e:
                raise ValueError(f"Error loading .npy file: {e}")
        else:
            raise ValueError(f"Unsupported file type: {input}")
    else:
        raise ValueError("Unsupported input type. Input must be a scalar, list, or path to a .npy file.")


"""
handler routines
"""

PARAMETER_HANDLERS = {
    "*.signals.*.scale": to_complex,
    "*.kernel_wf": load_kernel
}




if __name__ == "__main__":
    print(to_complex("(0.5, 60 deg)"))
    print(to_complex(-0.5))


    kn = load_kernel(0.1+0.2j)
    print(kn)
