import re
import numpy as np

"""
type conversion functions
"""
def to_complex(input):
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



def to_float(input):
    return float(input)


"""
handler routines
"""

PARAMETER_HANDLERS = {
    "*.signals.*.scale": to_complex
}








if __name__ == "__main__":
    print(to_complex("(0.5, 60 deg)"))
    print(to_complex(-0.5))
