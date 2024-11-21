import numpy as np

"""
type conversion functions
"""
def to_complex(value):
    return complex(str(value).replace(" ", ""))

def to_float(value):
    return float(value)


"""
handler routines
"""
PARAMETER_HANDLERS = {
    "*.signal.scale": to_complex,
}

