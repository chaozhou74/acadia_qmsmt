import numpy as np

def t2f(t_unit):
    if t_unit == "ns":
        return "GHz"
    elif t_unit == "us":
        return "MHz"
    elif t_unit == "ms":
        return "kHz"
    elif t_unit == "s":
        return "Hz"
    else:
        raise NameError("unsupported time unit")

def f2t(f_unit):
    if f_unit == "GHz":
        return "ns"
    elif f_unit == "MHz":
        return "us"
    elif f_unit == "kHz":
        return "ms"
    elif f_unit == "Hz":
        return "s"
    else:
        raise NameError("unsupported freq unit")

def freqUnit(f_unit):
    if f_unit == "GHz":
        return 10**9
    elif f_unit == "MHz":
        return 10**6
    elif f_unit == "kHz":
        return 10**3
    elif f_unit == "Hz":
        return 10**0
    else:
        raise NameError("unsupported freq unit")

def timeUnit(t_unit):
    if t_unit == "ps":
        return 10**-12
    elif t_unit == "ns":
        return 10**-9
    elif t_unit == "us":
        return 10**-6
    elif t_unit == "ms":
        return 10**-3
    elif t_unit == "s":
        return 10**0
    else:
        raise NameError("unsupported freq unit")


def rounder(value, digit=5):
    return f"{value:.{digit}e}"


def magPhase2realImag(mag, phase, phase_unit="rad"):
    lin = 10 ** (mag / 20.0)
    if phase_unit == "deg":
        phase = phase / 180 * np.pi
    elif phase_unit != "rad":
        raise NameError("wrong phse_unit")
    real = lin * np.cos(phase)
    imag = lin * np.sin(phase)
    return real, imag


def realImag2magPhase(real, imag):
    mag = 10 * np.log10(real ** 2 + imag ** 2)
    phase = np.unwrap(np.angle(real + 1j * imag))
    return mag, phase