import matplotlib.pyplot as plt
from IPython.core.getipython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")
import numpy as np


if __name__ == "__main__":
    from acadia_qmsmt.measurements import load_config
    from acadia_qmsmt.runtimes.qubit_spec import QubitSpecRuntime


    config_dict = load_config()

    plot = True
    iterations = 50
    qubit_freqs = np.linspace(-20e6, 20e6, 101) + 8.23e9

    rt = QubitSpecRuntime(**config_dict, qubit_frequencies=qubit_freqs, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "qubit_spec", files=rt.FILES)
    rt.display()
