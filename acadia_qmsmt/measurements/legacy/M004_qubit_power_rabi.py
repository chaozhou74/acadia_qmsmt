import matplotlib.pyplot as plt
from IPython.core.getipython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")
import numpy as np


if __name__ == "__main__":
    from acadia_qmsmt.measurements import load_config
    from acadia_qmsmt.runtimes.qubit_power_rabi import QubitPwrRabiRuntime


    config_dict = load_config()

    plot = True
    iterations = 200
    qubit_amp_scales = np.linspace(-1.5, 1.5, 61) # not the scale in "signal", is multiply factor on that


    rt = QubitPwrRabiRuntime(**config_dict, qubit_amp_scales=qubit_amp_scales, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "qubit_power_rabi", files=rt.FILES)    
    rt.display()

    # some ad-hoc processing
    # rt._event_loop.join()
    # rt.fig
