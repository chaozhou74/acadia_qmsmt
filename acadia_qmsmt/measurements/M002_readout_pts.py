import matplotlib.pyplot as plt
from IPython.core.getipython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")
import numpy as np


if __name__ == "__main__":
    from acadia_qmsmt.measurements import load_config
    from acadia_qmsmt.runtimes.readout_pts import ReadoutPtsRuntime

    config_dict = load_config()

    plot = True
    iterations = 5000

    rt = ReadoutPtsRuntime(**config_dict, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout_pts", files=rt.FILES)
    rt.display()

    # some ad hoc processing
    # rt._event_loop.join()
    # rt.fig

    # fig, ax = plt.subplots(1, 1)
    # for i, s_ in enumerate(["g", "e"]):
    #     data = rt.data[f"pts_{s_}"].records().squeeze()
    #     ax.plot(data[:, 0], data[:, 1], ".", ms=0.5)
    #     ax.set_aspect(1)
