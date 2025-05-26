import matplotlib.pyplot as plt
from IPython.core.getipython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")
import numpy as np


if __name__ == "__main__":
    from acadia_qmsmt.measurements import load_config
    from acadia_qmsmt.runtimes.readout_pts_active_cooling_demo import ActiveCoolingRuntime

    config_dict = load_config()

    plot = True
    iterations = 50000

    rt = ActiveCoolingRuntime(**config_dict, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout_pts_active_cooling_demo", files=rt.FILES)
    rt.display()

    # some ad hoc processing
    # rt._event_loop.join()
    # rt.fig
    
    # print(rt.data[f"pts_2"].records().shape)


