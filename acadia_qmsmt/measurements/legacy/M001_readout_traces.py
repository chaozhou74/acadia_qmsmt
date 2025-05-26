import matplotlib.pyplot as plt
from IPython.core.getipython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")
import numpy as np


if __name__ == "__main__":
    from acadia_qmsmt.measurements import load_config
    from acadia_qmsmt.runtimes.readout_traces import ReadoutTracesRuntime

    config_dict = load_config()

    plot = True
    iterations = 2000


    rt = ReadoutTracesRuntime(**config_dict, plot=plot, iterations=iterations, generate_kernel=True)
    rt.deploy("10.66.3.198", "readout_traces", files=rt.FILES, event_loop_period=0.5)
    rt.display()


    # # some ad hoc processing
    # rt._event_loop.join() # wait for runtime to finish, this will stop live plotting from working

    # # generate kernel using the acquired data
    # t_data, g_traces, e_traces = rt.parse_data(rt.data)
    # from acadia_qmsmt.analysis.generate_readout_kernel import KernelFromPreparedTraces
    # kernel_gen = KernelFromPreparedTraces(g_traces, e_traces, norm_factor=1, plot=True,
    #                                        decimation_used=rt.ro_capture["waveforms"]["ro_demod"]["decimation"])
    # kernel_gen.update_kernel(rt.yaml_path, "ro_capture.kernel_wf", r"../_develop", "test_kernel")


