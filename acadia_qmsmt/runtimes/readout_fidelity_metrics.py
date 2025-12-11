import numpy as np

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method


class ReadoutFidelityRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for readout fidelity metrics
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    iterations: int
    run_delay: int

    qubit_pulse_name: str = "R_x_180"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = None

    num_prep_rounds:int = 0
    post_prep_delay: float = 20e-9
    prep_capture_window_name: str = "matched_biased_g"

    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        self.data.add_group(f"prep_g", uniform=True)
        self.data.add_group(f"prep_e", uniform=True)

        # Create an array in the cache that we can use to pass what state to prep
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))


        # create a capture memory for storing the msmt results for the prep rounds
        if self.num_prep_rounds > 0:
            prep_msmt_mem = readout_capture_io.get_waveform_memory(self.capture_memory_name).duplicate()
            self.data.add_group(f"pre_prep", uniform=True)

        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            state_prep_selecter = a.sequencer().Register()

            
            prepare_register = a.sequencer().Register()

            # Load the counter with the value we put into the cache
            state_prep_selecter.load(cache[0])

            with a.sequencer().test(state_prep_selecter == 0):
                for i in range(self.num_prep_rounds):
                    qubit.prepare(1, readout_resonator, self.qubit_pulse_name,
                          self.readout_pulse_name,
                          prep_msmt_mem,
                          self.prep_capture_window_name, measurement_post_delay=self.post_prep_delay,state_register=prepare_register) # prep in g

            with a.sequencer().test(state_prep_selecter == 1):
                for i in range(self.num_prep_rounds):
                    qubit.prepare(1, readout_resonator, self.qubit_pulse_name,
                          self.readout_pulse_name,
                          prep_msmt_mem,
                          self.prep_capture_window_name, measurement_post_delay=self.post_prep_delay,state_register=prepare_register) # prep in g

                with a.channel_synchronizer():
                    qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name)

            with a.channel_synchronizer():
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_pulse_name)


        for i in range(self.iterations):
            for  prep in [0,1]:
                cache[0] = prep
                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
                if prep == 0:
                    self.data[f"prep_g"].write(wf.array)
                elif prep == 1:
                    self.data[f"prep_e"].write(wf.array)

                if self.num_prep_rounds > 0:
                    self.data[f"pre_prep"].write(prep_msmt_mem.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()


    def initialize(self):
        pass

    def update(self):
        # save current data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        from acadia_qmsmt.plotting import save_registered_plots
        save_registered_plots(self)



    @annotate_method(is_data_processor=True)
    def process_current_data(self,my_thresh=0.):
        # First make sure that we actually have new data to process
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data_g = reshape_iq_data_by_axes(self.data["prep_g"].records())
        data_e = reshape_iq_data_by_axes(self.data["prep_e"].records())
        if data_e is None:
            return
        else:
            completed_iterations = len(data_e)

        self.data_g, self.data_e = data_g, data_e
        if self.num_prep_rounds > 0:
            self.data_pre_prep = reshape_iq_data_by_axes(self.data["pre_prep"].records())
        else:
            self.data_pre_prep = np.zeros_like(data_g)

        # Threshold the data according to the I quadrature
        self.real_g = data_g[..., 0]
        self.real_e = data_e[..., 0]

        self.shots_prep_g_thresh = (1 - np.sign(self.real_g, dtype=np.int32))/2
        self.shots_prep_e_thresh = (1 - np.sign(self.real_e, dtype=np.int32))/2

        self.my_thresh = my_thresh

        prep_g_get_g_counts = np.sum(self.real_g > my_thresh)
        prep_g_get_e_counts = np.sum(self.real_g < my_thresh)
        prep_e_get_g_counts = np.sum(self.real_e > my_thresh)
        prep_e_get_e_counts = np.sum(self.real_e < my_thresh)
        num_counts = 1.*len(self.real_g)

        self.P_g_given_g = prep_g_get_g_counts/num_counts
        self.P_e_given_g = prep_g_get_e_counts/num_counts
        self.P_g_given_e = prep_e_get_g_counts/num_counts
        self.P_e_given_e = prep_e_get_e_counts/num_counts
        self.num_counts = num_counts

        self.Fidelity = 1./2.*(prep_g_get_g_counts/num_counts + prep_e_get_e_counts/num_counts)

        self.median_g = np.median(self.real_g)
        self.median_e = np.median(self.real_e)
        return completed_iterations

    @annotate_method(plot_name="Readout Histogram 2D", axs_shape=(1, 2))
    def plot_histograms_2d(self, axs=None, bins=50, log_scale:bool=False):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_multiple_hist2d
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,2))
        plot_multiple_hist2d(self.data_g, self.data_e, plot_ax=axs, bins=bins, log_scale=log_scale)
        axs[0].set_title("prep g")
        axs[1].set_title("prep e")
        return fig, axs

    @annotate_method(plot_name="Readout Histogram 1D", axs_shape=(1,1))
    def plot_histograms_1d(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, ax = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        ax.hist(self.real_g,bins=100,alpha=0.6,log=True)
        ax.hist(self.real_e,bins=100,alpha=0.6,log=True)

        ax.axvline(self.my_thresh,linestyle='dashed',color='black')
        ax.axvline(self.median_g,linestyle='dotted',color='black')
        ax.axvline(self.median_e,linestyle='dotted',color='black')
        
        ax.set_xlabel("I (uncalibrated)")
        ax.set_ylabel("Counts")

        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"10^{int(np.log10(y))}" if y > 0 else ""))
        ax.set_title(f"F: {np.round(self.Fidelity,5)}")

        ax.grid(True)

        fig.tight_layout()
        return fig, ax

    @annotate_method(plot_name="Readout_Params_Table", axs_shape=(1,1))
    def readout_params_table(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, ax = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        labels = ['Fidelity', 'P(g given g)', 'P(e given g)', 'P(g given e)', 'P(e given e)','median g','median e','separation','# of shots','threshold']
        values = [np.round(self.Fidelity,5), np.round(self.P_g_given_g,5) , np.round(self.P_e_given_g,5) , np.round(self.P_g_given_e,5), np.round(self.P_e_given_e,5), self.median_g,self.median_e, self.median_g - self.median_e,self.num_counts,self.my_thresh]

        # Table data: each row has [label, value]
        table_data = [[label, value] for label, value in zip(labels, values)]

        ax.axis('tight')
        ax.axis('off')

        # Add a top-left label by using colLabels
        ax.table(cellText=table_data,
                colLabels=["Label", "Value"],  # top-left label is "Label"
                cellLoc='center',
                loc='center')

        fig.tight_layout()
        return fig, ax
    
    @annotate_method(plot_name="Readout_pre_prep_result", axs_shape=(1,1))
    def plot_pre_prep(self, axs=None, bins:int=51, log_scale:bool=True):
        from acadia_qmsmt.plotting import  plot_multiple_hist2d
        fig, axs = plot_multiple_hist2d(self.data_pre_prep, plot_ax=axs, bins=bins, log_scale=log_scale)
        axs.set_title("pre-prep msmt results")
        return fig, axs