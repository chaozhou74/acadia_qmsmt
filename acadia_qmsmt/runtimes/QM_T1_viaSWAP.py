from typing import Union

import numpy as np
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig, QubitQmCooler
from acadia.runtime import annotate_method


class QMT1Runtime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for sweeping the rabi time
    """
    qubit_stimulus: IOConfig
    bs_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    delay_times: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    do_not_swap_back: bool = False

    qubit_pulse_name: str = None
    bs_pulse_name: str = None
    
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "matched"

    cool_swap_pulse_name: str = "swap"
    cool_qm_rounds: int = 2


    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        if self.do_not_swap_back:
            logger.warn("NOTE: I'M NOT SWAPPING BACK!!!! YOU SHOULD SEE A FLAT LINE!!!!")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")
        bs_stimulus_io = self.io("bs_stimulus")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)
        cooler = QubitQmCooler(qubit, readout_resonator, bs_stimulus_io)

        if self.cool_qm_rounds>0:
            self.data.add_group(f"prep", uniform=True)

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))
        prep_capture_mem = readout_capture_io.get_waveform_memory(self.capture_memory_name).duplicate()


        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            counter = a.sequencer().DSP()

            # Load the counter with the value we put into the cache
            counter.load(cache[0])

            if self.cool_qm_rounds>0:
                cooler.cool(1, self.qubit_pulse_name,
                    self.readout_pulse_name, self.capture_memory_name, self.capture_window_name,
                    self.cool_swap_pulse_name, self.cool_qm_rounds) # efficient use of register

            with a.channel_synchronizer(block=True):
                qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name)
                a.barrier()
                bs_stimulus_io.schedule_pulse(self.bs_pulse_name)

            with a.channel_synchronizer():
                qubit_stimulus_io.dwell(counter)

            if self.do_not_swap_back:
                    logger.warning("NOTE: I'M NOT SWAPPING BACK!!!! YOU SHOULD SEE A FLAT LINE!!!!")
            else:
                with a.channel_synchronizer():
                    bs_stimulus_io.schedule_pulse(self.bs_pulse_name)
                       
            with a.channel_synchronizer():
                readout_resonator.measure("readout", "readout_accumulated", self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse("readout")
        qubit_stimulus_io.load_pulse(self.qubit_pulse_name)
        bs_stimulus_io.load_pulse(self.bs_pulse_name)
        if self.cool_qm_rounds>0:
            bs_stimulus_io.load_pulse(self.cool_swap_pulse_name)

        # Determine how many cycles each delay interval should be 
        dsp_count_values = self.acadia.seconds_to_cycles(self.delay_times)

        configure_streams = False

        for i in range(self.iterations):
            for  delay in dsp_count_values:
                cache[0] = delay

                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory("readout_accumulated")
                self.data[f"points"].write(wf.array)
                configure_streams = True

                if self.cool_qm_rounds>0:
                    wf = readout_capture_io.get_waveform_memory(prep_capture_mem)
                    self.data[f"prep"].write(wf.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()


    def initialize(self):
        pass

    def update(self):
        # # get current completed data
        # self.process_current_data()

        # save current data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        from acadia_qmsmt.plotting import save_registered_plots
        if self.plot:
            save_registered_plots(self)


    @annotate_method(is_data_processor=True)
    def process_current_data(self):
        # First make sure that we actually have new data to process
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.delay_times)
        if data is None:
            return
        else:
            completed_iterations = len(data)

        self.data_iq = data.astype(float).view(complex).squeeze()
        self.avg_iq = np.mean(self.data_iq, axis=0)
        self.shots = (1-np.sign(self.data_iq.real))/2
        self.data_to_fit = np.mean(self.shots, axis=0)
        self.data_sigma = np.std(self.shots, axis=0)/np.sqrt(completed_iterations)

        if self.cool_qm_rounds > 0:
            # merge all sweep axes
            self.prep_data = reshape_iq_data_by_axes(self.data["prep"].records())

        from acadia_qmsmt.analysis.fitting import Exponential
        self.delay_times_us = self.delay_times * 1e6
        self.fit = Exponential(self.delay_times_us, self.data_to_fit, sigma=self.data_sigma)
        self.fitted_t1_us = self.fit.ufloat_results["tau"]

        return completed_iterations



    @annotate_method(plot_name="T1 thresholded", axs_shape=(1,1))
    def plot1_T1(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        # ax.errorbar(self.delay_times*1e6, np.abs(self.avg),self.std_errs,linestyle='',marker='o')

        self.fit.plot(axs, oversample=5,
                            result_kwargs={"label": f"T1 (us): {self.fitted_t1_us:.4g}"})
        axs.set_title(f"T1: {self.fit.ufloat_results['tau']:.5g}")

        axs.set_xlabel("Time [us]")
        axs.set_ylabel("e pop")
        axs.set_ylim(-0.02, 1.02)

        axs.legend()
        return fig, axs
    

    @annotate_method(plot_name="T1_bin_averaged", axs_shape=(1,1))
    def plot2_traces_raw(self, axs=None, n_avg:int=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)
        fig, axs = plot_binaveraged(self.delay_times_us, self.shots, axs, n_avg=n_avg, vmin=0, v_max=1)
        axs.set_ylabel("Time [us]")
        return fig, axs
    