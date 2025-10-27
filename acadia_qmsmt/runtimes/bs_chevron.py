from typing import Union

import numpy as np

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig, QubitQmCooler
from acadia.runtime import annotate_method


class BSChevronRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for sweeping the rabi time
    """
    qubit_stimulus: IOConfig
    bs_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    flat_length_list: Union[list, np.ndarray]
    bs_frequencies: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    bs_pulse_name: str = None
    bs_amp: float = None

    qubit_pulse_name: str = "R_x_180"

    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "boxcar"

    cool_swap_pulse_name: str = "swap"
    cool_qm_rounds: int = 2

    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

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
            bs_length_reg = a.sequencer().Register()
            bs_length_reg.load(cache[0])

            if self.cool_qm_rounds>0:
                cooler.cool(1, self.qubit_pulse_name,
                    self.readout_pulse_name, self.capture_memory_name, self.capture_window_name,
                    self.cool_swap_pulse_name, self.cool_qm_rounds) # efficient use of register

            with a.channel_synchronizer():
                qubit.schedule_pulse(self.qubit_pulse_name)
                qubit_stimulus_io.dwell(10e-9)
                a.barrier()
                bs_stimulus_io.schedule_pulse(self.bs_pulse_name, stretch_length=bs_length_reg)

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

        my_bs_scale = self.bs_amp if self.bs_amp is not None else bs_stimulus_io.get_config("pulses", self.bs_pulse_name, "scale")                                        
        bs_stimulus_io.load_pulse(self.bs_pulse_name, scale=my_bs_scale)

        # Determine how many cycles each flat_length_list should be
        stretch_cycles = self.acadia.seconds_to_cycles(self.flat_length_list)

        my_cal_nco_freq = bs_stimulus_io.get_config("channel_config","nco_frequency")
        configure_streams = True

        for i in range(self.iterations):
            for freq in self.bs_frequencies:
                bs_stimulus_io.set_frequency(freq)
                if self.cool_qm_rounds>0:
                    bs_stimulus_io.load_pulse(self.cool_swap_pulse_name, detune=my_cal_nco_freq - freq)

                for wf_idx, cyc in enumerate(stretch_cycles):
                    cache[0] = cyc

                    # capture data and put in the corresponding group
                    self.acadia.run(minimum_delay=self.run_delay, configure_streams=configure_streams)
                    wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
                    self.data[f"points"].write(wf.array)
                    configure_streams=False
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
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.bs_frequencies, self.flat_length_list)
        if data is None:
            return
        else:
            completed_iterations = len(data)

        if self.cool_qm_rounds > 0:
            # merge all sweep axes
            self.prep_data = reshape_iq_data_by_axes(self.data["prep"].records())

        # Threshold the data according to the I quadrature
        shots = (1 - np.sign(data[..., 0], dtype=np.int32))/2
        self.avg = np.mean(shots, axis=0)

        # do fft on line cuts
        from acadia_qmsmt.analysis import fft
        self.fft_freqs, self.fft_data= fft(self.flat_length_list, self.avg, axis=1, remove_zero_freq=True)
        
        # for plot title
        self.bs_scale = self.bs_amp if self.bs_amp is not None else self._ios["bs_stimulus"].get_config("pulses", self.bs_pulse_name, "scale")    
        self.bs_vop =self._ios["bs_stimulus"].get_config("channel_config", "vop")    

        return completed_iterations


    @annotate_method(plot_name="chevron", axs_shape=(1,1))
    def plot_chevron(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)
        pcm = axs.pcolormesh(self.bs_frequencies/1e9, self.flat_length_list*1e6, self.avg.T, vmin=0, vmax=1, cmap="bwr")
        fig.colorbar(pcm, ax=axs, label="epop")
        axs.set_xlabel("BS freq (GHz)")
        axs.set_ylabel("Time (us)")
        axs.set_title(f"bs_scale: {self.bs_scale:.4g}, VOP: {self.bs_vop}")
        fig.tight_layout()
        return fig, axs


    @annotate_method(plot_name="fft", axs_shape=(1,1))
    def plot_fft(self, axs=None, fft_peak_threshold=0.4, root_quadratic_fit:bool=True, 
                 fit_freq_min=None, fit_freq_max=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        from acadia_qmsmt.plotting.plotters import plot_pcolormesh_fft
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)
        plot_pcolormesh_fft(self.bs_frequencies/1e9, self.fft_freqs/1e6, self.fft_data, plot_ax=axs,
                            root_quadratic_fit=root_quadratic_fit, fft_peak_threshold=fft_peak_threshold,
                            fit_freq_min=fit_freq_min, fit_freq_max=fit_freq_max)

        axs.set_xlabel("BS freq (GHz)")
        axs.set_ylabel("FFT freq (MHz)")

        fig.tight_layout()
        return fig, axs


    # @annotate_method(button_name="update_NCO")
    # def update_NCO(self, NCO_Hz:float=None):
    #     self.update_io_yaml_field("bs_stimulus", "channel_config.nco_frequency", np.round(self.fitted_f0.n))

    
    # generate plots for each prep dynamically
    @annotate_method(is_customizer=True)
    def _generate_plots(self):

        def plot_factory(plot_name):
            @annotate_method(plot_name=plot_name, axs_shape=(1, 1))
            def plot(axs=None, log_scale:bool=False, bins:int=51):
                from acadia_qmsmt.plotting import plot_multiple_hist2d
                fig, axs = plot_multiple_hist2d(self.prep_data,
                                                plot_ax=axs, log_scale=log_scale, bins=bins)
                axs[0].set_title("prep")
                return fig, axs
            return plot

        # generate plotter functions and add them to class attributes 
        if self.cool_qm_rounds>0:
            setattr(self, "plot_prep_msmts", plot_factory("prep msmts"))