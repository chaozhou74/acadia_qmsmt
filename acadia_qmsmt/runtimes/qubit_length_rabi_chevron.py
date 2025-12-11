"""
We generally don't use a length Rabi experiment to tune up a π-pulse; this is just to show 
how a pulse length sweep could be done with a register. 
"""

from typing import Union, Annotated

import numpy as np

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method


class QubitLengthRabiChevronRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for sweeping the rabi time and qubit frequency
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    flat_length_list: Union[list, np.ndarray]
    qubit_frequencies: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_pulse_name: str = "R_x_180"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = None

    qubit_amp: float = None

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

        self.data.add_group(f"points", uniform=True)

        # Create an array in the cache that we can use to pass the
        # delay value to the sequencer so that we don't have to reassemble every time
        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))

        if not qubit_stimulus_io.get_config("pulses", self.qubit_pulse_name).get("use_stretch"):
            raise ValueError("qubit pulse must have use_stretch=True for the length to be sweepable via a register")


        def sequence(a: Acadia):
            # Initialize a DSP to act as a counter
            length_reg = a.sequencer().Register()

            # Load the counter with the value we put into the cache
            length_reg.load(cache[0])

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name, stretch_length=length_reg)

            with a.channel_synchronizer():
                readout_resonator.measure("readout", "readout_accumulated", self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_pulse_name, scale=self.qubit_amp)


        # Determine how many cycles each delay interval should be
        dsp_count_values = self.acadia.seconds_to_cycles(self.flat_length_list)

        
        for i in range(self.iterations):
            for freq in self.qubit_frequencies:
                qubit.set_frequency(freq)
                for wf_idx, stretch_len in enumerate(dsp_count_values):
                    cache[0] = stretch_len
                    # capture data and put in the corresponding group
                    self.acadia.run(minimum_delay=self.run_delay)
                    wf = readout_capture_io.get_waveform_memory("readout_accumulated")
                    self.data[f"points"].write(wf.array)

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
    def process_current_data(self, readout_classifier: Annotated[str, "IOConfig", "readout_capture.classifiers"]=None):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes

        data = reshape_iq_data_by_axes(self.data["points"].records(), self.qubit_frequencies,  self.flat_length_list, to_complex=True)
        if data is None:
            return

        completed_iterations = len(data)
        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))

        self.data_complex = data
        self.shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        self.avg_shots = np.mean(self.shots, axis=0)
        self.sigma_shots = np.std(self.shots, axis=0) / np.sqrt(completed_iterations)

        return completed_iterations
    

    @annotate_method(plot_name="1 qubit length rabi chevron", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        vmin = np.min(self.avg_shots)
        vmax = np.max(self.avg_shots)

        pcm = axs.pcolormesh(self.qubit_frequencies/1e6, self.flat_length_list*1e6, self.avg_shots.T, vmin=vmin, vmax=vmax, cmap="bwr")
        fig.colorbar(pcm, ax=axs)

        axs.set_xlabel("Qubit Frequency [MHz]")
        axs.set_ylabel("Pulse Flat Time [us]")

        fig.tight_layout()
        return fig, axs
    
