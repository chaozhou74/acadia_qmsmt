from typing import Union, Annotated

import numpy as np

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig, QubitQmCooler
from acadia.runtime import annotate_method


class BSAmpFreqSweepPulseSSBRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for sweeping the amp and freq of bs pulse without a rotation.
    This is what we call the "WiFi" experiment, when we use a long pulse and very low amplitude
    to measure frequency accurately
    """
    qubit_stimulus: IOConfig
    bs_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    bs_amplitudes: Union[list, np.ndarray]
    bs_detunes: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_pulse_name: str = "R_x_180"
    bs_pulse_name: str = None
    
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "boxcar"

    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    use_cool:bool = False
    cool_swap_pulse_name: str = None
    cool_qm_rounds: int = 2


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


        self.data.add_group(f"points", uniform=True)

        if self.cool_swap_pulse_name is None and self.use_cool:
            raise ValueError("If use_cool is True, cool_swap_pulse_name must be provided")

        def sequence(a: Acadia):
            # cool qubit and QM
            if self.use_cool:
                cooler.cool(1, self.qubit_pulse_name, # assuming the qubit pulse is going to be the pi pulse...
                            self.readout_pulse_name, self.capture_memory_name, self.capture_window_name,
                            self.cool_swap_pulse_name, self.cool_qm_rounds) 

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name)
                qubit_stimulus_io.dwell(10e-9)
                a.barrier()
                bs_stimulus_io.schedule_pulse(self.bs_pulse_name)
                a.barrier()
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_pulse_name)
        if self.use_cool:
            bs_stimulus_io.load_pulse(self.cool_swap_pulse_name)


        # run amp and freq sweep
        configure_streams = True
        for i in range(self.iterations):
            for freq_idx, freq in  enumerate(self.bs_detunes):
                for amp_idx, amp in enumerate(self.bs_amplitudes):

                    bs_stimulus_io.load_pulse(self.bs_pulse_name, scale = amp, detune = freq)

                    # capture data and put in the corresponding group
                    self.acadia.run(minimum_delay=self.run_delay, configure_streams=configure_streams) 
                    wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
                    self.data[f"points"].write(wf.array)

                    # no need to redo the configure stream after the first run (further reduce runtime to 90us/iter)
                    configure_streams = False

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()


    def initialize(self):
        pass

    def update(self):
        # get current completed data
        # self.process_current_data()

        # save current data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        from acadia_qmsmt.plotting import save_registered_plots
        if self.plot:
            save_registered_plots(self)


    @annotate_method(is_data_processor=True)
    def process_current_data(self, readout_classifier: Annotated[str, "IOConfig", "readout_capture.classifiers"]=None):
        # First make sure that we actually have new data to process
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data = reshape_iq_data_by_axes(self.data["points"].records(), self.bs_detunes, self.bs_amplitudes, to_complex=True)
        if data is None:
            return

        completed_iterations = len(data)
        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))

        self.data_complex = data
        self.shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        self.avg_shots = np.mean(self.shots, axis=0)

        return completed_iterations


    @annotate_method(plot_name="wifi", axs_shape=(1,1))
    def plot_wifi(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, ax = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        pcm = ax.pcolormesh(self.bs_detunes/1e6, self.bs_amplitudes, self.avg_shots.T, vmin=0, vmax=1, cmap="bwr")
        fig.colorbar(pcm, ax=ax, label="epop")

        ax.set_xlabel("BS freq (MHz)")
        ax.set_ylabel("BS amplitude (DAC)")

        fig.tight_layout()
        return fig, axs



