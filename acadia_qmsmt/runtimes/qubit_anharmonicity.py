from typing import Union, Annotated

import numpy as np
from numpy.typing import NDArray

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method


class QubitAnharmonicityRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` for measuring the anharmonicity of a qubit.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    detune_frequencies: Union[list, np.ndarray]

    iterations: int
    run_delay: int


    ef_pulse_config: dict = None

    qubit_pulse_name: str = "R_x_180"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "matched"


    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)

        self.data.add_group(f"points", uniform=True)

        def sequence(a: Acadia):

            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name)
                qubit_stimulus_io.schedule_pulse(self.ef_pulse_config)
                qubit_stimulus_io.schedule_pulse(self.qubit_pulse_name)
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
        

        for i in range(self.iterations):
            for frequency in self.detune_frequencies:
                # load the modulated pulse into the waveform
                qubit_stimulus_io.load_pulse(self.ef_pulse_config, detune=frequency)
                
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
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

        data = reshape_iq_data_by_axes(self.data["points"].records(), self.detune_frequencies, to_complex=True)
        if data is None:
            return

        completed_iterations = len(data)
        readout_resonator = MeasurableResonator(self.io("readout_stimulus"), self.io("readout_capture"))

        self.data_complex = data
        self.shots = readout_resonator.classify_measurement(self.data_complex, readout_classifier)
        self.avg_shots = np.mean(self.shots, axis=0)


        from acadia_qmsmt.analysis.fitting import Lorentzian
        self.fit = Lorentzian(self.detune_frequencies, self.avg_shots)
        self.fitted_f0_MHz = self.fit.ufloat_results["x0"]/1e6
        self.qubit_nco = self._ios["qubit_stimulus"].get_config("channel_config", "nco_frequency")
        return completed_iterations
    

    @annotate_method(plot_name="1 qubit spectrocopy", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        self.fit.plot(axs, oversample=5, 
                      result_kwargs=dict(label=f"{self.fitted_f0_MHz:.5g} MHz"))

        axs.set_xlabel("Detuning [Hz]")
        axs.set_ylabel("shots")
        axs.legend()
        axs.set_title(f"NCO: {self.qubit_nco/1e9} GHz, fitted_deune: {self.fitted_f0_MHz} MHz")
        return fig, axs


    @annotate_method(plot_name="2 bin averaged", axs_shape=(1,1))
    def plot_bin_avg(self, axs=None, n_avg=1):
        from acadia_qmsmt.plotting import prepare_plot_axes, plot_binaveraged
        fig, axs = plot_binaveraged(self.detune_frequencies/1e6, self.shots.real, axs, n_avg=n_avg)
        axs.set_ylabel("Detuning [MHz]")
        return fig, axs


    @annotate_method(button_name="update ef frequency")
    def update_freq(self, target_waveform=["R_x_180_ef", "R_x_180_selective_ef"]):
        fitted_f0 = np.round(self.fitted_f0_MHz.n*1e6)
        for wf in target_waveform:
            self.update_io_yaml_field("qubit_stimulus", f"pulses.{wf}.detune", fitted_f0)



