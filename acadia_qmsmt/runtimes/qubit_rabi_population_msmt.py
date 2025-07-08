from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
from acadia.runtime import annotate_method


class QubitRpmRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` for calibrating the amplitudes of pulses for qubit drives.plot_pcolormesh_fft
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    # Note that these amplitudes override the ``scale`` parameter in the configuration
    qubit_amplitudes: Union[list, np.ndarray]

    iterations: int
    run_delay: int

    qubit_ge_pulse_name: str = "R_x_180"
    qubit_ef_pulse_name: str = "R_x_180_ef"
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "matched"

    plot: bool = True
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

        qubit_ge_1st_pulse = qubit_stimulus_io.duplicate_pulse(self.qubit_ge_pulse_name)
        
        def sequence(a: Acadia):
            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(qubit_ge_1st_pulse)
                qubit_stimulus_io.schedule_pulse(self.qubit_ef_pulse_name)
                qubit_stimulus_io.schedule_pulse(self.qubit_ge_pulse_name)
                a.barrier()
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)
        qubit_stimulus_io.load_pulse(self.qubit_ge_pulse_name)

        qubit_ge_pi_pulse_scale = qubit_stimulus_io.get_config("pulses", self.qubit_ge_pulse_name, "scale")
        for i in range(self.iterations):
            for init_state in (0, 1):
                qubit_stimulus_io.load_pulse(qubit_ge_1st_pulse, scale = init_state * qubit_ge_pi_pulse_scale)
                for amplitude in self.qubit_amplitudes:
                    qubit_stimulus_io.load_pulse(self.qubit_ef_pulse_name, scale=amplitude)
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
        if self.plot:
            from acadia_qmsmt.plotting import save_registered_plots
            save_registered_plots(self)



    @annotate_method(is_data_processor=True)
    def process_current_data(self):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        data = reshape_iq_data_by_axes(self.data["points"].records(), (0, 1), self.qubit_amplitudes)
        if data is None:
            return
        else:
            completed_iterations = len(data)
        self.data_iq = data.astype(float).view(complex).squeeze()


        self.data_to_fit = np.mean((1-np.sign(self.data_iq.real))/2, axis=0)
        self.data_sigma = np.std((1-np.sign(self.data_iq.real))/2, axis=0)/np.sqrt(len(self.data_iq))


        from acadia_qmsmt.analysis.fitting import Cosine
        self.fit_1 = Cosine(self.qubit_amplitudes, self.data_to_fit[1], sigma=self.data_sigma[1],
                            params={"phi":{"value": np.pi,
                                         "fixed": True}
                                    })

        fitted_f_1 = self.fit_1.ufloat_results["f"].n

        # use the period of the high contrast one as fixed parameter for the low contrast one
        self.fit_0 = Cosine(self.qubit_amplitudes, self.data_to_fit[0], sigma=self.data_sigma[0],
                            params={"f":{"value":fitted_f_1,
                                         "fixed": True},
                                    "phi":{"value": np.pi,
                                         "fixed": True}
                                    })
        
        self.fitted_A_0 = abs(self.fit_0.ufloat_results["A"])
        self.fitted_A_1 = abs(self.fit_1.ufloat_results["A"])

        return completed_iterations
    

    @annotate_method(plot_name="1 qubit power rabi", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        self.fit_0.plot(axs, oversample=5, result_kwargs=dict(label=f"prep g, A: {self.fitted_A_0:.5g}"))
        self.fit_1.plot(axs, oversample=5, result_kwargs=dict(label=f"prep e, A: {self.fitted_A_1:.5g}"))

        axs.set_xlabel("Drive Amplitude [DAC]")
        axs.set_ylabel("e pop")
        axs.set_ylim(-0.02, 1.02)

        axs.legend()
        axs.set_title(f"Thermal pop: {(self.fitted_A_0/(self.fitted_A_0+self.fitted_A_1)*100):.5g} %")
        return fig, axs


