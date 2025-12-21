from typing import Union, Annotated

import numpy as np

from acadia import Acadia, DataManager
from acadia.utils import clock_monotonic_ns
from acadia.runtime import annotate_method
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, IOConfig

import logging

logger = logging.getLogger("acadia")

class LoopbackAmpSweepRuntime(QMsmtRuntime):
    """
    Play a pulse on the stimulus channel and capture the IQ trace on the capture channel.
    Sweep the amplitude of the pulse

    This can be used for experiments like ringdown or checking pulse shape
    """
    stimulus: IOConfig
    capture: IOConfig


    stimulus_pulse_name: str = "oct_test"
    capture_memory_name: str = "readout_trace"

    iterations: int
    run_delay:int = 200e3 #ns
    amp_list:Union[list, float, np.ndarray] = None # pulse amplitude to sweep over, if None, will use the pulse amplitude in the yaml file
    capture_delay: float = 0 # seconds, delay before starting the capture
    nco_frequency: float = None  # Hz, will overwrite the frequency in yaml file if not None

    figsize: tuple[int] = None
    
    yaml_path: str = None

    def main(self):
        stimulus_io = self.io("stimulus")
        capture_io = self.io("capture")

        readout_resonator = MeasurableResonator(stimulus_io, capture_io)


        # Create the record groups for saving captured data
        self.data.add_group("traces", uniform=True)
        self.data.add_group("t_data", uniform=False)

        # Core FPGA (PL) sequence
        def sequence(a: Acadia):

            with a.channel_synchronizer():

                readout_resonator.measure_trace(
                    stimulus_waveform_memory=self.stimulus_pulse_name,
                    capture_waveform_memory=self.capture_memory_name,
                    capture_delay=self.capture_delay
                )


        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()

        
        t_data = None
        if self.nco_frequency is not None:
            readout_resonator.set_frequency(self.nco_frequency)
        
        self._parse_amp_list()

        # Core python loop that will be running on the ARM (PS)
        for i in range(self.iterations):
            for amp in self.amp_list:

                # set the amplitude for the stimulus pulse
                stimulus_io.load_pulse(self.stimulus_pulse_name, scale=amp)
                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)
                wf = capture_io.get_waveform_memory(self.capture_memory_name)
                self.data[f"traces"].write(wf.array)
                
                # calculate t_list based on capture time and length of capture waveform
                if t_data is None:                    
                    capture_time = capture_io.get_config("memories", self.capture_memory_name,"length")
                    t_data = np.linspace(0, capture_time, len(wf.array), endpoint=False)
                    self.data["t_data"].write(t_data)

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

    def _parse_amp_list(self) -> list[float]:
        """
        Parse the amplitude list to a list of floats.
        If the input is a single float, return a list with that value.
        """
        if self.amp_list is None:
            self.amp_list = [self._ios["stimulus"].get_config("pulses", self.stimulus_pulse_name, "scale")]
        elif isinstance(self.amp_list, (int, float, complex)):
            self.amp_list = [self.amp_list]


    @annotate_method(is_data_processor=True)
    def process_current_data(self):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        self._parse_amp_list()
        # gather time and IQ trace data
        self.t_data = np.array(self.data["t_data"].records()).astype(float).squeeze()

        self.traces_iq = reshape_iq_data_by_axes(self.data["traces"].records(), self.amp_list, self.t_data)
        self.traces_pwr = self.traces_iq[...,0]**2 + self.traces_iq[...,1]**2
        

        if self.traces_iq is None:
            return
        else:
            completed_iterations = len(self.traces_iq)

        self.avg_trace_iq = np.mean(self.traces_iq, axis=0)
        self.avg_trace_pwr = np.mean(self.traces_pwr, axis=0)

        return  completed_iterations


    @annotate_method(plot_name="1 all traces, iq", axs_shape=(1,1))
    def plot_all_traces_iq(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1, 1), figsize=self.figsize)
        
        for i, amp in enumerate(self.amp_list):
            axs.plot(self.t_data, self.avg_trace_iq[i, :, 0], label=f"Amp: {amp:.2f}", color=f"C{i}")
            axs.plot(self.t_data, self.avg_trace_iq[i, :, 1], linestyle='--', color=f"C{i}")
        axs.set_xlabel("Time (ns)")
        axs.set_ylabel("Voltage [a.u.]")
        axs.legend()
        axs.grid(True)
        fig.tight_layout()
        return fig, axs

    @annotate_method(plot_name="2 all traces, pwr", axs_shape=(1,1))
    def plot_all_traces_pwr(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1, 1), figsize=self.figsize)
        
        for i, amp in enumerate(self.amp_list):
            axs.plot(self.t_data, self.avg_trace_pwr[i], label=f"Amp: {amp:.2f}", color=f"C{i}")
        axs.set_xlabel("Time (ns)")
        axs.set_ylabel("Power [a.u.]")
        axs.legend()
        axs.grid(True)
        fig.tight_layout()
        return fig, axs

    @annotate_method(plot_name="3 individual traces, iq", axs_shape=(1,1))
    def plot_ind_traces_iq(self, axs=None, sel_amp:Annotated[float, "slider", "self.amp_list"]=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1, 1), figsize=self.figsize)

        if sel_amp is None:
            sel_amp = self.amp_list[0]
        
        amp_index = np.argmin(np.abs(np.array(self.amp_list) - sel_amp))
        amp = self.amp_list[amp_index]
        axs.plot(self.t_data, self.avg_trace_iq[amp_index, :, 0], label=f"Amp: {amp:.2f}, real")
        axs.plot(self.t_data, self.avg_trace_iq[amp_index, :, 1], linestyle='--', label=f"Amp: {amp:.2f}, imag")
        axs.set_xlabel("Time (ns)")
        axs.set_ylabel("Voltage [a.u.]")
        axs.legend()
        axs.grid(True)
        return fig, axs


    @annotate_method(plot_name="4 individual traces, pwr", axs_shape=(1,1))
    def plot_ind_traces_pwr(self, axs=None, sel_amp:Annotated[float, "slider", "self.amp_list"]=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1, 1), figsize=self.figsize)

        if sel_amp is None:
            sel_amp = self.amp_list[0]
        
        amp_index = np.argmin(np.abs(np.array(self.amp_list) - sel_amp))
        amp = self.amp_list[amp_index]
        axs.plot(self.t_data, self.avg_trace_pwr[amp_index], label=f"Amp: {amp:.2f}, real")
        axs.set_xlabel("Time (ns)")
        axs.set_ylabel("Power [a.u.]")
        axs.legend()
        axs.grid(True)
        return fig, axs