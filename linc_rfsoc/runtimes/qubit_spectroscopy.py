from dataclasses import dataclass

from acadia import Runtime

# copied from the original acadia runtime

@dataclass
class QubitSpectroscopyRuntime(Runtime):
    """
    A :class:`Runtime` for sweeping the frequency of a drive on a qubit and its
    readout resonator.
    """
    
    qubit_frequencies: list 
    readout_frequencies: list 
    qubit_stimulus: dict
    readout_stimulus: dict
    readout_capture: dict
    readout_electrical_delay: float = 0
    plot: bool = False
    iterations: int = 10
        
    def main(self):        
        from acadia import Acadia, DataManager
        
        acadia = Acadia()
        qubit_channel = acadia.channel(self.qubit_stimulus["channel"])
        readout_stimulus_channel = acadia.channel(self.readout_stimulus["channel"])
        readout_capture_channel = acadia.channel(self.readout_capture["channel"])

        qubit_stimulus_waveform = acadia.create_waveform(qubit_channel, **self.qubit_stimulus["waveform"])
        readout_stimulus_waveform = acadia.create_waveform(readout_stimulus_channel, **self.qubit_stimulus["waveform"])
        readout_capture_waveform = acadia.create_waveform(readout_capture_channel, decimation=0, **self.readout_capture["waveform"]) 
                
        self.data.add_group("traces", uniform=True)
                
        def sequence(a: Acadia):
            capture_stream, kernel = acadia.configure_cmacc(readout_capture_channel, reset_fifo=True)
            acadia.cmacc_load(capture_stream, 0)

            with a.channel_synchronizer():
                a.schedule_waveform(qubit_stimulus_waveform)
                a.barrier()
                a.schedule_waveform(readout_stimulus_waveform)
                a.stream(capture_stream, readout_capture_waveform)

            return kernel

        kernel = acadia.compile(sequence)
        acadia.attach()
        acadia.align_tile_latencies()

        qubit_channel.set(nco_update_event_source="sysref", **self.qubit_stimulus["datapath"])
        readout_stimulus_channel.set(nco_update_event_source="sysref", **self.readout_stimulus["datapath"])
        readout_capture_channel.set(nco_update_event_source="sysref", **self.readout_capture["datapath"])  

        # Load the stimulus waveforms
        qubit_stimulus_waveform.set(**self.qubit_stimulus["signal"])
        readout_stimulus_waveform.set(**self.readout_stimulus["signal"])

        import numpy as np
        kernel.set(np.float64(0.5))

        acadia.assemble()
        acadia.load()

        for i in range(self.iterations):
            for readout_frequency in self.readout_frequencies:
                for qubit_frequency in self.qubit_frequencies:
                    
                    # update the frequencies of everything
                    acadia.update_nco_frequency(readout_stimulus_channel, frequency=readout_frequency)
                    acadia.update_nco_frequency(readout_capture_channel, frequency=-readout_frequency)
                    acadia.update_nco_frequency(qubit_channel, qubit_frequency)
                    
                    # ensure the readout NCOs are coherent by resetting the phase accumulator
                    acadia.reset_nco_phase(readout_stimulus_channel)
                    acadia.reset_nco_phase(readout_capture_channel)
                    acadia.reset_nco_phase(qubit_channel)

                    # synchronously update everything
                    acadia.update_ncos_synchronized()

                    acadia.run()
                    self.data["traces"].write(readout_capture_waveform.array)            
                    
                    # Check whether the host wants data or whether it's requesting a hangup
                    if self.data.serve() == DataManager.serve_hangup():
                        # The client will not be requesting any more data, close the data manager
                        # and exit
                        self.data.disconnect()
                        return
        
        self.final_serve()

    def initialize(self):
        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cm

            self.fig = plt.figure(figsize=(8,3))
            gs = self.fig.add_gridspec(1, 2, bottom=0.2, right=0.8, width_ratios=[20, 1], wspace=0.1)
            gs_plots = gs[0].subgridspec(1, 2, wspace=0.35)
            ax_mag, ax_phase = gs_plots.subplots()
            ax_colorbar = self.fig.add_subplot(gs[1])

            # Create a plot for the spectral magnitude
            cmap = plt.get_cmap("Spectral")
            norm = colors.Normalize(self.readout_frequencies[0], self.readout_frequencies[-1])
            sm = cm.ScalarMappable(norm, cmap)
            
            self.lines_mag = [DynamicLine(ax_mag, ".-", c=sm.to_rgba(a)) for a in self.readout_frequencies]
            ax_mag.set_xlabel("Qubit Frequency [MHz]")
            ax_mag.set_ylabel("Magnitude [arb. V*s]")
            ax_mag.set_title("Spectral Magnitude")
            ax_mag.grid()
            
            # Create a plot for the spectral phase
            self.lines_phase = [DynamicLine(ax_phase, ".-", c=sm.to_rgba(a)) for a in self.readout_frequencies]
            ax_phase.set_xlabel("Frequency [MHz]")
            ax_phase.set_ylabel("Phase [rad.]")
            ax_phase.set_title("Spectral Phase")
            ax_phase.grid()

            # Create a colorbar
            self.fig.colorbar(sm, cax=ax_colorbar, label="Readout Frequency")

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm
        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0
        self.readout_frequencies_progress_bar = tqdm(desc="Readout Frequencies", dynamic_ncols=True, total=len(self.readout_frequencies)*self.iterations)
        self.readout_frequencies_previous = 0
        self.qubit_frequencies_progress_bar = tqdm(desc="Qubit Frequencies", dynamic_ncols=True, total=len(self.readout_frequencies)*len(self.qubit_frequencies)*self.iterations)
        self.qubit_frequencies_previous = 0
        

        import numpy as np
        self.data_summed = None
        self.electrical_delay_phases = np.exp(2*np.pi*1j*self.readout_frequencies*self.readout_electrical_delay)[:,None]
        self.data_complex = np.empty((len(self.readout_frequencies), len(self.qubit_frequencies)), dtype=np.complex128)

    def update(self):
        import numpy as np
        from acadia.waveforms import Waveform

        # First make sure that we actually have new data to process
        if "traces" not in self.data:
            return
        
        # Update the progress bars
        completed_iterations = len(self.data["traces"]) // (len(self.readout_frequencies)*len(self.qubit_frequencies))
        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        completed_readout_frequencies = len(self.data["traces"]) // len(self.qubit_frequencies)
        self.readout_frequencies_progress_bar.update(completed_readout_frequencies - self.readout_frequencies_previous)

        completed_qubit_frequencies = len(self.data["traces"])
        self.qubit_frequencies_progress_bar.update(completed_qubit_frequencies - self.qubit_frequencies_previous)

        # Only continue processing data if we have at least one complete iteration
        if completed_iterations != 0:
            valid_traces = completed_iterations*len(self.qubit_frequencies)*len(self.readout_frequencies)
            data = self.data["traces"].records()[:valid_traces, ...]

            samples_per_trace = data.shape[-2]
            data_reshaped = data.reshape(-1, len(self.readout_frequencies), len(self.qubit_frequencies), samples_per_trace, 2)
            new_data = data_reshaped[self.iterations_previous:, :, :, :, :]

            # Sum the new data and then add it to the aggregated array of trace data
            new_data_summed = np.sum(new_data, axis=(0,3), keepdims=False)
            if self.data_summed is None:
                self.data_summed = new_data_summed
            else:
                self.data_summed += new_data_summed
            
            self.data_complex = Waveform.sample_to_complex(self.data_summed, scale=1/completed_iterations)

            # Apply the electrical delay
            self.data_complex *= self.electrical_delay_phases

            if self.plot:
                for idx,readout_frequency in enumerate(self.readout_frequencies):
                    # Don't rescale the plot when updating the lines, we'll do it all at once when we have the full plot
                    self.lines_mag[idx].update(self.qubit_frequencies, np.abs(self.data_complex[idx,:]), rescale_axis=False)
                    self.lines_phase[idx].update(self.qubit_frequencies, (np.angle(self.data_complex[idx,:])), rescale_axis=False)

                # Rescale axes and redraw plot
                self.lines_mag[0]._ax.relim()
                self.lines_mag[0]._ax.autoscale(tight=True)
                self.lines_phase[0]._ax.relim()
                self.lines_phase[0]._ax.autoscale(tight=True)
                self.fig.canvas.draw_idle()

        self.iterations_previous = completed_iterations
        self.readout_frequencies_previous = completed_readout_frequencies
        self.qubit_frequencies_previous = completed_qubit_frequencies

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        self.readout_frequencies_progress_bar.close()
        self.qubit_frequencies_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)  
