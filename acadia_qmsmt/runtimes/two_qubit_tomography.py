import numpy as np

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig, TwoQubit
from acadia.runtime import annotate_method


class TwoQubitTomographyRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` subclass for testing two qubit tomography with a prepared product state.
    With the option of symmetrized msmt.
    """
    qubit1_stimulus: IOConfig
    readout1_stimulus: IOConfig
    readout1_capture: IOConfig

    qubit2_stimulus: IOConfig
    readout2_stimulus: IOConfig
    readout2_capture: IOConfig


    iterations: int
    run_delay: int
    
    qubit1_prep_angles: tuple = (0, 0) # polar and azimuthal angles, in degree, for qubit 1 state prepartion
    qubit1_pi_pulse_name: str = "R_x_180"

    readout1_pulse_name: str = "readout"
    capture1_memory_name: str = "readout_accumulated"
    capture1_window_name: str = "matched"

    qubit2_prep_angles: tuple = (0, 0) # polar and azimuthal angles, in degree, for qubit 2 state prepartion
    qubit2_pi_pulse_name: str = "R_x_180"

    readout2_pulse_name: str = "readout"
    capture2_memory_name: str = "readout_accumulated"
    capture2_window_name: str = "matched"


    figsize: tuple[int] = None
    yaml_path: str = None
    symmetrize: bool=True # when True, do extra measurements along the antiparallel Bloch sphere axes in
                          # each tomo direction to symmetrize the msmt error

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit1_stimulus_io = self.io("qubit1_stimulus")
        readout1_stimulus_io = self.io("readout1_stimulus")
        readout1_capture_io = self.io("readout1_capture")
        readout1_resonator = MeasurableResonator(readout1_stimulus_io, readout1_capture_io)
        qubit1 = Qubit(qubit1_stimulus_io, readout1_resonator)

        qubit2_stimulus_io = self.io("qubit2_stimulus")
        readout2_stimulus_io = self.io("readout2_stimulus")
        readout2_capture_io = self.io("readout2_capture")
        readout2_resonator = MeasurableResonator(readout2_stimulus_io, readout2_capture_io)
        qubit2 = Qubit(qubit2_stimulus_io, readout2_resonator)

        two_qubits = TwoQubit(qubit1, qubit2)

        self.data.add_group(f"points1", uniform=False)
        self.data.add_group(f"points2", uniform=False)


        cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))
        qubit1_prep_pulse = qubit1.make_rotation_pulse(*self.qubit1_prep_angles, self.qubit1_pi_pulse_name)
        qubit2_prep_pulse = qubit2.make_rotation_pulse(*self.qubit2_prep_angles, self.qubit2_pi_pulse_name)

        def core(a:Acadia):
            with a.channel_synchronizer():
                qubit1_stimulus_io.schedule_pulse(qubit1_prep_pulse)
                qubit2_stimulus_io.schedule_pulse(qubit2_prep_pulse)

        def sequence(a: Acadia):
            two_qubits.full_2q_tomo(cache, core, self.qubit1_pi_pulse_name, self.qubit2_pi_pulse_name,
                                    self.readout1_pulse_name, self.capture1_memory_name, self.capture1_window_name,
                                    self.readout2_pulse_name, self.capture2_memory_name, self.capture2_window_name, 
                                    symmetrize=self.symmetrize)


        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout1_resonator.load_windows()
        readout1_stimulus_io.load_pulse(self.readout1_pulse_name)
        qubit1_stimulus_io.load_pulse(qubit1_prep_pulse)


        readout2_resonator.load_windows()
        readout2_stimulus_io.load_pulse(self.readout2_pulse_name)
        qubit2_stimulus_io.load_pulse(qubit2_prep_pulse)

        two_qubits.load_tomo_pulses(self.symmetrize)


        num_runs = 36 if self.symmetrize else 9  # represents 9 bases of msmt
        for i in range(self.iterations):
            for idx_basis in range(num_runs):
                # Update the basis in the cache
                cache[0] = idx_basis

                # Capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)
                wf1 = readout1_capture_io.get_waveform_memory(self.capture1_memory_name)
                self.data[f"points1"].write(wf1.array)
                wf2 = readout2_capture_io.get_waveform_memory(self.capture2_memory_name)                
                self.data[f"points2"].write(wf2.array)
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
    def process_current_data(self):
        from acadia_qmsmt.analysis import reshape_iq_data_by_axes
        from acadia_qmsmt.analysis.tomography import xyz_to_full_tomo, expvals_to_rho


        msmt_bases = np.arange(36 if self.symmetrize else 9, dtype=int)

        data1 = reshape_iq_data_by_axes(self.data["points1"].records(), msmt_bases)
        data2 = reshape_iq_data_by_axes(self.data["points2"].records(), msmt_bases)
        
        if data1 is None:
            return
        else:
            completed_iterations = len(data1)
        self.data1_iq = data1.astype(float).view(complex).squeeze()
        self.data2_iq = data2.astype(float).view(complex).squeeze()

        self.avg1_iq = np.mean(self.data1_iq, axis=0)
        self.shots1 = np.sign(self.data1_iq.real)

        self.avg2_iq = np.mean(self.data2_iq, axis=0)
        self.shots2 = np.sign(self.data2_iq.real)


        self.tomo_shots = xyz_to_full_tomo(self.shots1, self.shots2)
        self.tomo_shots.pop("II")
        self.tomo_avg = {k: np.mean(v) for k, v in self.tomo_shots.items()}
        self.tomo_err = {k: np.std(v)/np.sqrt(len(v)) for k, v in self.tomo_shots.items()}
        self.rho = expvals_to_rho(self.tomo_avg)
        
        return completed_iterations
    
    
    @annotate_method(plot_name="1 2-qubit tomo", axs_shape=(1,1))
    def plot_data(self, axs=None):
        from acadia_qmsmt.plotting.plotters import plot_histogram

        fig, axs = plot_histogram(self.tomo_avg, self.tomo_err, plot_axs=axs)
        axs.set_xlabel("Msmt Axis")
        axs.set_ylabel("Expectations")
        axs.set_ylim(-1.02, 1.02) 
        
        return fig, axs
    

    @annotate_method(plot_name="2 density matrix", axs_shape=(1,1))
    def plot_dens_mat(self, axs=None):
        from acadia_qmsmt.plotting.plotters import plot_density_matrix
        fig, axs = plot_density_matrix(self.rho, plot_ax=axs, max_amp=0.5)

        import qutip
        rho_qobj = qutip.Qobj(self.rho, [[2, 2], [2, 2]])
        conc = qutip.concurrence(rho_qobj)
        axs.set_title(f'Concurrence: {conc:.4g}')

        return fig, axs
    

  

