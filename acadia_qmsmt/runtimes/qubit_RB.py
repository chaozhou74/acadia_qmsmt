from typing import Union, Literal

import numpy as np
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig, QubitQmCooler
from acadia.runtime import annotate_method

def esplit(s, sep=','):
    'Split string by commas. If the string is empty return the empty list'
    if not s:
        return []
    return s.split(',')

class QubitRBRuntime(QMsmtRuntime):
    """
    A :class:`Runtime` that runs single qubit randomized benchmarking sequences. Currently does a gate set all 180 and 90 degree rotations
    """

    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    seq_lengths: Union[list, np.ndarray]

    iterations: int
    run_delay: int


    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    my_rng_seed: int = int(np.sum([ord(x) for x in "cz445"])) # seed set to cz445 by JWOG

    
    def compose_state_maps(self, operation_names): 
        """
            Return a list of length 6 where list[state] is what you get when
            you apply operation_names[-1], then operation_names[-2], then ..., operation_names[0] to state.
        """
        ret = []
        for psi in range(6):
            for name in operation_names[::-1]:
                psi = self.state_map_dict[name][psi]
            ret.append(psi)
        return ret
    def build_state_map_data(self):
        self.state_map_data = [self.compose_state_maps(esplit(name)) for name in self.operation_names]

    def main(self):

        self.state_map_dict = {
            'I' : [0, 1, 2, 3, 4, 5],
            'X' : [1, 0, 2, 3, 5, 4],
            'Xm': [1, 0, 2, 3, 5, 4],
            'Y' : [1, 0, 3, 2, 4, 5],
            'Ym': [1, 0, 3, 2, 4, 5],
            'Z' : [0, 1, 3, 2, 5, 4],
            'X2': [4, 5, 2, 3, 1, 0],
            'X2m':[5, 4, 2, 3, 0, 1],
            'Y2': [3, 2, 0, 1, 4, 5],
            'Y2m':[2, 3, 1, 0, 4, 5],
            'Z2': [0, 1, 5, 4, 2, 3],
            'Z2m':[0, 1, 4, 5, 3, 2],
            'H' : [2, 3, 0, 1, 5, 4],
        }
        
        
        # Section 1 - some setup code
        # @Chao I'd like more details about the finer details of the logger and how to use it
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)



        # Need to create a place for the acquired data to go. By convention, if we are only looking at thresholded values, we'll use "points", if we are taking entire trajectories we'll use "traces"
        
        self.data.add_group(f"points", uniform=True) # the 'uniform' boolean is about whether you'll be saving data of the same shape every single time (almost always true)


        num_pulses_cache = self.acadia.CacheArray(shape=(1,), dtype=np.dtype("<i4"))
        random_pulse_seq_cache = self.acadia.CacheArray(shape=(int(np.max(self.seq_lengths)),), dtype=np.dtype("<i4"))


        qubit_pi_0 = qubit.make_rotation_pulse(180.,0.,"R_x_180")
        qubit_pi_90 = qubit.make_rotation_pulse(180.,90.,"R_x_180")
        qubit_pi_180 = qubit.make_rotation_pulse(180.,180.,"R_x_180")
        qubit_pi_270 = qubit.make_rotation_pulse(180.,270.,"R_x_180")
        
        qubit_pi_over_2_0 = qubit.make_rotation_pulse(90.,0.,"R_x_180")
        qubit_pi_over_2_90 = qubit.make_rotation_pulse(90.,90.,"R_x_180")
        qubit_pi_over_2_180 = qubit.make_rotation_pulse(90.,180.,"R_x_180")
        qubit_pi_over_2_270 = qubit.make_rotation_pulse(90.,270.,"R_x_180")
        


        qubit_undo_pulse = qubit_stimulus_io.duplicate_pulse("R_x_180")

        # Section 2 - the sequence
        def sequence(a: Acadia):

            # Initialize a register for the number of pulses to be played
            pulses_to_play_reg = a.sequencer().Register()
            pulse_counter_reg = a.sequencer().Register()
            pulse_selecter_reg = a.sequencer().Register()

            # Load the counter with the number of pulses that we will perform
            pulse_counter_reg.load(0)
            pulse_selecter_reg.load(random_pulse_seq_cache[0]) # Load the pulse selecter with first index
            pulses_to_play_reg.load(num_pulses_cache[0])

            # prepare qubit
            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse("R_x_180")

            with a.sequencer().test(pulses_to_play_reg != 0): # if pulses_to_play = 0, just do readout
                with a.sequencer().repeat_until(pulse_counter_reg == pulses_to_play_reg):
                    with a.sequencer().test(a.channel_is_fifo_empty(qubit_stimulus_io.channel)):
                        with a.sequencer().test(pulse_selecter_reg == 0):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_0) # play a pulse
                        
                        with a.sequencer().test(pulse_selecter_reg == 1):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_90) # play a pulse

                        with a.sequencer().test(pulse_selecter_reg == 2):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_180) # play a pulse
                        
                        with a.sequencer().test(pulse_selecter_reg == 3):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_270) # play a pulse

                        with a.sequencer().test(pulse_selecter_reg == 4):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_0) # play a pulse
                        
                        with a.sequencer().test(pulse_selecter_reg == 5):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_90) # play a pulse
                                
                        with a.sequencer().test(pulse_selecter_reg == 6):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_180) # play a pulse
                        
                        with a.sequencer().test(pulse_selecter_reg == 7):
                            with a.channel_synchronizer(block=False):
                                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_270) # play a pulse

                        pulse_counter_reg.load(pulse_counter_reg + 1) # increment pulse counter
                        pulse_selecter_reg.load(random_pulse_seq_cache[pulse_counter_reg]) # pick next random pulse to play

                        with a.sequencer().test(pulse_counter_reg == pulses_to_play_reg): # if all the pulses have been played, now play the 'undo' pulse
                            with a.channel_synchronizer(block=True):
                                qubit_stimulus_io.schedule_pulse(qubit_undo_pulse) # play a pulse
                                a.barrier()
                                readout_resonator.measure("readout", "readout_accumulated", self.readout_window_name)
            
            with a.sequencer().test(pulses_to_play_reg == 0): # if pulses_to_play = 0, just do readout
                with a.channel_synchronizer(block=True):
                    readout_resonator.measure("readout", "readout_accumulated", self.readout_window_name)


        # section 3 - function calls equivalent to "make_tables"
        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        # Section 4 - more setup 
        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse("readout")

        qubit_pi_scale = qubit_stimulus_io.get_config("pulses", "R_x_180","scale")

        qubit_stimulus_io.load_pulse("R_x_180")


        qubit_stimulus_io.load_pulse(qubit_pi_0)
        qubit_stimulus_io.load_pulse(qubit_pi_90)
        qubit_stimulus_io.load_pulse(qubit_pi_180)
        qubit_stimulus_io.load_pulse(qubit_pi_270)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_0)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_90)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_180)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_270)
        
        # Section 5 - outer 'sweeper' for-loop(s)

        self.operation_names = ['X','Y','Xm','Ym','X2','Y2','X2m','Y2m']
        self.build_state_map_data()
        state_map = np.array(self.state_map_data).flatten()

        undo_angles = [0, 1., 1/2, -1/2, -1/2, 1/2]
        undo_phases = [0, 0, np.pi/2, np.pi/2, 0, 0]

    
        np.random.seed(self.my_rng_seed)
        for i in range(self.iterations): # always sweep iterations on the outer loop!
            for j,num_pulses in enumerate(self.seq_lengths):

                num_pulses_cache[0] = num_pulses # load number of pulses to play into cache

                my_random_sequence = np.random.randint(0,8,int(np.max(self.seq_lengths))) # generate random pulse sequence
                random_pulse_seq_cache[0:int(np.max(self.seq_lengths))] = list(my_random_sequence) # load into cache
                
                my_state = 0
                for operation_idx in random_pulse_seq_cache[0:num_pulses]: # compute undo state
                    my_state = state_map[6 * operation_idx + my_state]

                logger.info(f"gate string  {random_pulse_seq_cache[0:num_pulses]} yields state {my_state} after {num_pulses} pulses")
                qubit_stimulus_io.load_pulse(qubit_undo_pulse,scale=qubit_pi_scale*undo_angles[my_state]*np.exp(1j*undo_phases[my_state])) # load undo pulse
                
                # run and get results, write the results to data
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory("readout_accumulated")
                self.data[f"points"].write(wf.array)

            # This allows for aborting part-way through experiment
            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve() # If you're all done with the experiment, do various disconnect/wrapping up things in the background


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
        self.points_per_iter = len(self.seq_lengths)
        completed_iterations = len(self.data["points"]) // self.points_per_iter
        if "points" not in self.data or len(self.data["points"]) < self.points_per_iter:
            return

        valid_points = completed_iterations * self.points_per_iter
        data = self.data["points"].records()[:valid_points, ...]
        data = data.reshape(completed_iterations, len(self.seq_lengths), 2)

        # Threshold the data according to the I quadrature
        self.shots = (1 - np.sign(data[..., 0], dtype=np.int32))/2
        self.avg = np.mean(self.shots, axis=0)
        self.std_errs = np.std(self.shots, axis=0)/np.sqrt(len(self.shots))

        p0 = (abs(np.max(self.avg)) - abs(np.min(self.avg)), self.seq_lengths[-1]/5, self.avg[-1])

        from acadia_qmsmt.analysis.fitting.exponential import Exponential
        self.RB_fit = Exponential(self.seq_lengths, self.avg, sigma=self.std_errs, params={"tau": {"value": self.seq_lengths[-1]*0.2, "bounds": (0, self.seq_lengths[-1]*10.)},
                                                                                        "A": {"value": p0[0], "bounds": (0, 1)},
                                                                                        "of": {"value": 0.5, "bounds": (0, 1)}
                                                                                        })

        self.fitted_tau = self.RB_fit.ufloat_results['tau'].n
        self.gate_fidelity = np.exp(-1/self.fitted_tau) 
        return completed_iterations



    @annotate_method(plot_name="RB decay thresholded", axs_shape=(1,1))
    def plot_RB_decay(self, axs=None):

        fig, axs = self.RB_fit.plot(axs, result_kwargs={"label": f"tau: {self.fitted_tau:.5g} Gates"})
        axs.set_xlabel("Number of pulses")

        axs.grid(True)
        axs.set_ylim(-0.02, 1.02)
        
        axs.set_title(f"Qubit RB, F = {self.gate_fidelity:.5g}")
        fig.tight_layout()
        return fig, axs
