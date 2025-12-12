from typing import Union

import numpy as np

from acadia import Acadia, DataManager
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig
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

    qubit_pi_name: str = "R_x_180"

    post_sequence_dwell: int = 0e-9 # dwell time after the sequence before readout
    do_silent_pulse_debug: bool = False # if true, all pulses except for initial pi pulses have zero amplitude, should look like T1 experiment
    readout_pulse_name: str = "readout"
    capture_memory_name: str = "readout_accumulated"
    capture_window_name: str = "matched"
    

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

        silent_pulse_debug = self.do_silent_pulse_debug # if 1, all pulses are silent (for debugging timing issues)



        if self.do_silent_pulse_debug:
            logger.warning("NOTE: All pulses except for initial pi pulses have zero amplitude, should look like T1 experiment!!!")


        qubit_pi_0 = qubit.make_rotation_pulse(180.*(1-silent_pulse_debug),0.,self.qubit_pi_name)
        qubit_pi_90 = qubit.make_rotation_pulse(180.*(1-silent_pulse_debug),90.,self.qubit_pi_name)
        qubit_pi_180 = qubit.make_rotation_pulse(180.*(1-silent_pulse_debug),180.,self.qubit_pi_name)
        qubit_pi_270 = qubit.make_rotation_pulse(180.*(1-silent_pulse_debug),270.,self.qubit_pi_name)
        
        qubit_pi_over_2_0 = qubit.make_rotation_pulse(90.*(1-silent_pulse_debug),0.,self.qubit_pi_name)
        qubit_pi_over_2_90 = qubit.make_rotation_pulse(90.*(1-silent_pulse_debug),90.,self.qubit_pi_name)
        qubit_pi_over_2_180 = qubit.make_rotation_pulse(90.*(1-silent_pulse_debug),180.,self.qubit_pi_name)
        qubit_pi_over_2_270 = qubit.make_rotation_pulse(90.*(1-silent_pulse_debug),270.,self.qubit_pi_name)
        

        RB_pulse_channel = qubit_stimulus_io.channel#  self.acadia.channel("DAC4")
        pulse_command_cache = self.acadia.CacheArray(shape=int(max(self.seq_lengths)+1), dtype=np.dtype("<i4"))
        num_pulses_cache = self.acadia.CacheArray(shape=1, dtype=np.dtype("<i4"))

        # We'll create a dummy waveform that does nothing (see below)
        dummy = self.acadia.create_waveform_memory(RB_pulse_channel, length=20e-9)

        my_seq_GCD = np.gcd.reduce(self.seq_lengths)
        # Section 2 - the sequence
        def sequence(a: Acadia):


            # We'll use a DSP to keep track of which cache entry we'll load
            # Rather than just counting from zero, we'll load the DSP with the bus address 
            # of the first element of the array (that is, it's a pointer to the array's first element
            # in the physical address space of the sequencer). We'll then increment the pointer itself
            # to iterate through the array, so that the value in the DSP is the cache address to load
            # from rather than just a count.
            base_address = a._firmware.sequencer_bus_decoder["cache"].address().value()
            command_pointer = a.sequencer().DSP()
            command_pointer.load(base_address + pulse_command_cache.index)

            # # Now that we've loaded the DSP, the only thing we'll ever do to it from now on
            # # is increment it. Therefore, we'll preload the DSP configuration for increment-by-one
            # # This also makes it possible to increment the DSP just by pulsing CEP, which can be
            # # baked into an instruction.
            command_pointer.configure(mode="P+1", dsp_cep="reset")

            # # In order to know when we're done, we'll also store the value of the last
            # # pointer into a register and check to see when we reach it
            final_pointer = a.sequencer().Register()
            final_pointer.load(base_address + pulse_command_cache.index + num_pulses_cache[0])



            with a.channel_synchronizer():
                qubit_stimulus_io.schedule_pulse(self.qubit_pi_name)


            with a.sequencer().test(command_pointer != final_pointer): # convert the do_while into a while
                with a.sequencer().repeat_until(command_pointer == final_pointer): # This will play every pulse that we specify in cache

                    # Wait for the DMA to indicate that we can load more data
                    with a.sequencer().repeat_until(a.channel_is_fifo_almost_empty(RB_pulse_channel)): # converted to is_empty from is_almost_empty during testing
                        pass
                    # NOTE: If the above fifo check is commented out, minimum pulse time is 80ns instead of 110ns


                    # Retrieve the next command directly from the cache
                    # Because the DSP contains a bus address rather than an offset,
                    # we need to use the bus read function that accepts an arbitrary address.
                    # Different regions on the bus have different latencies; this is usually 
                    # taken care of by whatever high-level function is interacting with the bus, 
                    # but because we're reading from an arbitrary bus address, we need to make 
                    # sure that we wait an appropriate amount of time. Fortunately, we know that
                    # the read will always be in the cache, so we can tell the read function that the 
                    # latency to use is that of the cache
                    command = a.sequencer().bus_read(command_pointer, latency=a._bus_latency("cache"))


                    # Now we play the pulse by issuing the command from the cache directly to the DMA
                    with a.channel_synchronizer(block=False):
                        a.schedule_direct(RB_pulse_channel, command)

                    # Finally, we increment the cache pointer. Because we already configured 
                    # the DSP above for P+1, we just pulse CEP to increment it in-place
                    command_pointer.pulse_cep()




            with a.channel_synchronizer():
                readout_stimulus_io.dwell(self.post_sequence_dwell)
                readout_capture_io.dwell(self.post_sequence_dwell)
                readout_resonator.measure(self.readout_pulse_name, self.capture_memory_name, self.capture_window_name)

            readout_resonator.wait_until_measurement_done()


            with a.channel_synchronizer(block=True):
                qubit_stimulus_io.dwell(1e-6)
                qubit_stimulus_io.schedule_pulse(qubit_pi_0)
                qubit_stimulus_io.schedule_pulse(qubit_pi_90)
                qubit_stimulus_io.schedule_pulse(qubit_pi_180)
                qubit_stimulus_io.schedule_pulse(qubit_pi_270)
                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_0)
                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_90)
                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_180)
                qubit_stimulus_io.schedule_pulse(qubit_pi_over_2_270)


        # section 3 - function calls equivalent to "make_tables"
        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        # Section 4 - more setup 
        readout_resonator.load_windows()
        readout_stimulus_io.load_pulse(self.readout_pulse_name)

        qubit_stimulus_io.load_pulse(self.qubit_pi_name)


        qubit_stimulus_io.load_pulse(qubit_pi_0)
        qubit_stimulus_io.load_pulse(qubit_pi_90)
        qubit_stimulus_io.load_pulse(qubit_pi_180)
        qubit_stimulus_io.load_pulse(qubit_pi_270)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_0)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_90)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_180)
        qubit_stimulus_io.load_pulse(qubit_pi_over_2_270)

        qubit_pi_0_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_0))
        qubit_pi_90_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_90))
        qubit_pi_180_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_180))
        qubit_pi_270_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_270))
        qubit_pi_over_2_0_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_over_2_0))
        qubit_pi_over_2_90_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_over_2_90))
        qubit_pi_over_2_180_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_over_2_180))
        qubit_pi_over_2_270_command = self.acadia.waveform_dma_command(qubit_stimulus_io.get_waveform_memory(qubit_pi_over_2_270))
        
        # Section 5 - outer 'sweeper' for-loop(s)

        self.operation_names = ['X','Y','Xm','Ym','X2','Y2','X2m','Y2m']

        operation_commands = [qubit_pi_0_command, qubit_pi_90_command, qubit_pi_180_command, qubit_pi_270_command,
                              qubit_pi_over_2_0_command, qubit_pi_over_2_90_command, qubit_pi_over_2_180_command, qubit_pi_over_2_270_command]


        undo_commands = [self.acadia.waveform_dma_command(dummy),qubit_pi_0_command,
                            qubit_pi_over_2_90_command, qubit_pi_over_2_270_command, qubit_pi_over_2_180_command, qubit_pi_over_2_0_command]
        
        self.build_state_map_data()
        state_map = np.array(self.state_map_data).flatten()

        undo_ops = ["I", "X", "Y2", "Y2m", "X2m", "X2"]
        # undo_angles = [0, 1., 1/2, -1/2, -1/2, 1/2]
        # undo_phases = [0, 0, np.pi/2, np.pi/2, 0, 0]

    
        np.random.seed(self.my_rng_seed) # set RNG seed
        for i in range(self.iterations): # always sweep iterations on the outer loop!
            for j,num_pulses in enumerate(self.seq_lengths):


                num_pulses_cache[0] = num_pulses # load number of total pulses to play into cache

                # np.random.seed(self.my_rng_seed) # set RNG seed, FOR DEBUGGING
                my_random_sequence = np.random.randint(0,len(self.operation_names),int(np.max(self.seq_lengths))) # generate random pulse sequence

                # load all the commands into cache for N-1 pulses
                for k in range(num_pulses - 1):
                    pulse_command_cache[k] = operation_commands[my_random_sequence[k]] # load the pulse commands into the cache

                # figure out the undo command for the final slot
                my_state = 0
                for operation_idx in my_random_sequence[0:num_pulses-1]: # compute undo state
                    my_state = state_map[6 * operation_idx + my_state]

                pulse_command_cache[num_pulses-1] = undo_commands[my_state] # load into cache
                
                # my_gates_no_undo = [self.operation_names[my_random_sequence[k]] for k in range(num_pulses-1)]
                # my_undo_gate = undo_ops[my_state]
                # logger.info(f"gate string  {my_gates_no_undo} uses undo gate {my_undo_gate} for a total of {num_pulses} pulses")
                
                # capture data and put in the corresponding group
                self.acadia.run(minimum_delay=self.run_delay)
                wf = readout_capture_io.get_waveform_memory(self.capture_memory_name)
                self.data[f"points"].write(wf.array)

            # This allows for aborting part-way through experiment
            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve() # If you're all done with the experiment, do various disconnect/wrapping up things in the background


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
                                                                                        "A": {"value": p0[0], "bounds": (-1, 1)},
                                                                                        "of": {"value": 0.5, "bounds": (0, 1)}
                                                                                        })
        from uncertainties.umath import exp as uexp
        self.fitted_tau = self.RB_fit.ufloat_results['tau']
        self.gate_fidelity = uexp(-1/self.fitted_tau) 
        return completed_iterations



    @annotate_method(plot_name="RB decay thresholded", axs_shape=(1,1))
    def plot_RB_decay(self, axs=None):
        from acadia_qmsmt.plotting import prepare_plot_axes
        fig, axs = prepare_plot_axes(axs, axs_shape=(1,1), figsize=self.figsize)

        # axs.errorbar(self.seq_lengths, self.avg, yerr=self.std_errs, fmt='o',linestyle='', color='C0', markersize=3)

        self.RB_fit.plot(axs, oversample=5)
        axs.set_title(f"Qubit RB, F = {self.gate_fidelity:.5f}, tau = {self.fitted_tau:.4f} Gates")
        axs.set_xlabel("Number of pulses")

        axs.grid(True)
        axs.set_ylim(-0.02, 1.02)
        
        fig.tight_layout()
        return fig, axs
