from dataclasses import dataclass
from acadia.runtime import Runtime

from auto_config import AutoConfigMixin
from auto_config import FILE as config_helper_file


@dataclass
class ReadoutPtsRuntime(AutoConfigMixin, Runtime):
    """
    A :class:`Runtime` subclass for sending a signal out of a DAC and capturing
    it on an ADC.
    """

    q_stimulus: dict
    ro_stimulus: dict
    ro_capture: dict

    iterations: int
    plot: bool = True
    figsize: tuple[int] = None

    FILE = __file__

    def main(self):
        from acadia import Acadia, DataManager
        import numpy as np
        import logging

        logger = logging.getLogger("acadia")

        channel_configs = {"q_stimulus": self.q_stimulus,
                           "ro_stimulus": self.ro_stimulus,
                           "ro_capture": self.ro_capture
                           }

        # Create an acadia object and grab a couple of its channels
        acadia = Acadia()
        self.auto_config_channels(acadia, **channel_configs)

        # Allocate the waveform memories that we'll need
        self.auto_config_waveform_mems(acadia, **channel_configs)
        ro_drive = self.channel_waveforms["ro_stimulus"]["ro_drive"]

        # need two copies for changing qubit pulse within one sequence
        q_rotation_pi = acadia.create_waveform(self.channel_objs["q_stimulus"],  **self.q_stimulus["waveforms"]["q_rotation"])
        q_rotation_pi2 = acadia.create_waveform(self.channel_objs["q_stimulus"],  **self.q_stimulus["waveforms"]["q_rotation"])

        ro_demod0 = acadia.create_waveform(self.channel_objs["ro_capture"], length=2.4e-6, decimation=0, region="plddr")
        ro_demod1 = acadia.create_waveform(self.channel_objs["ro_capture"], length=2.4e-6, decimation=0, region="plddr")
        ro_demod2 = acadia.create_waveform(self.channel_objs["ro_capture"], length=2.4e-6, decimation=0, region="plddr")

        # Create a blank waveform between qubit drive and readout drive
        q_blank_wf = self.blank_waveform_generator(acadia, "q_stimulus")(40e-9)

        # Create blank waveform for delay between capture and stimulus
        capture_delay = self.ro_capture["capture_delay"]
        if capture_delay < 0:  # capture will be advanced by -capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_stimulus")(-capture_delay)
        elif capture_delay > 0:  # capture will be delayed by capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture")(capture_delay)
        
        ro_blank_gen = self.blank_waveform_generator(acadia, "ro_stimulus") # readout blank waveform generator

        kernel_wf = self.ro_capture.get("kernel_wf", 0.1)
        if type(kernel_wf) == float:  # constant value kernel
            kernel_wf = np.float64(kernel_wf)
            kernel_cmacc = None
        elif type(kernel_wf) == np.ndarray:
            kernel_cmacc = kernel_wf
        
        kernel_offset = self.ro_capture.get("kernel_offset", 0)

        # Create the record groups for saving captured data
        self.data.add_group(f"pts_0", uniform=True)
        self.data.add_group(f"pts_1", uniform=True)
        self.data.add_group(f"pts_2", uniform=True)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            
            ## prepare and first msmt
            capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel_cmacc,
                                                            reset_fifo=True, accumulator_done=False)

            a.cmacc_load(capture_stream, kernel_offset)

            with a.channel_synchronizer(): 
                a.schedule_waveform(q_rotation_pi2)
                a.schedule_waveform(q_blank_wf) 
                a.barrier()
                if capture_delay != 0:
                    a.schedule_waveform(blank_wf)
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_demod0)


            reg = a.sequencer().Register()
            reg.load(a.cmacc_get_quadrant(capture_stream))

            ## measure until we get the ground state
            # todo: trying getting the number of msmts in this loop using a counter register
            with a.sequencer().repeat_until(reg == a.CMACC_QUADRANT_3):
                capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel,
                                                            reset_fifo=True, accumulator_done=False)

                a.cmacc_load(capture_stream, kernel_offset)

                with a.channel_synchronizer():
                    a.schedule_waveform(q_rotation_pi)
                    a.schedule_waveform(q_blank_wf) 
                    a.barrier()
                    if capture_delay != 0:
                        a.schedule_waveform(blank_wf)
                    a.schedule_waveform(ro_drive)
                    # this will keep overwriting the re_demo1 waveform
                    # the final value will always be in the 3rd quadrant since that's the condition for exiting the loop
                    a.stream(capture_stream, ro_demod1)

                # get quadrant will wait until cmacc is done   
                reg.load(a.cmacc_get_quadrant(capture_stream))


            ## do a final msmt
            capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel,
                                                            reset_fifo=False, accumulator_done=False)
            a.cmacc_load(capture_stream, kernel_offset) 

            with a.channel_synchronizer():
                if capture_delay != 0:
                    a.schedule_waveform(blank_wf)
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_demod2)
                
            return kernel

        # Compile the sequence
        kernel = acadia.compile(sequence)
# 
        # Attach to the hardware
        acadia.attach()

        # Configure channel analog parameters
        self.auto_config_ncos(acadia, **channel_configs)

        # Set the amplitude of the boxcar integration kernel
        kernel.set(kernel_wf)

        # Assemble and load the program
        acadia.assemble()
        acadia.load()
        


        # set waveform for ro and qubit drive
        ro_drive.set(**self.ro_stimulus["signals"]["readout"])
        q_rotation_pi.set(**self.q_stimulus["signals"]["pi_pulse"])
        q_rotation_pi2.set(**self.q_stimulus["signals"]["pi_2_pulse"])

        # average iterations for preparing qubit in g and e
        for i in range(self.iterations):
            # capture data and put in the corresponding group
            acadia.run()
            self.data[f"pts_0"].write(ro_demod0.array)
            self.data[f"pts_1"].write(ro_demod1.array)
            self.data[f"pts_2"].write(ro_demod2.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (3, 5) if self.figsize is None else self.figsize

        if self.plot:
            from acadia.processing import DynamicReadoutHistogram
            import matplotlib.pyplot as plt

            self.fig, self.axs = plt.subplots(3, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            # self.hist = DynamicReadoutHistogram(self.ax) # todo: need to learn how this works...

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # First make sure that we actually have new data to process
        if "pts_1" not in self.data or len(self.data["pts_1"]) < 2:
            return

        # Update the progress bar based on the number of iterations that have been completed
        completed_iterations = len(self.data["pts_1"])
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations

        if self.plot:
            data = [self.data[f"pts_{s_}"].records().squeeze() for s_ in ["0", "1", "2"]]
            for i in range(3):
                # self.hist.update(data)
                self.axs[i].hist2d(data[i][:, 0], data[i][:, 1], bins=51, cmap="hot")
                self.axs[i].relim()
                self.axs[i].autoscale(tight=True)
                self.fig.tight_layout()
                self.fig.canvas.draw_idle()
                self.axs[i].set_aspect(1)
                self.axs[i].set_aspect(1)
                self.axs[i].tick_params(axis='both', which='major', labelsize=5)

        # Save the data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        self.progress_bar.close()
        if self.plot:
            self.savefig(self.fig)
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from linc_rfsoc.measurements import load_config
    from linc_rfsoc.analysis.generate_readout_kernel import ReadoutKernelGenerator, load_kernel

    from IPython.core.getipython import get_ipython

    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config()

    plot = True
    iterations = 50000

    rt = ReadoutPtsRuntime(**config_dict, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout_pts_rtl_cooling", files=[rt.FILE, config_helper_file])
    rt.display()

    # some ad hoc processing
    rt._event_loop.join()
    # rt.fig
    
    print(rt.data[f"pts_2"].records().shape)

    fig, ax = plt.subplots(1, 1)
    for i, s_ in enumerate(["0", "1", "2"]):
        data = rt.data[f"pts_{s_}"].records().squeeze()
        ax.plot(data[:, 0], data[:, 1], ".", ms=0.5)
        ax.set_aspect(1)


    def g_pct(data_group:str):
        data = rt.data[data_group].records().squeeze()
        n_g = len(np.where(data[:, 0]<config_dict["ro_capture"]["kernel_offset"])[0])
        n_tot = len(data)
        return n_g/n_tot

    for i in range(3):
        print(f"msmt {i} g_pct: {g_pct(f'pts_{i}')}" )
    # fig, axs = plt.subplots(3, 1)
    # for i, s_ in enumerate(["0", "1", "2"]):
    #     data = rt.data[f"pts_{s_}"].records().squeeze()
    #     axs[i].hist2d(data[:, 0], data[:, 1], bins=51)
    #     axs[i].set_aspect(1)