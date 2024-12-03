from dataclasses import dataclass
from typing import Tuple, Literal
import numpy as np
from numpy.typing import NDArray
from acadia.runtime import Runtime
from acadia import DataManager

from auto_config import AutoConfigMixin
from auto_config import FILE as config_helper_file

@dataclass
class ActiveCoolingRuntime(AutoConfigMixin, Runtime):
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
        channel_objs = self.obtain_channels(acadia, **channel_configs)

        # Allocate the waveform memories that we'll need
        q_rotation_pi = self.allocate_waveform_mem(acadia, "q_stimulus", "q_rotation")
        q_rotation_pio2 = self.allocate_waveform_mem(acadia, "q_stimulus", "q_rotation")
        ro_drive = self.allocate_waveform_mem(acadia, "ro_stimulus", "ro_drive")
        ro_pts_0 = self.allocate_waveform_mem(acadia, "ro_capture", "ro_pts")
        ro_pts_1 = self.allocate_waveform_mem(acadia, "ro_capture", "ro_pts")
        ro_pts_2 = self.allocate_waveform_mem(acadia, "ro_capture", "ro_pts")

        # Create a blank waveform between qubit drive and readout drive
        q_blank_wf = self.blank_waveform_generator(acadia, "q_stimulus")(40e-9)

        # Create blank waveform for delay between capture and stimulus
        capture_delay = self.ro_capture["capture_delay"]
        if capture_delay < 0: # stimulus will be delayed by -capture_delay compare to capture
            blank_wf = self.blank_waveform_generator(acadia, "ro_stimulus")(-capture_delay)
        elif capture_delay > 0: # capture will be delayed by capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture", "ro_demod")(capture_delay)

        # get the kernel and IQ offsets fom the config dict
        kernel_wf = self.ro_capture.get("kernel_wf", [0.1])
        cmacc_offset = self.ro_capture.get("cmacc_offset", 0)

        # Create the record groups for saving captured data
        self.data.add_group(f"pts_0", uniform=True)
        self.data.add_group(f"pts_1", uniform=True)
        self.data.add_group(f"pts_2", uniform=True)

        # get g state quadrant from config 
        g_quadrant = self.ro_capture["state_quadrants"][0]

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            ## Step 1: Prepare g+e and do first msmt
            capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel_wf,
                                                            reset_fifo=True, accumulator_done=False)

            a.cmacc_load(capture_stream, cmacc_offset)

            with a.channel_synchronizer(): 
                a.schedule_waveform(q_rotation_pio2)
                a.schedule_waveform(q_blank_wf) 
                a.barrier()
                if capture_delay != 0:
                    a.schedule_waveform(blank_wf)
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_pts_0)

            # put the first msmt result in a register
            reg = a.sequencer().Register()
            reg.load(a.cmacc_get_quadrant(capture_stream))

            ## Step 2: Measure + conditional flip, until we get the ground state
            # todo: try getting the number of msmts in this loop using a counter register
            with a.sequencer().repeat_until(reg==getattr(a, f"CMACC_QUADRANT_{g_quadrant}")):
                capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel,
                                                            reset_fifo=True, accumulator_done=False)

                a.cmacc_load(capture_stream, cmacc_offset) # todo: extra bias can be added here.

                with a.channel_synchronizer():
                    a.schedule_waveform(q_rotation_pi)
                    a.schedule_waveform(q_blank_wf)
                    a.barrier()
                    if capture_delay != 0:
                        a.schedule_waveform(blank_wf)
                    a.schedule_waveform(ro_drive)
                    # this will keep overwriting the re_pts1 waveform
                    # the final value will always be in the g state quadrant since that's the condition for exiting the loop
                    a.stream(capture_stream, ro_pts_1)

                # `cmacc_get_quadrant` will wait until cmacc is done   
                reg.load(a.cmacc_get_quadrant(capture_stream))


            ## Step 3: Do a final msmt
            capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel,
                                                            reset_fifo=False, accumulator_done=False)
            a.cmacc_load(capture_stream, cmacc_offset)

            with a.channel_synchronizer():
                if capture_delay != 0:
                    a.schedule_waveform(blank_wf)
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_pts_2)

            return kernel

        # Compile the sequence
        kernel = acadia.compile(sequence)
        # Attach to the hardware
        acadia.attach()
        # Configure channel analog parameters
        self.auto_config_ncos(acadia, **channel_configs)
        # Assemble and load the program
        acadia.assemble()
        acadia.load()

        # Set the amplitude of the boxcar integration kernel
        kernel.set(kernel_wf)
        # set waveform for ro and qubit drive
        ro_drive.set(**self.ro_stimulus["signals"]["readout"])
        q_rotation_pi.set(**self.q_stimulus["signals"]["pi_pulse"])
        q_rotation_pio2.set(**self.q_stimulus["signals"]["pi_2_pulse"])

        # average iterations for preparing qubit in g and e
        for i in range(self.iterations):
            # capture data and put in the corresponding group
            acadia.run()
            self.data[f"pts_0"].write(ro_pts_0.array)
            self.data[f"pts_1"].write(ro_pts_1.array)
            self.data[f"pts_2"].write(ro_pts_2.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (3, 5) if self.figsize is None else self.figsize

        if self.plot:
            import matplotlib.pyplot as plt
            self.fig, self.axs = plt.subplots(3, 1, figsize=self.figsize)

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):
        import numpy as np
        import matplotlib.pyplot as plt

        # First make sure that we actually have new data to process
        if "pts_2" not in self.data or len(self.data["pts_2"]) < 2:
            return

        # Update the progress bar based on the number of iterations that have been completed
        completed_iterations = len(self.data["pts_2"])
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations

        if self.plot:
            data = [self.data[f"pts_{s_}"].records().squeeze() for s_ in ["0", "1", "2"]]
            for i in range(3):
                self.axs[i].hist2d(data[i][:, 0], data[i][:, 1], bins=51, cmap="hot")
                self.axs[i].relim()
                self.axs[i].autoscale(tight=True)
                self.axs[i].set_aspect(1)
                self.axs[i].tick_params(axis='both', which='major', labelsize=5)
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()

        # Save the data
        self.data.save(self.local_directory)

    def finalize(self):
        super().finalize()
        self.progress_bar.close()
        if self.plot:
            self.savefig(self.fig)
        self.post_process()
        

    def post_process(self):
        """ Calculate the e percentage for the prepared states, assuming the 
            cmacc offset and state quadrants has been set properly in the config

        """
        from acadia_qmsmt.analysis import population_in_quadrant
        results = self.parse_data(self.data)

        g_quadrant = self.ro_capture["state_quadrants"][0]

        g_pcts = []
        fig, ax = plt.subplots(1, 1)
        for i, pts in enumerate(results):
            pct = population_in_quadrant(pts, g_quadrant)
            g_pcts.append(pct)
            print(f"g population, msmt_{i}:", pct)
            ax.plot(pts.real, pts.imag, ".", ms=0.5, label=f"msmt_{i}")
        ax.legend()
        ax.set_aspect(1)
        
        return g_pcts


    @staticmethod
    def parse_data(data_manager: DataManager) -> Tuple[NDArray[np.complex128]]:
        """Parse the acquired data in data_manager to an easy-to-process format

        :param data_manager: data manager object for data acquired using this runtime
        """
        dm = data_manager
        results = []
        for i in range(3):
            results.append(dm[f"pts_{i}"].records().astype(float).view(complex).squeeze())
        return results


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from acadia_qmsmt.measurements import load_config

    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    config_dict = load_config()

    plot = True
    iterations = 50000

    rt = ActiveCoolingRuntime(**config_dict, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout_pts_active_cooling_demo", files=[rt.FILE, config_helper_file])
    rt.display()

    # some ad hoc processing
    # rt._event_loop.join()
    # rt.fig
    
    # print(rt.data[f"pts_2"].records().shape)


