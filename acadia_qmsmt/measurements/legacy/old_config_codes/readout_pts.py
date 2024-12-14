from dataclasses import dataclass
from typing import Tuple, Literal
import numpy as np
from numpy.typing import NDArray
from acadia.runtime import Runtime
from acadia import DataManager

from auto_config import AutoConfigMixin

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

    def __post_init__(self):
        self.FILES = [__file__, super().FILE]

    def main(self):
        from acadia import Acadia, DataManager
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
        q_rotation = self.allocate_waveform_mem(acadia, "q_stimulus", "q_rotation")
        ro_drive = self.allocate_waveform_mem(acadia, "ro_stimulus", "ro_drive")
        ro_pts = self.allocate_waveform_mem(acadia, "ro_capture", "ro_pts")

        # Create a blank waveform between qubit drive and readout drive
        q_blank_wf = self.blank_waveform_generator(acadia, "q_stimulus")(40e-9)

        # Create blank waveform for delay between capture and stimulus
        capture_delay = self.ro_capture["capture_delay"]
        if capture_delay < 0: # stimulus will be delayed by -capture_delay compare to capture
            blank_wf = self.blank_waveform_generator(acadia, "ro_stimulus")(-capture_delay)
        elif capture_delay > 0: # capture will be delayed by capture_delay compare to stimulus
            blank_wf = self.blank_waveform_generator(acadia, "ro_capture", "ro_pts")(capture_delay)

        # get the kernel and IQ offsets fom the config dict
        kernel_wf = self.ro_capture.get("kernel_wf", [0.1])
        cmacc_offset = self.ro_capture.get("cmacc_offset", 0)

        # Create the record groups for saving captured data
        self.data.add_group(f"pts_g", uniform=True)
        self.data.add_group(f"pts_e", uniform=True)

        # Create a sequence for the sequencer to generate the pulse and capture it
        def sequence(a: Acadia):
            capture_stream, kernel = a.configure_cmacc(channel_objs["ro_capture"], kernel=kernel_wf)

            a.cmacc_load(capture_stream, cmacc_offset) # fixme: currently this only accepts positive integers, should take complex number instead

            with a.channel_synchronizer():
                a.schedule_waveform(q_rotation)
                a.schedule_waveform(q_blank_wf)
                a.barrier()
                if capture_delay != 0:
                    a.schedule_waveform(blank_wf)
                a.schedule_waveform(ro_drive)
                a.stream(capture_stream, ro_pts)

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
        # set waveform for ro drive
        ro_drive.set(**self.ro_stimulus["signals"]["readout"])

        # average iterations for preparing qubit in g and e
        pi_signal = self.q_stimulus["signals"]["pi_pulse"]
        pi_amp = pi_signal["scale"]
        for i in range(self.iterations):
            for state_ in ["g", "e"]:
                # set the pulse amplitude
                pi_signal["scale"] = 0 if state_ == "g" else pi_amp
                q_rotation.set(**pi_signal) # takes around 1ms

                # capture data and put in the corresponding group
                acadia.run()
                self.data[f"pts_{state_}"].write(ro_pts.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        # Set the matplotlib backend to one which we can actually update
        self.figsize = (3, 5) if self.figsize is None else self.figsize

        if self.plot:
            # from acadia.processing import DynamicReadoutHistogram
            # todo: need to learn how this works... Currently doing a simple hist2d
            
            import matplotlib.pyplot as plt
            self.fig, self.axs = plt.subplots(2, 1, figsize=self.figsize)

        from tqdm.notebook import tqdm
        self.progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.previous_completed_iterations = 0

    def update(self):

        # First make sure that we actually have new data to process
        if "pts_e" not in self.data or len(self.data["pts_e"]) < 2 :
            return

        # Update the progress bar based on the number of iterations that have been completed
        completed_iterations = len(self.data["pts_e"])
        self.progress_bar.update(completed_iterations - self.previous_completed_iterations)
        self.previous_completed_iterations = completed_iterations

        if self.plot:
            data = [self.data[f"pts_{s_}"].records().squeeze() for s_ in ["g", "e"]]
            for i in range(2):
                self.axs[i].clear()
                self.axs[i].hist2d(data[i][:, 0], data[i][:, 1], bins=51, cmap="hot")
                self.axs[i].relim()
                self.axs[i].autoscale(tight=True)
                self.axs[i].set_aspect(1)
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

    def post_process(self, i_threshold: int = 0, q_threshold: int = 0,
                     e_quadrant: Literal[1,2,3,4] = None):
        """ Calculate the e percentage for the prepared states

        :param i_threshold: I position of the state separation line. 
            Default to 0 assuming cmacc offset has been applied properly
        :param q_threshold: Q position of the state separation line. 
            Default to 0 assuming cmacc offset has been applied properly
        :param e_quadrant: Quadrant of the e state. Default to
            the quadrant saved in the yaml file.
        """
        from acadia_qmsmt.analysis import population_in_quadrant
        g_pts, e_pts = self.parse_data(self.data)

        if e_quadrant is None:
            e_quadrant = self.ro_capture.get("state_quadrants", [None, None])[1]
            if e_quadrant is None:
                print("e-state quadrant not provided.")
                return [None, None]

        e_pcts = []
        for state, pts in zip(("g", "e"), (g_pts, e_pts)):
            pct = population_in_quadrant(pts, e_quadrant, i_threshold, q_threshold)
            e_pcts.append(pct)
            print(f"e population, prepare {state}:", pct)

        return e_pcts


    @staticmethod
    def parse_data(data_manager: DataManager) -> Tuple[NDArray[np.complex128]]:
        """Parse the acquired data in data_manager to an easy-to-process format

        :param data_manager: data manager object for data acquired using this runtime
        """
        dm = data_manager
        g_pts = dm["pts_g"].records().astype(float).view(complex).squeeze()
        e_pts = dm["pts_e"].records().astype(float).view(complex).squeeze()
        return g_pts, e_pts


if __name__ == "__main__":

    # import matplotlib
    # matplotlib.use("module://ipympl")
    import matplotlib.pyplot as plt
    from IPython.core.getipython import get_ipython
    get_ipython().run_line_magic("matplotlib", "widget")

    import numpy as np
    from acadia_qmsmt.measurements import load_config
    config_dict = load_config()

    plot = True
    iterations = 5000

    rt = ReadoutPtsRuntime(**config_dict, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout_pts", files=rt.FILES)
    rt.display()

    # some ad hoc processing
    # rt._event_loop.join()
    # rt.fig

    # fig, ax = plt.subplots(1, 1)
    # for i, s_ in enumerate(["g", "e"]):
    #     data = rt.data[f"pts_{s_}"].records().squeeze()
    #     ax.plot(data[:, 0], data[:, 1], ".", ms=0.5)
    #     ax.set_aspect(1)
