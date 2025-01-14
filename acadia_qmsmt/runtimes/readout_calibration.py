import os
import time
from itertools import product
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

class ReadoutCalibrationRuntime(QMsmtRuntime):
    """
    Calibrates the frequency, amplitude, and filter shape for readout resonators
    connected to anharmonic oscillators.

    The key assumption is that for any level of the oscillator that we want to
    read out, we can make a sequence that will prepare that level in the oscillator
    more than any level above it. This allows us to incrementally "climb the ladder",
    and each time we introduce a new state that we've prepared, a new cluster will
    appear in the histogram of integrated signals. We can distinguish it from other 
    clusters because prior sequences will have prepared states in those clusters.

    The calibration algorithm is as follows:

        1. Execute sequences that prepare mostly level 0, 1, 2, etc. in the 
        oscillator and capture the resulting measurement signals. Denote 
        them sequence 0, 1, 2, etc.

        1. Sum over the trace times for every sequence's measurement, yielding 
        an array of integrated I/Q points.

        1. Aggregate all measurements (that is, temporarily discard knowledge 
        of which sequence prepared each measurement) and create a 2D histogram
        of the I/Q values.

        1. Fit the resulting histogram to a Gaussian mixture model (GMM) with a
        number of components equal to the number of sequences. This results in a 
        set of output classes, each corresponding to an average I/Q location and
        variance for a prepared state.

        1. For each sequence, use the trained GMM to predict the output classes
        of all measurements resulting from the sequence. Identify the most 
        commonly predicted class as the one corresponding to the state prepared 
        by the sequence, after ignoring all classes that have already been 
        assigned.

        1. Use the prediction results to create lists of time-domain traces for
        each system state, optionally rejecting measurements determined by the 
        user to be outliers or otherwise undesirable.
    """
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    readout_frequencies: NDArray[float]
    readout_amplitudes: NDArray[float]

    iterations: int
    num_clusters: int = 2
    saturation_pulse_fixed_length: float = 1e-3 - 100e-9
    saturation_pulse_ramp_time: float = 100e-9
    saturation_pulse_amplitude: float = 0.1
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    plot: bool = True
    figsize: tuple[int] = (11,4)
    histogram_bins_I: int = 50
    histogram_bins_Q: int = 50
    histogram_colormap: str = "hot"
    histogram_circle_alpha: float = 1.0
    histogram_circle_facecolor: str = "white"
    histogram_circle_fill: bool = False
    histogram_circle_edgecolor: str = "white"
    histogram_circle_linewidth: float = 1.0

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        # Create the qubit saturation waveform manually, as this is not a waveform
        # that the user will likely need to keep in the configuration file after this
        # measurement
        saturation_waveform = self.acadia.create_waveform_memory(
            qubit._stimulus.channel, 
            length=self.saturation_pulse_ramp_time, 
            fixed_length=self.saturation_pulse_fixed_length)

        self.data.add_group(f"traces", uniform=True)

        def sequence(a: Acadia):
            readout_resonator.prepare_cmacc(self.readout_window_name, output_type="input", output_last_only=False)

            with a.channel_synchronizer():
                a.schedule_waveform(saturation_waveform)
                a.barrier()
                readout_resonator.measure("readout", "readout_trace")

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()

        for i in range(self.iterations):
            for readout_frequency in self.readout_frequencies:
                readout_resonator.set_frequency(readout_frequency)
                for readout_amplitude in self.readout_amplitudes:
                    # On alternating runs, turn the pulse on and off
                    readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name, scale=readout_amplitude)

                    for factor in [0,1]:
                        saturation_waveform.set("hann", factor*self.saturation_pulse_amplitude)

                        self.acadia.run()
                        wf = readout_capture_io.get_waveform_memory("readout_trace")

                        self.data[f"traces"].write(wf.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        
        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
            from matplotlib.scale import get_scale_names
            from IPython.display import display
            from ipywidgets import Box, Layout, Dropdown, Checkbox, FloatText, Label, HTML, Button, Text

            self.fig = plt.figure(figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(hspace=0.3, left=0.1, bottom=0.25)

            gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1,2])
            self.ax_histogram = self.fig.add_subplot(gs[0])
            self.ax_histogram.set_xlabel("I [arb.]")
            self.ax_histogram.set_ylabel("Q [arb.]")

            gs_traces = GridSpecFromSubplotSpec(self.num_clusters, 1, subplot_spec=gs[1])
            self.ax_traces = [self.fig.add_subplot(spec) for spec in gs_traces]
            self.lines_re = [DynamicLine(ax, ".-", label="I") for ax in self.ax_traces]
            self.lines_im = [DynamicLine(ax, ".-", label="Q") for ax in self.ax_traces]
            for ax in self.ax_traces:
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Amplitude [arb.]")
                ax.grid()
                ax.legend(loc="upper right")

            # Interaction elements for assorted viewing options
            self.frequency_dropdown = Dropdown(options=self.readout_frequencies, description="Frequency:", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            self.amplitude_dropdown = Dropdown(options=self.readout_amplitudes, description="Amplitude:", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            self.histogram_view_dropdown = Dropdown(options=list(range(self.num_clusters)), description="View sequence:", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            self.histogram_scale_dropdown = Dropdown(options=["linear", "log"], description="Scale:", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            refresh_button = Button(description="Refresh Plots", layout=Layout(align_self="center"))
            refresh_button.on_click(lambda change: self.update_plot())

            global_view_settings_layout = Layout(display="flex", flex_flow="column", max_width="25%", align_items="stretch", border="solid")
            global_view_settings_box = Box(children=[
                HTML(value="<b>Global view settings</b>", layout=Layout(align_self="center")),
                self.frequency_dropdown, 
                self.amplitude_dropdown, 
                self.histogram_view_dropdown, 
                self.histogram_scale_dropdown,
                refresh_button], layout=global_view_settings_layout)
            
            # Settings for filter calculation
            self.matched_filter_trace1_dropdown = Dropdown(value=0, options=list(range(self.num_clusters)), description="Cluster 1:", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            self.matched_filter_trace2_dropdown = Dropdown(value=1, options=list(range(self.num_clusters)), description="Cluster 2:", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            self.matched_filter_name_input = Text(description="Name:", value="matched", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            self.matched_filter_save_button = Button(description="Save", layout=Layout(max_width="30%", align_self="center"))
            self.matched_filter_save_button.on_click(lambda args: self.save_matched_filter())
            matched_filter_layout = Layout(display="flex", flex_flow="column", align_items="stretch")
            matched_filter_box = Box(children=[
                Label(value="Matched linear filter", layout=Layout(max_width="95%", align_self="center"), style={"description_width": "initial"}),
                self.matched_filter_trace1_dropdown,
                self.matched_filter_trace2_dropdown,
                self.matched_filter_name_input, 
                self.matched_filter_save_button], layout=matched_filter_layout)
            
            # Settings for GMM training
            self.gmm_name_input = Text(description="Name:", value="gmm", layout=Layout(max_width="95%"), style={"description_width": "initial"})
            self.gmm_save_button = Button(description="Save", layout=Layout(max_width="30%", align_self="center"))
            self.gmm_save_button.on_click(lambda args: self.save_gmm())
            gmm_layout = Layout(display="flex", flex_flow="column", align_items="stretch")
            gmm_box = Box(children=[
                Label(value="Gaussian mixture model", layout=Layout(max_width="95%", align_self="center"), style={"description_width": "initial"}),
                self.gmm_name_input, 
                self.gmm_save_button], layout=gmm_layout)

            classifier_box_layout = Layout(display="flex", flex_flow="column", max_width=f"25%", align_items="stretch", border="solid")
            classifier_box = Box(children=[HTML(value="<b>Classifier training</b>", layout=Layout(align_self="center")), matched_filter_box, gmm_box], layout=classifier_box_layout)

            # Settings for fits
            self.circle_settings = []
            for idx in range(self.num_clusters):
                label = HTML(f"<b>Cluster {idx}</b> (n = 0/0)", layout=Layout(align_self="center"))
                show = Checkbox(value=True, description="Display circle", style={"description_width": "initial"}, layout=Layout(max_width="95%"))
                radius = FloatText(value=1, description="Inclusion radius [std. devs.]:", style={"description_width": "initial"}, layout=Layout(max_width="95%"))
                I_offset = FloatText(value=0, description="Center I offset:", style={"description_width": "initial"}, layout=Layout(max_width="95%"))
                Q_offset = FloatText(value=0, description="Center Q offset:", style={"description_width": "initial"}, layout=Layout(max_width="95%"))
                self.circle_settings.append(
                    {"label": label, 
                    "show": show, 
                    "radius": radius, 
                    "I_offset": I_offset, 
                    "Q_offset": Q_offset})


            sequence_settings_layout = Layout(display="flex", flex_flow="column", max_width=f"{round(50/self.num_clusters)}%", align_items="stretch", border="solid")
            sequence_settings_boxes = [Box(children=list(s.values()), layout=sequence_settings_layout) for s in self.circle_settings]

            all_settings_layout = Layout(display="flex", flex_flow="row", align_items="stretch")
            all_settings_box = Box(children=[global_view_settings_box, classifier_box, *sequence_settings_boxes], layout=all_settings_layout)

            display(all_settings_box)

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0
        self.time_axis = None
        
        # we'll fit gmms for each setting of frequency and amplitude
        self.gmms = np.empty((len(self.readout_frequencies), len(self.readout_amplitudes)), dtype=object)

        # we need to hold onto the plot objects so that we can clear them each time we update the plot
        self.cluster_circles = [] 
        self.histogram_plot = None

        # Using the offset and selection radii specified by UI elements 
        # along with the GMM fit parameters, we'll store the cluster properties
        # This is really only needed for separating traces by sequence, but will
        # also be good for plotting so that we can show the cluster selection circles
        self.cluster_centers = np.empty((len(self.readout_frequencies), len(self.readout_amplitudes), self.num_clusters, 2), dtype=np.float64)
        self.cluster_radii = np.empty((len(self.readout_frequencies), len(self.readout_amplitudes), self.num_clusters), dtype=np.float64)
        
        # Because different numbers of traces will be in different clusters, 
        # we need to make a ragged array
        # The elements of cluster_trace_indices are lists of ints, where 
        # each integer is a row index after selecting the traces for a given 
        # frequency and amplitude and reshaping the resulting array to have shape 
        # (completed_iterations*num_clusters, samples_per_trace, 2)
        self.cluster_trace_indices = np.empty((len(self.readout_frequencies), len(self.readout_amplitudes), self.num_clusters), dtype=object)
        for idx_f, idx_a, idx_p in product(range(len(self.readout_frequencies)), range(len(self.readout_amplitudes)), range(self.num_clusters)):
            self.cluster_trace_indices[idx_f, idx_a, idx_p] = []

        self.average_traces = None

    def update(self):
        records_per_iteration = self.num_clusters*len(self.readout_amplitudes)*len(self.readout_frequencies)

        # First make sure that we actually have new data to process
        if "traces" not in self.data or len(self.data["traces"]) < records_per_iteration:
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["traces"]) // records_per_iteration
        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)

        # Only continue processing data if we have at least one complete iteration
        if completed_iterations > 1:
            self.update_plot(lock=False) # The lock is already acquired by the event loop         
            
            if completed_iterations % 200 == 0:
                self.data.save(self.local_directory)

        self.iterations_previous = completed_iterations

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        # if self.plot:
        #     self.savefig(self.fig, close_canvas=False)

    def process_data(self):
        from sklearn.mixture import GaussianMixture
        
        records_per_iteration = self.num_clusters*len(self.readout_amplitudes)*len(self.readout_frequencies)
        completed_iterations = len(self.data["traces"]) // records_per_iteration
        valid_records = completed_iterations*self.num_clusters*len(self.readout_amplitudes)*len(self.readout_frequencies)
        data = self.data["traces"].records()[:valid_records, ...]

        # Get the collection of data and reshape it so that the axes index as: 
        # (iteration, frequency, amplitude, sequence, sample time, sample quadrature)
        samples_per_trace = data.shape[-2]
        self.data_reshaped = data.reshape(-1, len(self.readout_frequencies), len(self.readout_amplitudes), self.num_clusters, samples_per_trace, 2)

        # Sum over the trace length to get an integrated point
        # Don't sum over iterations
        self.data_iq_points = np.sum(self.data_reshaped, axis=4, keepdims=False)

        # Identify clusters by fitting to gaussian mixture model
        for idx_f, idx_a in product(range(len(self.readout_frequencies)), range(len(self.readout_amplitudes))):
            # First, slice the data to make life easier later
            data_iq_points_fa = self.data_iq_points[:, idx_f, idx_a, :, :] # (iterations, self.num_clusters, 2)
            traces_fa = self.data_reshaped[:, idx_f, idx_a, :, :, :] # (iterations, self.num_clusters, samples_per_trace, 2)

            # Fit all the data to a GMM
            gmm = GaussianMixture(n_components=self.num_clusters, covariance_type="spherical")
            gmm.fit(data_iq_points_fa.reshape(-1,2).astype("<f8"))
            self.gmms[idx_f, idx_a] = gmm

            # Now that we have a trained model, go back and predict the class of every 
            # point, but reshape it so that we can index the predictions by the
            # sequence that prepared the corresponding IQ point
            # We only need this so that we can relate the GMM class labels to the 
            # sequences that prepared them
            predictions = gmm.predict(data_iq_points_fa.reshape(-1,2)).reshape(-1, self.num_clusters)

            # Now that we've done the fit, because the GMM randomly assigns
            # labels to the output classes, we need to figure out which class
            # corresponds to each sequence
            # For each sequence, find the most populated cluster that 
            # hasn't already been identified 
            # Repeat for all sequences (there are always an equal number of 
            # sequences as clusters, so this will fully match everything)
            sequence_to_gmm_class = np.zeros(self.num_clusters, dtype=np.int32)
            gmm_class_to_sequence = np.zeros(self.num_clusters, dtype=np.int32)

            for idx_p in range(self.num_clusters):
                # np.bincount returns an array where the value at index i is 
                # the number of times that i appears in the input array
                # therefore, if we pass in the array of predictions, 
                # the indices of bincount correspond to cluster indices
                # and the value at a index j tells us how many points are in
                # cluster j
                bincount = np.bincount(predictions[:,idx_p], minlength=self.num_clusters)

                # For all the bins we've already found, mask their bin counts
                # so that we don't include them when trying to find the most 
                # populated bin
                bincount[sequence_to_gmm_class[:idx_p]] = -1

                most_populated_cluster = np.argmax(bincount)
                sequence_to_gmm_class[idx_p] = most_populated_cluster
                gmm_class_to_sequence[most_populated_cluster] = idx_p

            # Determine the locations and sizes of the inclusion zones for each cluster
            for idx_p in range(self.num_clusters):
                settings = self.circle_settings[idx_p]
                gmm_class = sequence_to_gmm_class[idx_p]
                cluster_radius = settings["radius"].value*np.sqrt(gmm.covariances_[gmm_class])
                cluster_center = gmm.means_[gmm_class,:] + np.array([settings["I_offset"].value, settings["Q_offset"].value])
                self.cluster_centers[idx_f, idx_a, idx_p, :] = cluster_center
                self.cluster_radii[idx_f, idx_a, idx_p] = cluster_radius

            # Find all the traces in the inclusion zone for each cluster
            # Use the predicted class for each IQ point, and if the point 
            # falls in the corresponding inclusion zone, associate it with
            # the sequence that created that class
            # Although we have to iterate over sequences/clusters, we don't actually
            # care which sequence the point came from, since we use the prediction
            # to associate it with a cluster
            for idx_p in range(self.num_clusters):
                self.cluster_trace_indices[idx_f, idx_a, idx_p].clear()

            for iteration,idx_p in product(range(completed_iterations), range(self.num_clusters)):
                predicted_sequence = gmm_class_to_sequence[predictions[iteration,idx_p]]
                point = self.data_iq_points[iteration, idx_f, idx_a, idx_p, :]
                trace = self.data_reshaped[iteration, idx_f, idx_a, idx_p, :, :]
                cluster_center = self.cluster_centers[idx_f, idx_a, predicted_sequence, :]
                cluster_radius = self.cluster_radii[idx_f, idx_a, predicted_sequence]
                point_distance = np.sqrt(np.sum((cluster_center - point)**2))
                if point_distance < cluster_radius:
                    # Add the indices to the corresponding array
                    self.cluster_trace_indices[idx_f, idx_a, predicted_sequence].append(iteration*self.num_clusters + idx_p)

            # Now that we have all the traces that integrate into a given cluster, 
            # compute and save the average traces for each cluster
            # We want this both for plotting and for computing the matched filter
            if self.average_traces is None:
                self.average_traces = np.empty((len(self.readout_frequencies), len(self.readout_amplitudes), self.num_clusters, samples_per_trace, 2), dtype=np.float64)

            traces_all = self.data_reshaped[:, idx_f, idx_a, :, :, :].reshape(completed_iterations*self.num_clusters, samples_per_trace, 2)
            for idx_p in range(self.num_clusters):
                trace_indices = np.array(self.cluster_trace_indices[idx_f, idx_a, idx_p], dtype=np.uint64)
                if len(trace_indices) > 0:
                    np.mean(traces_all[trace_indices, :, :], axis=0, out=self.average_traces[idx_f, idx_a, idx_p, :, :])
                        

    def update_plot(self, lock: bool = True, process: bool = True):
        """
        Update the view of the plot. We can choose to optionally acquire the update 
        lock so that we can call this in response to UI events, outside of the 
        runtime event loop
        """
        if self.plot:
            if lock and not self._update_lock.acquire(timeout=5):
                raise ValueError("Failed to acquire update lock when updating plot")

            if process:
                self.process_data()

            from matplotlib.patches import Circle
            from matplotlib.colors import Normalize, LogNorm

            # Index the selected data
            idx_f = self.frequency_dropdown.index
            idx_a = self.amplitude_dropdown.index 
            idx_p = self.histogram_view_dropdown.index

            # Create the histogram view
            if self.histogram_plot is not None:
                self.histogram_plot.remove()

            # Given all the points we collected, compute bin edges so that the histogram view 
            # will contain all the points regardless of which sequence we choose to view
            all_I_values = np.reshape(self.data_iq_points[:, idx_f, idx_a, :, 0], -1)
            histogram_I_edges = np.histogram_bin_edges(all_I_values, bins=self.histogram_bins_I)
            all_Q_values = np.reshape(self.data_iq_points[:, idx_f, idx_a, :, 1], -1)
            histogram_Q_edges = np.histogram_bin_edges(all_Q_values, bins=self.histogram_bins_Q)
            
            histogram, _, _ = np.histogram2d(
                self.data_iq_points[:, idx_f, idx_a, idx_p, 0], 
                self.data_iq_points[:, idx_f, idx_a, idx_p, 1], 
                bins=[histogram_I_edges, histogram_Q_edges],
                density=True)
            
            if self.histogram_scale_dropdown.value == "log":
                histogram_flat = histogram.reshape(-1)
                nonzero_indices = np.nonzero(histogram_flat)
                vmin = np.min(histogram_flat[nonzero_indices])
                norm = LogNorm(vmin=vmin, vmax=np.max(histogram), clip=True)
            else:
                norm = Normalize(vmin=np.min(histogram), vmax=np.max(histogram), clip=True)
            
            self.histogram_plot = self.ax_histogram.pcolormesh(
                histogram_I_edges, 
                histogram_Q_edges, 
                histogram.T,
                cmap=self.histogram_colormap,
                norm=norm)
            self.ax_histogram.set_xlim(histogram_I_edges[0], histogram_I_edges[-1])
            self.ax_histogram.set_ylim(histogram_Q_edges[0], histogram_Q_edges[-1])

            # Add circles for the GMM clusters
            for e in self.cluster_circles:
                e.remove()
            
            self.cluster_circles.clear()
            for i in range(self.num_clusters):
                # Adapted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py
                # Get the magnitude and direction of the principal components of the data
                # v, w = np.linalg.eigh(self.gmms[idx_f, idx_a].covariances_[i])
                # u = w[0] / np.linalg.norm(w[0])
                # angle = 180*(1 + np.arctan2(u[1], u[0])) / np.pi
                # v = 2.0 * np.sqrt(2.0) * np.sqrt(v)

                # Plot the cluster in a circle
                if self.circle_settings[i]["show"].value:
                    e = Circle(xy=self.cluster_centers[idx_f, idx_a, i, :], 
                                radius=self.cluster_radii[idx_f, idx_a, i],
                                fill=self.histogram_circle_fill, 
                                facecolor=self.histogram_circle_facecolor,
                                edgecolor=self.histogram_circle_edgecolor,
                                linewidth=self.histogram_circle_linewidth)
                    e.set_clip_box(self.ax_histogram.bbox)
                    e.set_alpha(self.histogram_circle_alpha)
                    self.cluster_circles.append(e)
                    self.ax_histogram.add_artist(e)

                    # Add text identifying the circle
                    ann = self.ax_histogram.annotate(
                        xy=self.cluster_centers[idx_f, idx_a, i, :],
                        text=f"{i}",
                        weight="bold", 
                        color="green", 
                        annotation_clip=True, 
                        fontsize=14, 
                        horizontalalignment="center",
                        verticalalignment="center")
                    self.cluster_circles.append(ann)
                
            # Update the average traces
            samples_per_trace = self.data_reshaped.shape[4]
            if self.time_axis is None:
                self.time_axis = np.linspace(
                    start=0, 
                    stop=self.io("readout_capture")._config["memories"]["readout_trace"]["length"], 
                    num=samples_per_trace, 
                    endpoint=False)

            traces_all = self.data_reshaped[:, idx_f, idx_a, :, :, :].reshape(-1, samples_per_trace, 2)
            for idx_p in range(self.num_clusters):
                trace_indices = np.array(self.cluster_trace_indices[idx_f, idx_a, idx_p], dtype=np.uint64)
                self.circle_settings[idx_p]["label"].value = f"<b>Cluster {idx_p}</b> (n = {len(trace_indices)}/{self.data_iq_points.shape[0]*self.num_clusters})"
                if len(trace_indices) > 0:
                    self.lines_re[idx_p].update(self.time_axis, self.average_traces[idx_f, idx_a, idx_p, :, 0], rescale_axis=False)
                    self.lines_im[idx_p].update(self.time_axis, self.average_traces[idx_f, idx_a, idx_p, :, 1], rescale_axis=False)
                    self.ax_traces[idx_p].relim()
                    self.ax_traces[idx_p].autoscale(tight=True)

            self.fig.canvas.draw_idle() 

            if lock:
                self._update_lock.release()

    def save_matched_filter(self, 
                            idx_frequency: int = None, 
                            idx_amplitude: int = None, 
                            idx_trace1: int = None, 
                            idx_trace2: int = None, 
                            name: str = None,
                            lock: bool = True):
        """
        Compute and save the matched filter that optimally separates two clusters.
        """

        if lock and not self._update_lock.acquire(timeout=5):
            raise ValueError("Failed to acquire update lock when saving matched filter")

        if idx_frequency is None:
            idx_frequency = self.frequency_dropdown.index

        if idx_amplitude is None:
            idx_amplitude = self.amplitude_dropdown.index
        
        if idx_trace1 is None:
            idx_trace1 = self.matched_filter_trace1_dropdown.value

        if idx_trace2 is None:
            trace2 = self.matched_filter_trace2_dropdown.value

        if name is None:
            name = self.matched_filter_name_input.value

        # View the trace data as complex 
        traces_complex = self.average_traces[idx_frequency, idx_amplitude, :, :, :].view(np.complex128)

        # Subtract the average traces and normalize the kernel
        # We want to normalize it by its maximum magnitude
        # TODO: take into account the scale of the boxcar used to collect the data
        #       for now we can ignore this because this will be a very small error compared to the noise
        kernel_trace = np.conjugate(traces_complex[idx_trace1, :] - traces_complex[idx_trace2, :])
        kernel_trace *= 0.9999 / np.max(np.abs(kernel_trace), keepdims=False)
        
        # Find the offset after transforming
        transformed_iq1 = np.dot(kernel_trace, traces_complex[idx_trace1, :])
        transformed_iq2 = np.dot(kernel_trace, traces_complex[idx_trace2, :])
        center = (transformed_iq1 + transformed_iq2) / 2.0
        center_rounded = (int(round(center.real)), int(round(center.imag)))

        # Save into a numpy file and update the configuration
        filename = f"{name}_{os.path.basename(self.local_directory)}.npy"
        if isinstance(self.readout_capture, str):
            filename = f"{self.readout_capture}_{filename}"

        np.save(file=filename, arr=kernel_trace)

        from acadia_qmsmt.helpers.yaml_editor import update_yaml
        window_info = {"data": filename, 
                        "stimulus_waveform_name": self.readout_stimulus_waveform_name,
                        "offset": center_rounded}
        self.update_ioconfig("readout_capture", f"windows.{name}", window_info)

        if lock:
            self._update_lock.release()

    def save_gmm(self, 
                idx_frequency: int = None, 
                idx_amplitude: int = None, 
                name: int = None,
                lock: bool = True):
        """
        Train and save the parameters of a Gaussian mixture model.
        """
        if lock and not self._update_lock.acquire(timeout=5):
            raise ValueError("Failed to acquire update lock when saving GMM")

        if idx_frequency is None:
            idx_frequency = self.frequency_dropdown.index

        if idx_amplitude is None:
            idx_amplitude = self.amplitude_dropdown.index

        if name is None:
            name = self.gmm_name_input.value

        filename = f"{name}_{os.path.basename(self.local_directory)}.npz"
        if isinstance(self.readout_capture, str):
            filename = f"{self.readout_capture}_{filename}"

        gmm = self.gmms[idx_frequency, idx_amplitude]

        if lock:
            self._update_lock.release()

