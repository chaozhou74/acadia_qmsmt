from typing import Union
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt

from acadia import Acadia, DataManager, Runtime, WaveformMemory
from acadia_qmsmt import QMsmtRuntime, MeasurableResonator, Qubit, IOConfig

def decay(f, A, w, B):
    return A * np.exp(-w * np.pi * f) + B

def lorentzian(x, A, x0, w):
    return A / ((x-x0)**2 + (w/2))

def quadratic(n, a, b, c):
    return a*n**2 + b*n + c

class DisplacedCavityQubitSpectroscopyRuntime(QMsmtRuntime):
    """
    This Runtime allows the simultaneous calibration of displacement 
    amplitudes as well as measurements of dispersive shifts.

    Note that there must be at least one cavity pulse amplitude of zero. This curve is used
    as the basis for a number of following analysis routines.
    """

    cavity_stimulus: IOConfig
    qubit_stimulus: IOConfig
    readout_stimulus: IOConfig
    readout_capture: IOConfig

    qubit_frequencies: Union[list, np.ndarray]
    cavity_pulse_amplitudes: Union[list, np.ndarray]

    iterations: int
    run_delay: int
    cavity_pulse: dict[str,float] = {"length": 100e-9}
    qubit_saturation_pulse: dict[str,float] = {"length": 100e-9, "stretch_length": 5e-6}
    readout_window_name: str = None
    readout_stimulus_waveform_name: str = None
    plot: bool = True
    figsize: tuple[int] = None
    yaml_path: str = None

    def main(self):
        import logging
        logger = logging.getLogger("acadia")

        cavity_stimulus_io = self.io("cavity_stimulus")
        qubit_stimulus_io = self.io("qubit_stimulus")
        readout_stimulus_io = self.io("readout_stimulus")
        readout_capture_io = self.io("readout_capture")

        readout_resonator = MeasurableResonator(readout_stimulus_io, readout_capture_io)
        qubit = Qubit(qubit_stimulus_io)

        cavity_pulse = self.acadia.create_waveform_memory(
            cavity_stimulus_io.channel, 
            length=self.cavity_pulse.get("length", 0.0)
        )

        qubit_saturation_pulse = self.acadia.create_waveform_memory(
            qubit_stimulus_io.channel,
            length=self.qubit_saturation_pulse.get("length", 0.0)
        )

        self.data.add_group(f"points", uniform=True)

        def sequence(a: Acadia):
            with a.channel_synchronizer():
                a.schedule_waveform(cavity_pulse, stretch_length=self.cavity_pulse.get("stretch_length", 0.0))
                a.barrier()
                qubit.pulse(qubit_saturation_pulse, stretch_length=self.qubit_saturation_pulse.get("stretch_length", 0.0))
                a.barrier()
                readout_resonator.measure("readout", "readout_accumulated", self.readout_window_name)

        self.acadia.compile(sequence)
        self.acadia.attach()
        self.configure_channels()
        self.acadia.assemble()
        self.acadia.load()

        readout_resonator.load_windows()
        readout_stimulus_io.load_waveform("readout", self.readout_stimulus_waveform_name)
        qubit_stimulus_io.load_waveform(qubit_saturation_pulse, 
            waveform={"data": self.qubit_saturation_pulse.get("data", "hann")}, 
            scale=self.qubit_saturation_pulse.get("scale", 0.9999))
        
        for i in range(self.iterations):
            for cavity_pulse_amplitude in self.cavity_pulse_amplitudes:
                cavity_stimulus_io.load_waveform(cavity_pulse, {"data": "hann"}, scale=cavity_pulse_amplitude)
                for qubit_frequency in self.qubit_frequencies:
                    qubit.set_frequency(qubit_frequency)

                    self.acadia.run(minimum_delay=self.run_delay)
                    wf = readout_capture_io.get_waveform_memory("readout_accumulated")
                    self.data[f"points"].write(wf.array)

            if self.data.serve() == DataManager.serve_hangup():
                self.data.disconnect()
                return

        self.final_serve()

    def initialize(self):
        
        if self.plot:
            from acadia.processing import DynamicLine
            import matplotlib.pyplot as plt
            from IPython.display import display
            from ipywidgets import Label, Layout, Box

            self.figsize = (8, 3) if self.figsize is None else self.figsize
            self.fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.25, bottom=0.25)

            self.line_pops = [DynamicLine(ax, ".-", label=f"{a}") for a in self.cavity_pulse_amplitudes]
            ax.set_xlabel("Qubit Frequency [Hz]")
            ax.set_ylabel("Qubit Population [FS]")
            ax.set_xlim(self.qubit_frequencies[0], self.qubit_frequencies[-1])
            ax.set_ylim(-1.1, 1.1)
            ax.grid()
            ax.legend(loc="lower left")

            from tqdm.notebook import tqdm

        else:
            from tqdm import tqdm

        self.iterations_progress_bar = tqdm(desc="Iterations", dynamic_ncols=True, total=self.iterations)
        self.iterations_previous = 0

    def update(self):
        # First make sure that we actually have new data to process
        if "points" not in self.data or len(self.data["points"]) < len(self.qubit_frequencies)*len(self.cavity_pulse_amplitudes):
            return

        # Update the progress bar based on the number of iterations
        completed_iterations = len(self.data["points"]) // (len(self.qubit_frequencies)*len(self.cavity_pulse_amplitudes))
        self.iterations_progress_bar.update(completed_iterations - self.iterations_previous)
        if completed_iterations == 0:
            return

        self.process_data()

        if self.plot:
            for idx_amp,_ in enumerate(self.cavity_pulse_amplitudes):
                self.line_pops[idx_amp].update(self.qubit_frequencies, self.avg[idx_amp,:], rescale_axis=False)
            self.fig.canvas.draw_idle() 

        self.data.save(self.local_directory)
        self.iterations_previous = completed_iterations

    def process_data(self):
        completed_iterations = len(self.data["points"]) // (len(self.qubit_frequencies)*len(self.cavity_pulse_amplitudes))
        valid_points = completed_iterations*len(self.qubit_frequencies)*len(self.cavity_pulse_amplitudes)
        data = self.data["points"].records()[:valid_points, ...]
        data = data.reshape(completed_iterations, len(self.cavity_pulse_amplitudes), len(self.qubit_frequencies), 2)

        # Threshold the data according to the I quadrature
        shots = np.sign(data[:,:,:,0], dtype=np.int32)
        self.avg = np.mean(shots, axis=0)

    def finalize(self):
        super().finalize()
        self.iterations_progress_bar.close()
        if self.plot:
            self.savefig(self.fig)

    def analyze_dispersion(self, assigned_peak_numbers=None, peak_min_width=5, peak_max_width=20, peak_noise_perc=30, peak_min_length=5):
        qubit_frequencies_MHz = self.qubit_frequencies*1e-6

        self.dispersion_analysis = {}

        # First, find the noise floor of each inidividual trace and subtract it off
        noise_floors = np.zeros_like(self.avg)
        for i in range(self.avg.shape[0]):
            trace = self.avg[i,:]
            noise_floors[i,:] = np.mean(trace[trace > np.percentile(trace, 50)])
        self.dispersion_analysis["noise_floors"] = noise_floors

        traces = noise_floors - self.avg
        zero_trace = traces[0,:]
        self.dispersion_analysis["traces"] = traces

        # Now lay all the traces "on top" of one another by taking the max across traces
        combined = np.max(traces, axis=0)
        self.dispersion_analysis["combined_trace"] = combined

        # Count the peaks and get an initial guess for their locations
        # this is only a guess because the scipy peak-finding functions always return
        # an index in the input array, but the true peak location will be somewhere in between
        peak_indices = find_peaks_cwt(
            combined, 
            np.arange(peak_min_width,peak_max_width), 
            noise_perc=peak_noise_perc, 
            min_length=peak_min_length)
        self.dispersion_analysis["peak_guess_indices"] = peak_indices

        # Take the fourier transform of the 0-displacement trace to get a good guess of the peak width
        # The fourier transform is F[L(x)](f) = exp(2 pi i f x0 - w pi |f|)
        # The magnitude of this is exp(-w pi |f|), so we can fit to an exponential
        fft = np.fft.rfft(zero_trace)
        frequency_spacing = qubit_frequencies_MHz[1] - qubit_frequencies_MHz[0]
        fftfreq = np.fft.rfftfreq(len(zero_trace), d=frequency_spacing)
        fftmag = np.abs(fft)
        fit_guess = (fftmag[1], 0, fftmag[-1])
        bounds = ([1e-8, 0, 0],[np.inf, np.inf, np.inf])
        fft_fit_params,_ = curve_fit(decay, fftfreq[1:], fftmag[1:], p0=fit_guess, bounds=bounds)
        self.dispersion_analysis["fft_fit_params"] = fft_fit_params
        self.dispersion_analysis["fftmag"] = fftmag
        self.dispersion_analysis["fftfreq"] = fftfreq

        # Now fit the peaks with the width that we found using the FFT
        lorentzian_with_width = partial(lorentzian, w=fft_fit_params[1])

        peaks = []
        for peak_guess in peak_indices:
            p0 = (combined[peak_guess], qubit_frequencies_MHz[peak_guess])
            bounds = ([0, qubit_frequencies_MHz[0]], [np.inf, qubit_frequencies_MHz[-1]])
            try:
                params,_ = curve_fit(lorentzian_with_width, qubit_frequencies_MHz, combined, p0=p0, bounds=bounds)
                peaks.append(params)
            except:
                pass

        if assigned_peak_numbers is None:
            assigned_peak_numbers = np.arange(len(peaks)-1, -1, -1)

        # Assign peak numbers, since if a peak was missed or one was added the 
        # indices will not correspond to the photon numbers
        #  np.concatenate(([-1,-1], np.arange(23,-1,-1)))
        peaks = [{"A": p[0], "x0": p[1], "num": num} for p,num in zip(peaks, assigned_peak_numbers) if num >= 0]

        # Sort the peaks by assigned number
        peaks = sorted(peaks, key=lambda p: p["num"])
        self.dispersion_analysis["peaks"] = peaks

        # Now that we've sorted the peaks in terms of photon number, fit to a quadratic
        chi_fit,_ = curve_fit(
            quadratic, 
            [p["num"] for p in peaks], 
            [p["x0"] for p in peaks], 
            p0=(0, 0, peaks[0]["x0"]))

        self.dispersion_analysis["chi_fit"] = {"a": chi_fit[0], "b": chi_fit[1], "c": chi_fit[2]}

    def plot_dispersion_analysis(self):
        import matplotlib.pyplot as plt

        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(self.qubit_frequencies, self.dispersion_analysis["traces"].T, ".-")
        ax.set_xlabel("Qubit Frequency [GHz]")
        ax.set_ylabel("Measurement Polarization")
        ax.set_title("Raw Measurement with Noise Floor Offset")
        ax.grid()

        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(self.qubit_frequencies, self.dispersion_analysis["combined_trace"], ".-")
        ax.set_xlabel("Qubit Frequency [GHz]")
        ax.set_ylabel("Measurement Polarization")
        ax.set_title("Detected Peaks in Combined Data")
        ax.grid()
        for peak in self.dispersion_analysis["peak_guess_indices"]:
            ax.axvline(self.qubit_frequencies[peak], color="red")

        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(self.dispersion_analysis["fftfreq"][1:], self.dispersion_analysis["fftmag"][1:], ".", color="white")
        ax.plot(self.dispersion_analysis["fftfreq"][1:], decay(self.dispersion_analysis["fftfreq"][1:], *self.dispersion_analysis["fft_fit_params"]), "--", color="red")
        ax.set_title(f"FFT of Zero-Displacement Peak\nwidth={round(self.dispersion_analysis['fft_fit_params'][1], 3)} MHz")
        ax.set_xlabel("Time Bin [1/MHz]")
        ax.set_ylabel("FFT Magnitude")
        ax.grid()
        # ax.set_yscale("log")
        ax.set_xlim(0, 5)

        qubit_frequencies_MHz = self.qubit_frequencies*1e-6

        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(qubit_frequencies_MHz, self.dispersion_analysis["combined_trace"], ".")
        ax.set_title("Fit Peaks and their Assigned Numbers")
        ax.set_xlabel("Qubit Frequency [MHz]")
        ax.set_ylabel("Measurement Polarization")
        ax.grid()
        for peak in self.dispersion_analysis["peaks"]:
            ax.plot(qubit_frequencies_MHz, lorentzian(qubit_frequencies_MHz, A=peak["A"], x0=peak["x0"], w=self.dispersion_analysis["fft_fit_params"][1]), "--", color="red")
            ax.text(s=f"{peak['num']}", x=peak['x0'], y=1, color="red")

        peak_numbers = np.array([p["num"] for p in self.dispersion_analysis["peaks"]])
        peak_centers = np.array([p["x0"] for p in self.dispersion_analysis["peaks"]])

        chi_fit = self.dispersion_analysis['chi_fit']
        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(peak_numbers, peak_centers, ".")
        ax.plot(peak_numbers, quadratic(peak_numbers, **chi_fit), "--", color="red")
        ax.set_title(f"Peak Location vs. Number\nf = {round(chi_fit['c'], 3)} MHz + n*{round(chi_fit['b'], 3)} MHz + n**2*{round(chi_fit['a']*1e6, 3)} Hz")
        ax.set_xlabel("Peak Number")
        ax.set_ylabel("Center Location [MHz]")
        ax.grid()

        fig,ax = plt.subplots(figsize=(10,3))
        ax.plot(peak_numbers, peak_centers - peak_numbers * chi_fit['b'] - chi_fit['c'], ".")
        ax.plot(peak_numbers, quadratic(peak_numbers, **chi_fit) - peak_numbers * chi_fit['b'] - chi_fit['c'], "--", color="red")
        ax.set_title("Peak Location Nonlinear Deviation")
        ax.set_xlabel("Peak Number")
        ax.set_ylabel("Peak Center Location - f0 - chi*n [MHz]")
        ax.grid()
