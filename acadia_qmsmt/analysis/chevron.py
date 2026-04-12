import logging

import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import curve_fit

from acadia_qmsmt.plotting import prepare_plot_axes
from acadia_qmsmt.utils.fourier_transform import fft_mag

logger = logging.getLogger(__name__)


class Chevron:
    def __init__(self, sweep_freqs_Hz: np.ndarray, t_list_sec: np.ndarray, data: np.ndarray, do_fits: bool = True):
        """

        :param sweep_freqs_Hz:
        :param t_list_sec:
        :param data:
        :param do_fits:
        """
        self.sweep_freqs_Hz = sweep_freqs_Hz
        self.t_list_sec = t_list_sec
        self.data = data

        self.fft_freqs, self.fft_data = fft_mag(
            self.t_list_sec,
            self.data,
            axis=1,
            remove_zero_freq=True,
        )

        # Public analysis / fit results
        self.fitted_f0 = None
        self.fitted_g = None
        self.fitted_g0 = None
        self.fitted_t0 = None
        self.best_swap_freq = None
        self.best_swap_time = None
        self.best_swap_str = ""

        # Public fit bookkeeping that downstream code may access
        self.line_cut_fit_popt = None
        self.line_cut_fit_model = self._get_linecut_model()
        self.sweep_freq_fit = None
        self.fft_freq_fit = None
        self.fit_freq_scale = None
        self.fit_fft_freq_min = None
        self.fit_fft_freq_max = None
        self.fit_sweep_freq_min = None
        self.fit_sweep_freq_max = None

        if do_fits:
            try:
                self.fit_fft()
                self.fit_center_time_linecut()
            except Exception:
                logger.exception("Chevron auto-fit failed during initialization.")

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _update_best_swap_str(self):
        if (self.best_swap_freq is not None) and (self.best_swap_time is not None):
            self.best_swap_str = (
                f"Best SWAP freq: {self.best_swap_freq / 1e9:.6g} GHz, "
                f"time: {self.best_swap_time * 1e9:.3g} ns"
            )
        else:
            self.best_swap_str = ""

    def _require_fft_fit(self):
        if self.fitted_f0 is None or self.fitted_g is None:
            raise RuntimeError("FFT fit has not been run yet. Call fit_fft() first.")

    def _get_center_freq_idx(self):
        self._require_fft_fit()
        return int(np.argmin(np.abs(self.sweep_freqs_Hz - self.fitted_f0)))

    def _get_fft_fit_model(self, freq_scale):
        def _fit_model(f, f0, g):
            # apply freq scale for non-1st order processes
            # we enforce proper unit at the input of the class
            return np.sqrt(g ** 2 + (f - f0) ** 2 * freq_scale ** 2)

        return _fit_model

    def _get_linecut_model(self):
        # Only consider decay during beamsplitting for now. Ignored dephasing effect in the fitting model.
        if np.mean(self.data[:, 0]) < 0.5:
            def _fit_model(t, A, g0, t0, tau):
                return A * (1 + np.cos(2 * np.pi * 2 * g0 * (t - t0))) * np.exp(-t / tau)
        else:
            def _fit_model(t, A, g0, t0, tau):
                return A * (1 - np.cos(2 * np.pi * 2 * g0 * (t - t0))) * np.exp(-t / tau)

        return _fit_model

    def _get_masked_fft_data(self, fft_freq_min, fft_freq_max):
        fft_freq_mask = (self.fft_freqs > fft_freq_min) & (self.fft_freqs < fft_freq_max)
        fft_freqs = self.fft_freqs[fft_freq_mask]
        fft_data = self.fft_data[:, fft_freq_mask]

        if fft_data.size == 0:
            raise ValueError("No FFT data left after applying fft frequency mask.")

        return fft_freqs, fft_data

    def _build_peak_mask(
            self,
            fft_maxes,
            sweep_freq_min,
            sweep_freq_max,
            peak_threshold,
    ):
        peak_cutoff = np.min(fft_maxes) + np.ptp(fft_maxes) * peak_threshold
        amplitude_mask = fft_maxes > peak_cutoff
        sweep_freq_mask = (
                (self.sweep_freqs_Hz >= sweep_freq_min)
                & (self.sweep_freqs_Hz <= sweep_freq_max)
        )
        return amplitude_mask & sweep_freq_mask

    def _get_linecut_initial_guess(self, time_linecut):
        self._require_fft_fit()

        g_guess = max(float(self.fitted_g), 1e-12)
        t0_guess = 1.0 / (4 * g_guess)
        tau_guess = max(float(self.t_list_sec[-1] * 10), 1e-12)
        amp_guess = max(float(np.max(time_linecut) / 2), 1e-12)

        return amp_guess, g_guess, t0_guess, tau_guess

    # ----------------------------
    # Plotting
    # ----------------------------
    def plot_chevron(self, ax: Axes = None, figsize=None):
        fig, axs = prepare_plot_axes(ax, axs_shape=(1, 1), figsize=figsize)

        pcm = axs.pcolormesh(
            self.sweep_freqs_Hz / 1e9,
            self.t_list_sec * 1e6,
            self.data.T,
            vmin=0,
            vmax=1,
            cmap="bwr",
        )
        fig.colorbar(pcm, ax=axs, label="epop")

        axs.set_xlabel("BS freq (GHz)")
        axs.set_ylabel("Time (us)")
        axs.set_xlim(min(self.sweep_freqs_Hz) / 1e9, max(self.sweep_freqs_Hz) / 1e9)
        axs.set_ylim(min(self.t_list_sec) * 1e6, max(self.t_list_sec) * 1e6)

        self._update_best_swap_str()

        if self.fitted_f0 is not None and self.best_swap_time is not None:
            x = self.fitted_f0 / 1e9
            y = self.best_swap_time * 1e6

            axs.plot(
                x,
                y,
                marker="x",
                color="black",
                markersize=14,
                markeredgewidth=4,
                linestyle="",
            )
            axs.plot(
                x,
                y,
                marker="x",
                color="yellow",
                markersize=10,
                markeredgewidth=3,
                linestyle="",
            )

        fig.tight_layout()
        return fig, axs

    def plot_fft(self, ax: Axes = None, figsize=None):
        fig, ax = prepare_plot_axes(ax, axs_shape=(1, 1), figsize=figsize)

        pcm = ax.pcolormesh(
            self.sweep_freqs_Hz / 1e9,
            self.fft_freqs / 1e6,
            self.fft_data.T,
            cmap="inferno",
            shading="auto",
        )
        fig.colorbar(pcm, ax=ax)
        ax.set_title("FFT of time signal")

        if self.fitted_f0 is not None:
            fit_model = self._get_fft_fit_model(self.fit_freq_scale)

            ax.plot(self.sweep_freq_fit / 1e9, self.fft_freq_fit / 1e6, "w.", linestyle="")

            title_text = (
                f"f0: {self.fitted_f0 / 1e9:.6g} [GHz],   "
                f"g: {self.fitted_g / 1e6:.4g} [MHz],  "
                f"Drive order: {self.fit_freq_scale}"
            )
            ax.set_title(f"FFT of time signal\n{title_text}")

            fine_x = np.linspace(
                min(self.sweep_freqs_Hz),
                max(self.sweep_freqs_Hz),
                len(self.sweep_freqs_Hz) * 10,
            )
            ax.plot(
                fine_x / 1e9,
                fit_model(fine_x, self.fitted_f0, self.fitted_g * 2) / 1e6,
                "w--",
                label="Fit",
            )
            ax.axvline(
                self.fitted_f0 / 1e9,
                linestyle="dotted",
                color="w",
                label="On-Resonance",
            )
            ax.set_ylim(self.fit_fft_freq_min / 1e6, self.fit_fft_freq_max / 1e6)
            ax.set_xlim(self.fit_sweep_freq_min / 1e9, self.fit_sweep_freq_max / 1e9)

        ax.set_ylabel("FFT Frequency (MHz)")
        ax.set_xlabel("Sweep Frequency (GHz)")
        fig.tight_layout()
        return fig, ax

    def plot_linecut_fit(self, ax: Axes = None, figsize=None):
        center_freq_idx = self._get_center_freq_idx()
        time_linecut = self.data[center_freq_idx, :]

        fig, ax = prepare_plot_axes(ax, axs_shape=(1, 1), figsize=figsize)
        ax.plot(self.t_list_sec, time_linecut, "o", label="Data")

        t_list_fine = np.linspace(
            self.t_list_sec[0],
            self.t_list_sec[-1],
            len(self.t_list_sec) * 10,
        )

        title = f"Linecut at BS frequency {self.sweep_freqs_Hz[center_freq_idx]:.6g}"

        if self.line_cut_fit_popt is not None:
            ax.plot(
                t_list_fine,
                self.line_cut_fit_model(t_list_fine, *self.line_cut_fit_popt),
                label="fit",
            )
            ax.axvline(
                self.best_swap_time,
                color="red",
                linestyle="--",
                label=f"best swap time: {self.best_swap_time * 1e9:.1f} ns",
            )
            title += (
                f"\n"
                f"tau: {self.line_cut_fit_popt[3] * 1e6:.3f} us, "
                f"g0: {self.fitted_g0 / 1e6:.4g} MHz"
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("epop")
        ax.set_title(title)
        ax.set_xlim(self.t_list_sec[0], self.t_list_sec[-1])
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig, ax

    # ----------------------------
    # Fitting
    # ----------------------------
    def fit_fft(
            self,
            peak_threshold=0.4,
            fft_freq_min=None,
            fft_freq_max=None,
            sweep_freq_min=None,
            sweep_freq_max=None,
            freq_scale: int = 1,
    ):
        fft_freq_min = min(self.fft_freqs) if fft_freq_min is None else fft_freq_min
        fft_freq_max = max(self.fft_freqs) if fft_freq_max is None else fft_freq_max
        sweep_freq_min = min(self.sweep_freqs_Hz) if sweep_freq_min is None else sweep_freq_min
        sweep_freq_max = max(self.sweep_freqs_Hz) if sweep_freq_max is None else sweep_freq_max

        fft_freqs, fft_data = self._get_masked_fft_data(fft_freq_min, fft_freq_max)

        fft_max_idxes = np.argmax(fft_data, axis=1)
        fft_maxes = np.max(fft_data, axis=1)

        peak_mask = self._build_peak_mask(
            fft_maxes=fft_maxes,
            sweep_freq_min=sweep_freq_min,
            sweep_freq_max=sweep_freq_max,
            peak_threshold=peak_threshold,
        )

        self.sweep_freq_fit = self.sweep_freqs_Hz[peak_mask]
        self.fft_freq_fit = fft_freqs[fft_max_idxes][peak_mask]

        if len(self.sweep_freq_fit) < 2:
            raise ValueError("Not enough points selected for FFT fit.")

        fit_model = self._get_fft_fit_model(freq_scale)

        p0 = (
            self.sweep_freq_fit[np.argmin(self.fft_freq_fit)],
            np.min(self.fft_freq_fit),
        )
        bounds = (
            (np.min(self.sweep_freq_fit), 0),
            (np.max(self.sweep_freq_fit), np.max(self.fft_freq_fit)),
        )

        popt, pcov = curve_fit(
            fit_model,
            self.sweep_freq_fit,
            self.fft_freq_fit,
            p0=p0,
            bounds=bounds,
        )

        self.fitted_f0 = popt[0]
        self.fitted_g = abs(popt[1] / 2)

        self.fit_freq_scale = freq_scale
        self.fit_fft_freq_min = fft_freq_min
        self.fit_fft_freq_max = fft_freq_max
        self.fit_sweep_freq_min = sweep_freq_min
        self.fit_sweep_freq_max = sweep_freq_max

        self.best_swap_freq = self.fitted_f0
        self._update_best_swap_str()

        return self.fitted_f0, self.fitted_g

    def fit_center_time_linecut(self):
        center_freq_idx = self._get_center_freq_idx()
        time_linecut = self.data[center_freq_idx, :]

        p0 = self._get_linecut_initial_guess(time_linecut)
        bounds = ((0, 0, 0, 0), (1.1, np.inf, np.inf, np.inf))

        popt, pcov = curve_fit(
            self.line_cut_fit_model,
            self.t_list_sec,
            time_linecut,
            p0=p0,
            bounds=bounds,
        )

        self.fitted_g0 = abs(popt[1])
        self.fitted_t0 = popt[2]
        self.best_swap_time = round(self.fitted_t0 * 1e9 / 5) * 5e-9
        self.line_cut_fit_popt = popt
        self._update_best_swap_str()

        return self.fitted_g0, self.fitted_t0
