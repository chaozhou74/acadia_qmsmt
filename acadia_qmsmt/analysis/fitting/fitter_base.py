from typing import Any, Callable, Dict, Optional, Tuple, Union
import inspect
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

logger = logging.getLogger(__name__)

class FitterBase:
    """
    Base class for fitting data using scipy.optimize.curve_fit with named parameter support,
    optional fixed parameters, uncertainty tracking, and plotting utilities.

    Subclasses must implement:
    - `model(coordinates, **params)`: the model function to be fitted
    - `guess(coordinates, data)`: initial parameter guesses as a dict
    """

    def __init__(self, coordinates: np.ndarray, data: np.ndarray,
                 params: Optional[Dict[str, Union[float, Dict[str, Any]]]] = None, sigma: Optional[np.ndarray] = None,
                 dry: bool = False, absolute_sigma: bool = True, **fit_kwargs):
        """

        :param coordinates: Independent variable(s), passed to the model.
        :param data: Measured dependent data.
        :param params: Dict of parameter definitions. Each value can be:
                    - float (initial guess)
                    - dict with keys:
                        - 'value': initial guess
                        - 'bounds': (lower, upper)
                        - 'fixed': True if parameter should not be varied
        :param sigma: Optional standard deviations of data for weighted fit.
        :param dry: If True, do not perform the fit; just use initial guesses.
        :param absolute_sigma: If True, errors are treated as absolute in uncertainty calc.
        :param fit_kwargs: Additional keyword arguments for `curve_fit`.
        """
        self.coordinates = np.array(coordinates)
        self.data = np.array(data)
        if sigma is not None:
            sigma = np.clip(sigma, np.mean(sigma/1e10), np.inf) 
        self.sigma = sigma
        self.absolute_sigma = absolute_sigma

        self.param_order = self._extract_param_order()
        params = {} if params is None else params
        self.params_def = self.guess(self.coordinates, self.data)
        self.params_def.update(params)

        # Validate provided param keys match expected
        provided_keys = set(self.params_def.keys())
        expected_keys = set(self.param_order)
        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys
        if missing:
            raise ValueError(f"Missing parameters for model: {missing}")
        if extra:
            raise ValueError(f"Unexpected parameters not used by model: {extra}")

        self._parse_params()

        self.popt = None
        self.pcov = None
        self.perr = None
        self.ufloat_results = {}

        # default settings
        use_fit_kwargs = {"maxfev": 1000, "method": "dogbox"}  # or 'dogbox' for more robust fitting but slower
        use_fit_kwargs.update(fit_kwargs)

        if dry:
            self.popt = self.initial_full
            self.pcov = np.full((len(self.popt), len(self.popt)), np.nan)
        else:
            try:
                fit_popt, fit_pcov = curve_fit(self._fit_model, self.coordinates, self.data,
                                                p0=self.initial, bounds=self.bounds, sigma=self.sigma,
                                                absolute_sigma=self.absolute_sigma, **use_fit_kwargs)
                self.popt = self._reinsert_fixed(fit_popt)
                self.pcov = self._reconstruct_cov(fit_pcov)
            except Exception as e:
                self.popt = [np.nan] * len(expected_keys)
                logger.error(e, exc_info=True)

        self._process_results()

        self.result=CompatResult(self)

    @staticmethod
    def model(coordinates: np.ndarray, **params) -> np.ndarray:
        """The model function to fit. Must be overridden by subclass."""
        raise NotImplementedError

    @staticmethod
    def guess(coordinates, data) -> Dict[str, Union[float, Dict[str, Any]]]:
        """Provide initial parameter guesses. Must be overridden by subclass."""
        raise NotImplementedError

    def _extract_param_order(self):
        """Extract parameter names from model signature."""
        sig = inspect.signature(self.model)
        names = list(sig.parameters.keys())
        return names[1:] if names and names[0] in {"x", "coordinates"} else names

    def _parse_params(self):
        """Parse parameter dictionary into internal fit structures."""
        self.initial = []
        self.bounds_lower = []
        self.bounds_upper = []
        self.fixed_mask = []
        self.fit_order = []
        self.initial_full = []

        for name in self.param_order:
            raw = self.params_def[name]
            if isinstance(raw, dict):
                val = raw.get("value", 0.0)
                fixed = raw.get("fixed", False)
                bounds = raw.get("bounds", (-np.inf, np.inf))
            else:
                val = raw
                fixed = False
                bounds = (-np.inf, np.inf)

            self.initial_full.append(val)

            if fixed:
                self.fixed_mask.append(True)
            else:
                self.fixed_mask.append(False)
                self.initial.append(val)
                self.bounds_lower.append(bounds[0])
                self.bounds_upper.append(bounds[1])
                self.fit_order.append(name)

        self.bounds = (np.array(self.bounds_lower), np.array(self.bounds_upper))

    def _fit_model(self, coordinates, *fit_params):
        """Internal callable passed to curve_fit."""
        full_params = self._reinsert_fixed(fit_params)
        return self.model(coordinates, **dict(zip(self.param_order, full_params)))

    def _reinsert_fixed(self, fit_params):
        """Reinsert fixed parameters into the full parameter list."""
        full = []
        idx = 0
        for is_fixed, val in zip(self.fixed_mask, self.initial_full):
            if is_fixed:
                full.append(val)
            else:
                full.append(fit_params[idx])
                idx += 1
        return np.array(full)

    def _reconstruct_cov(self, fit_pcov):
        """Reconstruct full covariance matrix including fixed parameters."""
        full_size = len(self.param_order)
        full_pcov = np.full((full_size, full_size), np.nan)
        idxs = [i for i, fixed in enumerate(self.fixed_mask) if not fixed]
        for i_full, i_fit in zip(idxs, range(len(idxs))):
            for j_full, j_fit in zip(idxs, range(len(idxs))):
                full_pcov[i_full, j_full] = fit_pcov[i_fit, j_fit]
        return full_pcov

    def _process_results(self):
        """Process fit results into ufloat dictionary and compute residuals."""
        errs = (
            [np.nan] * len(self.popt)
            if self.pcov is None
            else np.sqrt(np.diag(self.pcov))
        )
        self.perr = errs
        for name, val, err in zip(self.param_order, self.popt, errs):
            self.ufloat_results[name] = ufloat(val, err)

        self.residuals = self.data - self.eval(self.coordinates)
        self.r_squared = 1 - np.sum(self.residuals ** 2) / np.sum((self.data - np.mean(self.data)) ** 2)
        self.chi_squared = (
            np.sum(self.residuals ** 2 / self.sigma ** 2)
            if self.sigma is not None
            else np.sum(self.residuals ** 2)
        )

        self.result_string = ""
        for k, v in self.ufloat_results.items():
            self.result_string += f"{k}: {v:.4g}\n"
        self.result_string = self.result_string[:-1]

    def eval(self, coordinates: np.ndarray=None) -> np.ndarray:
        """Evaluate the fitted model at given coordinates."""
        if coordinates is None:
            coordinates = self.coordinates
        return self.model(coordinates, **dict(zip(self.param_order, self.popt)))

    def print(self):
        """
        print fitted result
        """
        print(f"Fit result:")
        print(self.result_string)

    def plot(self, ax=None, oversample=5, data_kwargs=None, result_kwargs=None):
        """
        Plot the fitted model over a finer grid.

        :param ax: Optional matplotlib axis. If None, creates a new figure.
        :param oversample: Oversampling factor for fine plotting.
        :param kwargs: Additional plot keyword args.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        data_kwargs_ = dict(linestyle='', marker='o', markersize=7, label="data")
        result_kwargs_ = {"label": f'fit: {self.result_string}'}

        data_kwargs_.update(data_kwargs or {})
        result_kwargs_.update(result_kwargs or {})
        if self.sigma is None:
            ax.plot(self.coordinates, self.data, **data_kwargs_)
        else:
            ax.errorbar(self.coordinates, self.data, self.sigma, **data_kwargs_)
        
        self.plot_fitted(ax=ax, oversample=oversample, **result_kwargs_)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig, ax

    def plot_fitted(self, ax=None, oversample=5, **kwargs):
        """
        Plot the fitted model over a finer grid.

        :param ax: Optional matplotlib axis. If None, creates a new figure.
        :param oversample: Oversampling factor for fine plotting.
        :param kwargs: Additional plot keyword args.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        fine_x = np.linspace(self.coordinates[0], self.coordinates[-1], len(self.coordinates) * oversample)
        ax.plot(fine_x, self.eval(fine_x), **kwargs)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        return fig, ax


class CompatResult:
    def __init__(self, fitter: FitterBase):
        """
        Lightweight mock of lmfit's Result object to maintain compatibility
        with older code expecting .eval() and .params.

        :param fitter: A fitted FitterBase instance providing the needed methods/attributes.
        """
        self.eval = fitter.eval
        self.params = fitter.ufloat_results


if __name__ == "__main__":
    from acadia_qmsmt.analysis.fitting import Cosine

    # generate fake data
    x = np.linspace(-0, 1, 100)
    y_true = 1.5 * np.cos(20 * x + 5)
    y_data = y_true + 0.1 * np.random.randn(len(x))

    # do fit
    # fit = Cosine(x, y_data)
    # if want to adjust guess parameter
    fit = Cosine(x, y_data, params={"f": {"value":5, "fixed": True}, "A": {"value": 2, "bounds":(1, np.inf)}})
    fit.print()
    fit.plot()
    plt.show()
