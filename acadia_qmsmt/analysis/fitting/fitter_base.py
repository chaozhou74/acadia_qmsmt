from typing import Any, Callable, Dict, Optional, Tuple, Union
import inspect
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

logger = logging.getLogger(__name__)

# Optional lmfit backend
try:
    import lmfit
    _HAS_LMFIT = True
except Exception:
    lmfit = None
    _HAS_LMFIT = False


class FitterBase:
    """
    Base class for fitting data using lmfit if available, else scipy.optimize.curve_fit,
    with named parameter support, optional fixed parameters, uncertainty tracking,
    and plotting utilities.

    Subclasses must implement:
    - `model(coordinates, **params)`: the model function to be fitted
    - `guess(coordinates, data)`: initial parameter guesses as a dict
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        data: np.ndarray,
        sigma: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Union[float, Dict[str, Any]]]] = None,
        dry: bool = False,
        absolute_sigma: bool = True,
        **fit_kwargs
    ):
        """
        :param coordinates: Independent variable(s), passed to the model.
        :param data: Measured dependent data.
        :param sigma: Optional standard deviations of data for weighted fit.
        :param params: Dict of parameter definitions. Each value can be:
                    - float (initial guess)
                    - dict with keys:
                        - 'value': initial guess
                        - 'bounds': (lower, upper)
                        - 'fixed': True if parameter should not be varied
        :param dry: If True, do not perform the fit; just use initial guesses.
        :param absolute_sigma: If True, errors are treated as absolute in uncertainty calc.
        :param fit_kwargs: Additional keyword arguments for the backend optimizer.
        """
        self.coordinates = np.array(coordinates)
        self.data = np.array(data)
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            floor = max(np.mean(np.abs(sigma)) * 1e-10, np.finfo(float).eps)
            sigma = np.clip(sigma, floor, np.inf)
        self.sigma = sigma
        self.absolute_sigma = absolute_sigma

        self.param_order = self._extract_param_order()
        params = {} if params is None else params
        guess_def = self.guess(self.coordinates, self.data)

        def _norm_params(p):
            if isinstance(p, dict):
                return {
                    "value": p.get("value", 0.0),
                    "bounds": p.get("bounds", (-np.inf, np.inf)),
                    "fixed": bool(p.get("fixed", False)),
                }
            return {"value": float(p), "bounds": (-np.inf, np.inf), "fixed": False}

        # Start from guess; only update fields user supplied (keeps guess value if only bounds given)
        params_def = {name: _norm_params(guess_def[name]) for name in self.param_order}
        for name, supplied in params.items():
            if name not in params_def:
                raise ValueError(f"Unexpected parameter not used by model: {name}")
            if isinstance(supplied, dict):
                for k in ("value", "bounds", "fixed"):
                    if k in supplied:
                        params_def[name][k] = supplied[k]
            else:
                # user passed a bare float -> set value only
                params_def[name]["value"] = float(supplied)

        self.params_def = {name: params_def[name] for name in self.param_order}

        # Validate keys
        provided_keys = set(self.params_def.keys())
        expected_keys = set(self.param_order)
        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys
        if missing:
            raise ValueError(f"Missing parameters for model: {missing}")
        if extra:
            raise ValueError(f"Unexpected parameters not used by model: {extra}")

        # Parse -> vectors used by SciPy backend (and for bookkeeping)
        self._parse_params()

        self.popt = None
        self.pcov = None
        self.perr = None
        self.ufloat_results = {}

        # Default optimizer kwargs (used by whichever backend)
        self._fit_kwargs = {"max_nfev": 1000}  # for lmfit least_squares; SciPy maps to maxfev
        self._fit_kwargs.update(fit_kwargs)

        self.backend = "lmfit" if _HAS_LMFIT else "scipy"

        if dry:
            self.popt = np.array(self.initial_full, dtype=float)
            self.pcov = np.full((len(self.popt), len(self.popt)), np.nan)
            self._backend_result = None
        else:
            try:
                if _HAS_LMFIT:
                    self._fit_with_lmfit()
                else:
                    self._fit_with_scipy()
            except Exception as e:
                logger.error("Fit failed", exc_info=True)
                self.popt = np.array([np.nan] * len(self.param_order))
                self.pcov = np.full((len(self.param_order), len(self.param_order)), np.nan)
                self._backend_result = None

        self._process_results()
        self.result = CompatResult(self)

    # ----- abstract API -----
    @staticmethod
    def model(coordinates: np.ndarray, **params) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def guess(coordinates, data) -> Dict[str, Union[float, Dict[str, Any]]]:
        raise NotImplementedError

    # ----- internals -----
    def _extract_param_order(self):
        sig = inspect.signature(self.model)
        names = list(sig.parameters.keys())
        return names[1:] if names and names[0] in {"x", "coordinates"} else names

    def _parse_params(self):
        self.initial = []
        self.bounds_lower = []
        self.bounds_upper = []
        self.fixed_mask = []
        self.fit_order = []
        self.initial_full = []

        for name in self.param_order:
            spec = self.params_def[name]
            val, bounds, fixed = spec["value"], spec["bounds"], bool(spec["fixed"])
            self.initial_full.append(val)
            if fixed:
                self.fixed_mask.append(True)
            else:
                self.fixed_mask.append(False)
                self.initial.append(val)
                self.bounds_lower.append(bounds[0])
                self.bounds_upper.append(bounds[1])
                self.fit_order.append(name)

        self.bounds = (np.array(self.bounds_lower, dtype=float),
                       np.array(self.bounds_upper, dtype=float))

    # ---------- SciPy backend ----------
    def _fit_model(self, coordinates, *fit_params):
        full_params = self._reinsert_fixed(fit_params)
        return self.model(coordinates, **dict(zip(self.param_order, full_params)))

    def _reinsert_fixed(self, fit_params):
        full = []
        idx = 0
        for is_fixed, val in zip(self.fixed_mask, self.initial_full):
            if is_fixed:
                full.append(val)
            else:
                full.append(fit_params[idx])
                idx += 1
        return np.array(full, dtype=float)

    def _reconstruct_cov(self, fit_pcov):
        full_size = len(self.param_order)
        full_pcov = np.full((full_size, full_size), np.nan, dtype=float)
        idxs = [i for i, fixed in enumerate(self.fixed_mask) if not fixed]
        for i_full, i_fit in zip(idxs, range(len(idxs))):
            for j_full, j_fit in zip(idxs, range(len(idxs))):
                full_pcov[i_full, j_full] = fit_pcov[i_fit, j_fit]
        return full_pcov

    def _fit_with_scipy(self):
        # Map generic kwargs to curve_fit
        cf_kwargs = {}
        if "max_nfev" in self._fit_kwargs and "maxfev" not in self._fit_kwargs:
            cf_kwargs["maxfev"] = self._fit_kwargs["max_nfev"]
        cf_kwargs.update({k: v for k, v in self._fit_kwargs.items() if k not in {"max_nfev"}})
        # A robust method by default (user can override via fit_kwargs)
        cf_kwargs.setdefault("method", "dogbox")

        fit_popt, fit_pcov = curve_fit(
            self._fit_model,
            self.coordinates,
            self.data,
            p0=self.initial if len(self.initial) else None,
            bounds=self.bounds if len(self.initial) else (-np.inf, np.inf),
            sigma=self.sigma,
            absolute_sigma=self.absolute_sigma,
            **cf_kwargs,
        )
        self.popt = self._reinsert_fixed(fit_popt)
        self.pcov = self._reconstruct_cov(fit_pcov)
        self._backend_result = ("scipy", {"popt": fit_popt, "pcov": fit_pcov})

    # ---------- lmfit backend ----------
    def _lmfit_params(self):
        p = lmfit.Parameters()
        for name in self.param_order:
            spec = self.params_def[name]
            val, (lo, hi), fixed = spec["value"], spec["bounds"], bool(spec["fixed"])
            p.add(name, value=float(val), min=float(lo), max=float(hi), vary=not fixed)
        return p

    def _residual_lmfit(self, params: "lmfit.Parameters", coords, data, sigma):
        vals = {name: params[name].value for name in self.param_order}
        res = self.model(coords, **vals) - data
        if sigma is not None:
            # absolute_sigma semantics: we use sigma as true stdev to weight residuals
            res = res / sigma
        return res

    def _fit_with_lmfit(self):
        params = self._lmfit_params()
        # Map a few user kwargs to lmfit.minimize (least_squares)
        method = self._fit_kwargs.get("method", "least_squares")
        max_nfev = self._fit_kwargs.get("max_nfev", 1000)

        result = lmfit.minimize(
            self._residual_lmfit,
            params,
            args=(self.coordinates, self.data, self.sigma),
            method=method,
            max_nfev=max_nfev,
        )

        # Extract best-fit values
        self.popt = np.array([result.params[n].value for n in self.param_order], dtype=float)

        # Build full covariance (if available, lmfit.covar is for varying params only)
        if result.covar is not None and result.var_names is not None:
            # Map var_names (only varied) into full square matrix
            name_to_fit_idx = {n: i for i, n in enumerate(result.var_names)}
            full_size = len(self.param_order)
            full_pcov = np.full((full_size, full_size), np.nan, dtype=float)
            for i_full, ni in enumerate(self.param_order):
                if ni not in name_to_fit_idx:
                    continue
                ii = name_to_fit_idx[ni]
                for j_full, nj in enumerate(self.param_order):
                    if nj not in name_to_fit_idx:
                        continue
                    jj = name_to_fit_idx[nj]
                    full_pcov[i_full, j_full] = result.covar[ii, jj]
            self.pcov = full_pcov
        else:
            self.pcov = np.full((len(self.param_order), len(self.param_order)), np.nan, dtype=float)

        self._backend_result = ("lmfit", result)

    # ---------- post-processing ----------
    def _process_results(self):
        errs = (
            [np.nan] * len(self.popt)
            if self.pcov is None
            else np.sqrt(np.diag(self.pcov))
        )
        self.perr = errs
        self.ufloat_results = {}
        for name, val, err in zip(self.param_order, self.popt, errs):
            self.ufloat_results[name] = ufloat(val, err)

        self.residuals = self.data - self.eval(self.coordinates)
        denom = np.sum((self.data - np.mean(self.data)) ** 2)
        self.r_squared = np.nan if denom == 0 else 1 - np.sum(self.residuals ** 2) / denom
        self.chi_squared = (
            np.sum(self.residuals ** 2 / self.sigma ** 2)
            if self.sigma is not None
            else np.sum(self.residuals ** 2)
        )

        # Build the same printable summary string
        self.result_string = ""
        for k, v in self.ufloat_results.items():
            self.result_string += f"{k}: {v:.4g}\n"
        self.result_string = self.result_string[:-1]

    # ---------- public helpers ----------
    def eval(self, coordinates: np.ndarray = None) -> np.ndarray:
        if coordinates is None:
            coordinates = self.coordinates
        return self.model(coordinates, **dict(zip(self.param_order, self.popt)))

    def print(self):
        print(f"Fit result:")
        print(self.result_string)

    def plot(self, ax=None, oversample=5, data_kwargs=None, result_kwargs=None):
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
        Lightweight mock to keep old code working:
        - .eval() -> uses fitter.eval
        - .params -> dict of ufloat results
        """
        self.eval = fitter.eval
        self.params = fitter.ufloat_results
