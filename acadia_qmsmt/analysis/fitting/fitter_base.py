from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal
import inspect
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import lmfit

logger = logging.getLogger(__name__)


class FitterBase:
    """
    Base class for fitting data using lmfit, with named parameter support, optional fixed parameters,
    uncertainty tracking, and plotting utilities.

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
        sigma_floor_frac: float = 0.01,
        error_model: Literal["auto", "binomial", "wls"]="auto",
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
        :param sigma_floor_frac:
            Fractional floor applied to `sigma` values to avoid numerical instabilities.
            Each `sigma` is clipped such that
            `sigma >= max(mean(|sigma|) * sigma_floor_frac, eps)`.

        :param error_model:
            Specifies how measurement uncertainty is handled. Options are:
                - **"wls"**: Weighted least squares. Uses `sigma` as absolute standard deviations
                  (weights = 1/sigma).
                - **"binomial"**: Binomial deviance model. Gives more robust fitting when the data are readout shots, in
                  which case the distribution is Bernoulli, so the variance is largest at y=0.5
                  If counts `k` and `N` are not provided, they will be inferred from the mean and `sigma`.
                - **"auto"**: Automatically chooses `"binomial"` when all `|data| <= 1`, otherwise `"wls"`.
            This choice affects both the residual definition and how uncertainties are propagated.

        :param fit_kwargs:
            Additional keyword arguments forwarded to the underlying optimizer (`lmfit.minimize`).
            Common keys include:
                - **method** (str): Optimization method (default `"least_squares"`).
                - **max_nfev** (int): Maximum number of function evaluations (default 1000).
                - **k**, **N**: Optional binomial counts per data point, used only when
                  `error_model="binomial"`.
        """
        self.coordinates = np.array(coordinates)
        self.data = np.array(data)
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            floor = max(np.mean(np.abs(sigma)) * sigma_floor_frac, np.finfo(float).eps)
            sigma = np.clip(sigma, floor, np.inf)
        self.sigma = sigma


        self.params_supplied = {} if params is None else params
        self.params_guess = self.guess(self.coordinates, self.data)
        self._lmfit_params:lmfit.Parameters = None
        self._make_lmfit_params()
        self.param_order = list(self._lmfit_params.keys())


        self.popt = None
        self.pcov = None
        self.perr = None
        self.ufloat_results = {}

        # Default optimizer kwargs (used by whichever backend)
        self._fit_kwargs = {"max_nfev": 1000}  # for lmfit least_squares; SciPy maps to maxfev
        self._fit_kwargs.update(fit_kwargs)


        self.error_model = self._decide_error_model() if error_model=="auto" else error_model
        # optional counts for binomial model
        self._binom_k = self._fit_kwargs.pop("k", None)
        self._binom_N = self._fit_kwargs.pop("N", None)
        self._last_residuals = None  # for diagnostics / chi2 reporting
        if self.error_model == "binomial":
            self.residual_fn = lambda p, x, y, s: self._residual_binomial_deviance(
                p, x, y, s, k=self._binom_k, N=self._binom_N
            )
        else:
            self.residual_fn = self._residual_wls

        try:
            self._fit_with_lmfit()
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
    def _decide_error_model(self)-> Literal["binomial", "wls"]:
        """
        decide the error model to use for the fitting based on the provided data
        Currently just assumes if the abs value of data are <=1 then we are taking shots, so it is binomial.
        Can be improved to smarter strategy later.
        """
        # If user passes k/N, or data are clearly proportions in (0,1), assume binomial.
        if self._fit_kwargs.get("k") is not None or self._fit_kwargs.get("N") is not None:
            return "binomial"
        y = np.asarray(self.data)
        if np.all(abs(y)<=1):
            return "binomial"
        return "wls"

    def _make_lmfit_params(self):
        self._lmfit_params = lmfit.Model(self.model).make_params()
        for name, param in self._lmfit_params.items():
            if name not in self.params_guess:
                raise ValueError(f"Missing guess parameter {name}")
            self._parse_params(param, self.params_guess[name])
            if name in self.params_supplied: # overwrite with supplied parameter
                self._parse_params(param, self.params_supplied[name])

    def _parse_params(self, lmfit_param:lmfit.Parameter, param_spec):
        if isinstance(param_spec, dict):
            val = param_spec.get("value")
            lo, hi = param_spec.get("bounds", [None, None])
            fixed = param_spec.get("fixed")
            lmfit_param.set(value=val, min=lo, max=hi, vary=None if fixed is None else not fixed)
        else:
            lmfit_param.set(value=param_spec)

    # ------- residual functions --------------
    def _clip_prob(self, p, eps=1e-12):
        return np.clip(p, eps, 1 - eps)

    def _residual_wls(self, params, coords, data, sigma):
        vals = {name: params[name].value for name in self.param_order}
        res = self.model(coords, **vals) - data
        if sigma is not None:
            res = res / sigma
        return res

    def _infer_counts_from_mean_sigma(self, y, sigma):
        """
        Infer effective counts (N_eff, k_eff) from mean y and its SE sigma.
        Using sigma^2 ≈ y(1-y)/N  =>  N ≈ y(1-y)/sigma^2,  k = N*y.
        Clipped to avoid pathologies near 0/1 and tiny sigmas.
        """
        y = np.asarray(y, float)
        sigma = np.asarray(sigma, float)
        y_c = np.clip(y, 1e-6, 1 - 1e-6)
        var = np.maximum(sigma**2, 1e-24)
        N_eff = y_c * (1.0 - y_c) / var
        # Cap ridiculously large N from tiny sigma to stabilize logs:
        N_eff = np.clip(N_eff, 1.0, 1e12)
        k_eff = N_eff * y_c
        return k_eff, N_eff

    def _residual_binomial_deviance(self, params, coords, data, sigma, k=None, N=None):
        """
        Binomial deviance residuals:
        r_i = sign(k - N p) * sqrt( 2 [ k log(k/(N p)) + (N-k) log((N-k)/(N(1-p))) ] )
        with 0*log(0)=0 and p clipped to (0,1).
        If k,N not supplied, infer from (y=data, sigma).
        """
        theta = {name: params[name].value for name in self.param_order}
        p = self._clip_prob(self.model(coords, **theta))

        if k is None or N is None:
            if sigma is None:
                # no counts and no sigma -> fall back to WLS/unweighted
                return self._residual_wls(params, coords, data, sigma=None)
            k, N = self._infer_counts_from_mean_sigma(data, sigma)

        k = np.asarray(k, float); N = np.asarray(N, float)
        mu = N * p
        # terms with safe 0*log(0) handling
        t1 = np.where(k > 0, k * np.log(np.maximum(k, 1e-300) / np.maximum(mu, 1e-300)), 0.0)
        t2 = np.where((N - k) > 0,
                    (N - k) * np.log(np.maximum(N - k, 1e-300) / np.maximum(N - mu, 1e-300)),
                    0.0)
        dev = 2.0 * (t1 + t2)
        res = np.sign(k - mu) * np.sqrt(np.maximum(dev, 0.0))
        return res


    # ---------- lmfit backend ----------
    def _fit_with_lmfit(self):
        params = self._lmfit_params
        method = self._fit_kwargs.get("method", "least_squares")
        max_nfev = self._fit_kwargs.get("max_nfev", 1000)

        result = lmfit.minimize(
            self.residual_fn,
            params,
            args=(self.coordinates, self.data, self.sigma),
            method=method,
            max_nfev=max_nfev,
            scale_covar=(self.error_model != "wls" or self.sigma is None)
        )

        # save latest residuals for reporting
        try:
            self._last_residuals = self.residual_fn(
                result.params, self.coordinates, self.data, self.sigma
            )
        except Exception:
            self._last_residuals = None

        # Extract best-fit values
        self.popt = np.array([result.params[n].value for n in self.param_order], dtype=float)

        # Expand covariance as before
        if result.covar is not None and result.var_names is not None:
            name_to_fit_idx = {n: i for i, n in enumerate(result.var_names)}
            full_size = len(self.param_order)
            full_pcov = np.full((full_size, full_size), np.nan, dtype=float)
            for i_full, ni in enumerate(self.param_order):
                ii = name_to_fit_idx.get(ni, None)
                if ii is None:
                    continue
                for j_full, nj in enumerate(self.param_order):
                    jj = name_to_fit_idx.get(nj, None)
                    if jj is None:
                        continue
                    full_pcov[i_full, j_full] = result.covar[ii, jj]
            self.pcov = full_pcov
        else:
            self.pcov = np.full((len(self.param_order), len(self.param_order)), np.nan, dtype=float)

        self._backend_result = ("lmfit", result)


    # ---------- post-processing ----------
    def _process_results(self):
        errs = ([np.nan] * len(self.popt) if self.pcov is None
                else np.sqrt(np.diag(self.pcov)))
        self.perr = errs
        self.ufloat_results = {n: ufloat(v, e) for n, v, e in zip(self.param_order, self.popt, errs)}

        yfit = self.eval(self.coordinates)
        self.residuals = self.data - yfit

        denom = np.sum((self.data - np.mean(self.data)) ** 2)
        self.r_squared = np.nan if denom == 0 else 1 - np.sum(self.residuals ** 2) / denom

        if self.error_model == "binomial" and self._last_residuals is not None:
            # Sum of squares of deviance residuals = model deviance
            self.chi_squared = float(np.sum(self._last_residuals**2))
        else:
            self.chi_squared = (np.sum(self.residuals ** 2 / self.sigma ** 2)
                                if self.sigma is not None else np.sum(self.residuals ** 2))

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
