from typing import Tuple, Any, Optional, Union, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import lmfit
from lmfit import Parameter
from uncertainties import ufloat


class FitterBase():
    def __init__(self, coordinates: Union[Tuple[np.ndarray, ...], np.ndarray], data: np.ndarray,
                 params:Dict[str, Union[float, Parameter]]=None, dry=False, **fit_kwargs):
        """
        Base class for fitting data with lmfit.

        :param coordinates: The independent variable(s) for the fitting process.
        :param data: The dependent variable to be fitted.
        :param params: Dictionary of initial guess parameters. Each parameter can be:
            - A float: Represents the initial guess for the parameter.
            - An lmfit.Parameter: Specifies more detailed properties, such as bounds or whether
              the parameter is fixed.
        :param dry: When True, runs with the input parameters without performing the actual fit.
        :param fit_kwargs: Additional keyword arguments to be passed to the `lmfit.Model.fit`.
        """
        self.coordinates = coordinates
        self.data = data
        self.pre_process()
        self.result = self.run(params, dry, **fit_kwargs)

        # put fit result in a dict of ufloats for easy access
        self.ufloat_results = {}
        for k, v in self.result.params.items():
            stderr = np.nan if v.stderr is None else v.stderr
            self.ufloat_results[k] = ufloat(v.value, stderr)

    def pre_process(self):
        self.coordinates = np.array(self.coordinates)
        self.data = np.array(self.data)

    @staticmethod
    def model(*arg, **kwarg) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def guess(coordinates, data) -> Dict[str, Any]:
        raise NotImplementedError

    def run(self, params=None, dry=False, **fit_kwargs) -> lmfit.model.ModelResult:
        """
        Run the fit with lmfit.
        :param params:
        :param dry:
        :param fit_kwargs:
        :return:
        """
        params = {} if params is None else params
        
        model = lmfit.model.Model(self.model)

        lmfit_params = lmfit.Parameters()
        coordinates, data = self.coordinates, self.data
        guess_params = self.guess(coordinates, data)
        guess_params.update(params)

        for pn, pv in guess_params.items():
            if isinstance(pv, lmfit.Parameter):
                lmfit_params.add(pv)
            else:
                lmfit_params.add(pn, value=pv)

        if dry:
            lmfit_result = lmfit.model.ModelResult(model, params=lmfit_params,
                                                   data=data,
                                                   fcn_kws=dict(coordinates=coordinates))
        else:
            lmfit_result = model.fit(data, params=lmfit_params,
                                     coordinates=coordinates, **fit_kwargs)

        return lmfit_result

    def plot_fitted(self, ax=None, oversample:int=10, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = ax.get_figure()

        fine_coords = np.linspace(self.coordinates[0], self.coordinates[-1], len(self.coordinates) * oversample)
        ax.plot(fine_coords, self.result.eval(coordinates=fine_coords), **kwargs)
        ax.grid()
        ax.legend()
        fig.tight_layout()
        return fig, ax
