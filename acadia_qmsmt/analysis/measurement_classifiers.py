from typing import Union, Any
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

def create_classifier_from_config(config: dict):
    """
    Create a classifier from configuration information.
    """
    if not isinstance(config, dict):
        raise TypeError(f"Classifier configuration must be a dictionary;"
                        f" received object of type {type(config)}")

    if "type" not in config.keys():
        raise KeyError(f"Classifier configuration dict missing type; received {config}")

    if not isinstance(config["type"], str):
        raise TypeError(f"Classifier type must be specified by a string;"
                        f" received object of type {type(config['type'])}")

    config_copy = deepcopy(config)
    cls = eval(config_copy.pop("type"))
    classifier = cls.__new__(cls)
    classifier.__setstate__(config_copy)
    return classifier

class BaseClassifier(ABC):

    @abstractmethod
    def classify(self, data: NDArray) -> NDArray:
        """
        Classify an array of inputs.

        In general, the output will be expected to have the same shape as the input,
        but classifiers may eliminate axes as necessary.
        """
        pass

    @abstractmethod
    def train(self, data: NDArray, data_labels: NDArray):
        """
        Train the classifier on an array of inputs and their corresponding labels.
        This should only be called for supervised classifiers.
        """
        pass


class UnsupervisedClassifier(BaseClassifier):

    def train(self, data: NDArray, data_labels: NDArray):
        """
        Train the classifier on an array of inputs and their corresponding labels.
        This should only be called for supervised classifiers.
        """
        raise TypeError("Unsupervised classifiers cannot be trained.")


class RealQuadratureClassifier(UnsupervisedClassifier):
    """
    Given an array of complex input data, this classifier classifies each input sample 
    according to the sign of its real component.

    Input data should be an N-dimensional array of complex numbers.
    """

    def __getstate__(self):
        raise ValueError("A RealQuadratureClassifier has no state.")

    def __setstate__(self, state: dict):
        if len(state) == 0:
            return

        raise ValueError("A RealQuadratureClassifier has no state.")

    def classify(self, data: NDArray) -> NDArray:
        data_thresholded = (1.0 - np.sign(data.real)) / 2
        return data_thresholded

    @property
    def num_labels(self):
        return 2


class MaximalVarianceAxisClassifier(RealQuadratureClassifier):
    """
    Given an array of complex input data, this classifier finds the rotation which 
    maximizes the variance along the real axis, and classifies each input sample 
    according to the sign of its transformed real component.

    Input data should be an N-dimensional array of complex numbers.
    """

    def classify(self, data: NDArray) -> NDArray:

        def std_q(theta_):
            data_temp = data * np.exp(1j * theta_)
            return np.std(data_temp.imag)

        res = minimize_scalar(std_q, bounds=[0, 2*np.pi])
        angle = res.x
        data_rotated = data * np.exp(1j * angle)
        # guess I component center based on min and max of rotated blobs
        i_center = (np.min(data_rotated.real) + np.max(data_rotated.real)) / 2
        data_centered = data_rotated - i_center
        return super().classify(data_centered)