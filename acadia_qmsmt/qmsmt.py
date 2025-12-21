import sys
import shutil
import os
from typing import Callable, Literal, Dict, List, Any, Union, Literal, Tuple
from pathlib import Path
import json
import logging
import math
import functools

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import hann as scipy_hann

from acadia import Acadia, Channel, Runtime, WaveformMemory, WaveformMemory, Operation
from acadia.compiler import ManagedResource, Symbol
from acadia.sample_arithmetic import complex_to_sample


##############################################################
# Todo: IN ACADIA_QMSMT
# Bug: stretch_length does not seem to work if your waveform memory is less than 3 cycles, ie less than 15 ns
# Bug: if ramp = 0 and flat is an odd multiple of clock_sample_time, the first sample is zero
# Todo: Implement a gaussian pulse
# Todo: Change Qubit, MeasurableResonator, and QMsmtRuntime to use InputOutput
# Todo: Should we allow get_waveform_memory to take a dictionary? 

##############################################################
# Todo: IN RUNTIMES
# Todo: Avoid hardcoding the pulse/memory names in the runtimes
##############################################################
# Todo: LOW PRIORITY
# Todo: Cleanup functions which aren't needed
# Todo: Adjust the imports in the __init__ file of acadia_qmsmt
# Todo: load_chromatic_waveform need not interact with the waveform memory
# Todo: Giving samples whose maximum value is 1 is causing overflow in the DAC
# Todo: Have type checking for all values read from a config file
# Todo: Do users even check logger warnings? Is there a better way to raise them?

__all__ = ["InputOutput", "InputOutputWaveforms", "QMsmtRuntime", "MeasurableResonator"]
logger = logging.getLogger("acadia_qmsmt")
# This is for handling floating point tolerance when computing waveforms
TOLERANCE = 1e-15
# This is the list of kwargs which aren't passed to waveform shape functions (hann etc) defined in InputOutputWaveforms.
KWARG_EXCLUDE_LIST = ["data", "scale", "phase", "detune", "ramp", "flat", "use_stretch", "memory_length", "name"]

def make_hash(val):
    '''Converts a dict/list/tuple to a hashable type'''
    if val is None:
        return None
    elif isinstance(val, dict):
        # Convert dict to a sorted tuple of (key, value) pairs, recursively hashable
        return ("__dict__", tuple((k, make_hash(v)) for k, v in sorted(val.items())))
    elif isinstance(val, np.ndarray):
        # Store bytes, dtype, shape as metadata
        return ("__ndarray__", val.tobytes(), str(val.dtype), val.shape)
    elif isinstance(val, list):
        # Tag as list and recursively hash
        return ("__list__", tuple(make_hash(v) for v in val))
    elif isinstance(val, tuple):
        # Tag as tuple and recursively hash
        return ("__tuple__", tuple(make_hash(v) for v in val))
    else:
        return val
    
def invert_hash(val):
    '''Inverse of the function `make_hash`'''
    if val is None:
        return None
    elif isinstance(val, tuple):
        if len(val) == 0:
            return ()
        tag = val[0]
        if tag == "__dict__":
            # val[1] is a tuple of (key, value) pairs
            return {k: invert_hash(v) for k, v in val[1]}
        elif tag == "__list__":
            # val[1] is a tuple of items
            return [invert_hash(v) for v in val[1]]
        elif tag == "__tuple__":
            # val[1] is a tuple of items
            return tuple(invert_hash(v) for v in val[1])
        elif tag == "__ndarray__":
            # val[1]: bytes, val[2]: dtype, val[3]: shape
            return np.frombuffer(val[1], dtype=val[2]).reshape(val[3])
        else:
            # Not a tagged tuple, treat as a tuple of values
            return tuple(invert_hash(v) for v in val)
    else:
        return val
    
class InputOutputWaveforms:
    """
    A class containing methods for all of the waveform shapes that can be 
    referenced by string in a configuration file.
    """

    @staticmethod
    def hann(t: NDArray) -> NDArray:
        """
        Calculate a Hann (or raised-cosine) function for a given list of times
        """
        return 0.5 * (1 - np.cos(2 * np.pi * t))
    
    @staticmethod
    def hann_drag(t: NDArray, rel_drag: float) -> NDArray:
        """
        Calculate a DRAG pulse with a Hann base pulse for a given list of times
        """
        return 0.5 * (1 - np.cos(2 * np.pi * t)) + (rel_drag * np.pi * np.sin(2 * np.pi * t) * 1j)

    @staticmethod
    def hamming(t: NDArray) -> NDArray:
        """
        Calculate a Hamming function for a given list of times
        """
        return 0.54 - 0.46 * np.cos(2 * np.pi * t)
    
    @staticmethod
    def blackman(t: NDArray) -> NDArray:
        """
        Calculate a Blackman function for a given list of times
        """
        return 0.42 - 0.5 * np.cos(2 * np.pi * t) + 0.08 * np.cos(4 * np.pi * t)
    
    @staticmethod
    def sum_of_cosines(t: NDArray, coeffs: NDArray) -> NDArray:
        """
        Return a generalized sum-of-cosines envelope. The entire output is filled.
        """
        z = 2 * np.pi * t
        output = np.zeros_like(t, dtype=np.float64)
        for idx, coeff in enumerate(coeffs):
            output[:] += coeff*np.cos(idx*z)
        
        # The signal has its maximum value at z = 1/2, where the cosines are passed integer multiples of pi
        # So the zeroth, second, etc have value 1 and the first, third, etc. terms have value -1
        signs = np.ones(len(coeffs), dtype=np.float64)
        signs[1::2] *= -1
        output[:] /= np.dot(coeffs, signs)
        return output

    @staticmethod
    def hft248d(t: NDArray) -> NDArray:
        """
        Calculate an HFT248d function. The entire output is filled.
        """
        coeffs = np.array([1, -1.985844164102, 1.791176438506, -1.282075284005, 0.667777530266, -0.240160796576, 0.056656381764, -0.008134974479, 0.000624544650, -0.000019808998, 0.000000132974])
        return InputOutputWaveforms.sum_of_cosines(t, coeffs)

    @staticmethod
    def matlab_flat_top(t: NDArray) -> NDArray:
        """
        Return a pulse with the MATLAB flat-top function. The entire output is filled.
        """
        coeffs = np.array([1, -0.41663158/0.21557895, 0.277263158/0.21557895, -0.083578947/0.21557895, 0.006947386/0.21557895])
        return InputOutputWaveforms.sum_of_cosines(t, coeffs)

    @staticmethod
    def piecewise_cosine(t: NDArray, handle_times: NDArray, handle_amplitudes: NDArray) -> None:
        """
        A piecewise-defined pulse, where the form of the signal between its 
        "handles" is a half-period of a cosine. The handle times are fractions 
        of the total pulse length. The entire output is filled.
        """

        handle_times = np.array(handle_times)
        handle_amplitudes = np.array(handle_amplitudes)

        piece_amplitudes = np.diff(handle_amplitudes)
        piece_lengths = np.diff(handle_times)

        # Rectangular windows that "select" the different pieces
        piece_rects = np.logical_and(t[None,:] >= handle_times[:-1,None], 
                                     t[None,:] < handle_times[1:,None]).astype(np.complex128)

        piece_cosines = piece_amplitudes[:,None] * 0.5 * (1 - np.cos(np.pi * (t[None,:] - handle_times[:-1,None]) / piece_lengths[:,None]))
        piece_shifted_cosines = piece_cosines + handle_amplitudes[:-1,None]
        return np.sum(piece_shifted_cosines * piece_rects, axis=0)

    @staticmethod
    def flattop_generator(
        output: NDArray, func: Callable, ramp: float = 0.5, flat: float = 0.5, **kwargs) -> None:
        '''
        Wrapper to add a flattop to functions that generate pulses.
        The function returns a center justified pulse with a flat top.
        ramp and flat are fractions of the total pulse length.
        The function is defined as follows:
        - For t <= (1 - ramp - flat)/2 or t >= ramp + flat, the value is 0.
        - For (1 - ramp - flat)/2 < t <= (1 - flat)/2, the function uses 
            the ramping function over the interval [0, 0.5) (ramp up).
        - For (1 - flat)/2 < t <= (1 + flat)/2, the waveform holds flat at func(0.5)
        - For (1 + flat)/2 < t <= (1 + ramp + flat)/2, the function uses the ramping
            function over the interval [0.5, 1] (ramp down).
        :param output: The output array to be filled with the pulse 
            (This will be modified in place)
        :param func: The function to be used for the ramping part of the pulse
        :param ramp: The length of the ramping part of the pulse, expressed as a
            fraction of the total pulse length
        :param flat: The length of the flat part of the pulse, expressed as a
            fraction of the total pulse length
        '''
        t = np.arange(output.size) / output.size

        bound1 = np.round((1 - ramp - flat)/2/TOLERANCE) * TOLERANCE
        bound2 = np.round((1 - flat)/2/TOLERANCE) * TOLERANCE
        bound3 = np.round((1 + flat)/2/TOLERANCE) * TOLERANCE
        bound4 = np.round((1 + ramp + flat)/2/TOLERANCE) * TOLERANCE
        # cond1 = t <= bound1 # Not needed, vals are zero
        cond2 = (t > bound1) & (t <= bound2)
        cond3 = (t >= bound2) & (t <= bound3)
        cond4 = (t > bound3) & (t <= bound4)
        # cond5 = t > bound4 # Not needed, vals are zero
    
        output[:] = 0  # Set all elements to zero in-place
        output[cond2] = func((t[cond2] - 0.5*(1 - ramp - flat))/ramp, **kwargs)
        output[cond3] = func(np.array([0.5]), **kwargs)
        output[cond4] = func((t[cond4] - 0.5*(1 - ramp + flat))/ramp, **kwargs)

    @staticmethod
    def scale_detune_pulse(pulse: NDArray, scale: Union[complex, list, tuple, NDArray] = None, phase: Union[complex, list, tuple, NDArray] = None,
                           detune: Union[float, list, tuple, NDArray] = None, sample_frequency: float = 8e8) -> NDArray:
        ''' 
        Scale and detune a pulse. Multichromatic pulses are implemented by 
        providing tuples, lists or NDarrays of scale and detune values, which must have the
        same length. :phase: can also be passed as an alternative to giving a complex scale.
        '''
        if isinstance(scale, (list, tuple, np.ndarray)) and isinstance(detune, (list, tuple, np.ndarray)):
            if len(scale) != len(detune):
                raise ValueError("If scale and detune are tuples or lists, they must have the same length.")
            if phase is not None:
                if len(phase) != len(scale): # NOTE: currently not accounting for case where you want multiple scales and detunes, but single phase
                    raise ValueError("If phase is a tuple or list, it must have the same length as scale and detune.")
                    
        else:
            if isinstance(scale, (tuple, list, np.ndarray)):
                detune = tuple(detune for _ in scale)
                phase = tuple(phase for _ in scale)
            elif isinstance(detune, (tuple, list, np.ndarray)):
                scale = tuple(scale for _ in detune) 
                phase = tuple(phase for _ in detune)
            else:
                scale = (scale,)
                detune = (detune,)
                phase = (phase,)
        pulse_out = np.zeros_like(pulse, dtype=np.complex128)
        for s, d, p in zip(scale, detune, phase):
            if s is None:
                s = 1
            if d is not None:
                t = np.arange(pulse.size, dtype=np.float64) / sample_frequency
                s = np.multiply(s, np.exp(2 * np.pi * 1j * d * t))
            if p is not None:
                s = np.multiply(s, np.exp(1j * p)) # NOTE: phase in radians
            pulse_out = np.add(pulse_out, np.multiply(pulse, s))
        if (np.abs(pulse_out) > 1).any():
            raise ValueError("The scaled pulse exceeds the maximum value of 1. ")
        return pulse_out
            
class InputOutput:
    """
    A base class for abstracting input and output channels. In some
    sense this abstracts Acadia's `Channel` object, but at a higher level
    so that certain patterns (such as for allocating memory) can be carried
    out with full knowledge of the configuration as provided by a dictionary.
    """

    def __init__(self, name: str, acadia: Acadia, config: Dict[str,Any]):
        self._name = name
        self._config: Dict[str,Any] = config
        self._acadia: Acadia = acadia
        self._channel: Channel = acadia.channel(self.get_config("channel"))
        self._pulse_cache: Dict[str, Dict] = {} # Stores all computed pulses and waveform memories 
        self._allocated_memories: Dict[str, WaveformMemory] = {}
        self._samples_per_cycle = self._acadia._firmware["rfdc"]["dac"]["channel_interface_width"][self._channel.num()] // 32
        
    def get_config(self, *cfg_path: str):
        """
        Retrieve a configuration value from the nested config dict of this IO channel.
        
        This function does the same thing as directly getting values using `self._config[][]...`,
        but with detailed error messages to identify the missing key and its location 
        when the path is invalid.

        :param cfg_path: Strings representing the path to the  desired configuration 
            value (e.g., ("stimulus", "memories", "readout", "length") ).

        """
        current_level = self._config
        for i, k in enumerate(cfg_path):
            # pick the only value if key is None and dict has only one value
            if k is None and len(current_level) == 1:
                current_level = next(iter(current_level.values()))
                continue
            # raise when key is missing
            if k not in current_level:
                raise KeyError(f"Missing key '{k}' in config dict under "
                               f"'{'.'.join([self._name] + list(cfg_path[:i]))}'")
            
            current_level = current_level[k]
        
        return current_level

    # ----------Pulse calculation and DAC specific methods ------------
    def get_pulse_config(self, pulse: Union[str,dict,None] = None) -> dict:
        """
        This method retrieves the configuration for a pulse, either from the
        configuration dictionary or from a provided dictionary. If no pulse is
        specified, the first pulse in the configuration is used.
        :param pulse: The name of the pulse in the configuration, or a dictionary
            containing the configuration of the pulse. If None, the first pulse in
            the configuration is used.
        :return: A dictionary containing the configuration of the specified pulse.
        """
        # took ~ 10 us
        if pulse is None:
            pulse = list(self._config["pulses"].values())[0]
            pulse["name"] = list(self._config["pulses"].keys())[0]
        elif isinstance(pulse, dict):
            if "name" not in pulse:
                pulse["name"] = None
        elif isinstance(pulse, str):
            pulse_name = pulse
            try:
                pulse = self.get_config("pulses", pulse_name)
            except KeyError:
                raise KeyError(f"Pulse '{pulse_name}' not found in the YAML config file.")
            pulse["name"] = pulse_name
        else:
            raise TypeError(f"Unable to specify pulse with object of type {type(pulse)}")
        
        if "data" not in pulse:
            raise KeyError("Pulse configuration missing required \"data\" key.")

        return pulse
    
    def _make_pulse_id_message(self, pulse: Union[str,dict,None] = None) -> str:
        """
        Make a informative string for specifying a pulse in log/error messages
        """
        pulse_config = self.get_pulse_config(pulse)
        pulse_identifier = pulse_config.get("name", make_hash(pulse_config))
        return f"IO '{self._name}' pulse '{pulse_identifier}'"

    def _compute_ramp_flat_memlen_stretchlen(self, pulse_config: dict) -> Tuple[float, float, float, float]:
        '''
        This helper function takes a pulse configuration dictionary and computes the
        ramp, flat, memory length and stretch length. The ramp and flat are fractions 
        of the memory length, and the stretch length is not None only if `use_stretch` is
        set to True in the pulse configuration.
        :param pulse_config: The pulse configuration dictionary, obtained from
            `get_pulse_config` method.
        :return: A tuple containing the ramp, flat, memory length and stretch length.
        '''

        # Prepare common variables
        use_stretch = pulse_config.get("use_stretch", False)
        f = pulse_config.get("flat", 0e-9)
        clock_sample_time = 1 / self._acadia.sequencer_clock_frequency()
        stretch_length = None # Initialize stretch length to None
        
        if use_stretch: # This part computes 'f' appropriately if use_stretch is True
            if f is None or f == 0:
                logger.warning("Warning: flat is None, setting it to minimum value of %.3g seconds.", clock_sample_time)
                f = clock_sample_time
            # Find the closest multiple of clock_sample_time to f
            stretch_factor = np.ceil(f / clock_sample_time)
            # Remember the stretch length
            # TODO: document behavior of adding a flat cycle in the middle to guarantee proper behavior when stretching
            stretch_length = (stretch_factor -  1) * clock_sample_time
            # Passing 0 to the stretch_length variable causes errors
            stretch_length = stretch_length if stretch_length > 0 else None
            f_rounded = round(f / clock_sample_time) * clock_sample_time
            if not math.isclose(f, f_rounded, abs_tol=1e-12):
                logger.warning(f"{self._make_pulse_id_message(pulse_config)} has a flat length that is not a multiple of {clock_sample_time}, and 'use_stretch = True'. "
                                f"The actual flat played will be {f_rounded:.9g} seconds, instead of {f:.9g}")
            f = clock_sample_time # The flat part is always set to clock_sample_time when using stretch

        # If the pulse data is a string, we compute the ramp and flat fractions and         
        if isinstance(pulse_config["data"], str): 
            # TODO: make it ok to accept "length"
            r = pulse_config["ramp"]
            memory_length = pulse_config.get("memory_length", None)
            ml = memory_length # Create new variable to be modified later
            
            if use_stretch:
                if r is None or r == 0:
                    logger.warning(f"Warning: {self._make_pulse_id_message(pulse_config)} ramp is None, and use_stretch = True."
                                   "Your pulse will be zero padded with 5 ns on either side")
                    r = 1e-12 # TODO: update flow to make resulting sizes clear
                # Find the padding necessary to ensure that there is a flat cycle in the middle of the pulse
                tau_pad = np.ceil(r / (2 * clock_sample_time)) * 2 * clock_sample_time - r
                # Adjust memory length
                ml = tau_pad + r + f
                if memory_length is not None:
                    logger.warning(f"Warning: {self._make_pulse_id_message(pulse_config)} memory_length can't be set when using stretch, the value used is {ml:.3g}")
            else:
                if memory_length is None:
                    # Find the closest multiple of clock_sample_time to r + f
                    ml = np.ceil((r + f) / clock_sample_time) * clock_sample_time
                elif (memory_length - r - f) < -1e-12:
                    ml = memory_length
                    logger.warning(f"Warning: {self._make_pulse_id_message(pulse_config)} memory_length {memory_length:.3g} is less than ramp + flat {(r + f):.3g}. Truncating the pulse.")
                elif (memory_length - r - f) > 1e-12:
                    ml = memory_length
                    logger.warning(f"Warning: {self._make_pulse_id_message(pulse_config)}" + "memory_length %s is greater than ramp + flat %.3g. Zero padding the pulse with %.3g seconds.",
                                   memory_length, r + f, ml - r - f)

            # Calculate the ramp and flat lengths as fractions of the memory length
            ramp_frac = r / ml
            flat_frac = f / ml

        else:
            ml = len(pulse_config["data"]) * clock_sample_time/self._samples_per_cycle
            ramp_frac = None
            flat_frac = None

        return ramp_frac, flat_frac, ml, stretch_length

    def prepare_pulse_params(self, pulse_config: Union[str, dict, None] = None) -> None:
        '''
        This function reads the pulse config and populates the cache with the
        pulse params needed to create waveform memory or compute pulse

        :param pulse: The name of the pulse in the configuration, or a dictionary
            containing the configuration of the pulse. If None, the first pulse in
            the configuration is used.
        '''
         # took ~ 90us
        pulse_config = self.get_pulse_config(pulse_config)
        pulse_name = pulse_config["name"]
        pulse_hash = make_hash(pulse_config) # took ~ 70us
        # Don't populate cache if pulse_name is None
        if pulse_name is None:
            return pulse_hash
        # If pulse doesn't exist, create a new entry in the cache
        if pulse_name not in self._pulse_cache:
            self._pulse_cache[pulse_name] = {'waveforms': {}, 'memory': None}
        # If pulse_hash already exists, return
        if self._pulse_cache[pulse_name]['waveforms'].get(pulse_hash) is not None:
            return pulse_hash
        else: # If pulse_hash doesn't exist, compute the pulse parameters
            waveform_container = {}
            ramp_frac, flat_frac, memory_length, stretch_length = self._compute_ramp_flat_memlen_stretchlen(pulse_config)
            waveform_container["ramp_frac"] = ramp_frac
            waveform_container["flat_frac"] = flat_frac
            waveform_container["memory_length"] = memory_length
            waveform_container["stretch_length"] = stretch_length
            self._pulse_cache[pulse_name]['waveforms'][pulse_hash] = waveform_container
            return pulse_hash

    def load_pulse(self,
                   memory: Union[str, Dict, WaveformMemory] = None,
                   pulse: Union[str, Dict, NDArray, float, complex] = None, **kwargs) -> None:
        """
        Compute and load a pulse into the specified waveform memory.
        :param memory: The name of the waveform memory to load the pulse into, or
            a WaveformMemory object that can be loaded.
        :param pulse: The name of the pulse to load, or a dictionary containing
            the pulse config. If a string is provided, it should correspond to a
            pulse name in the configuration. If a dictionary is provided, it should
            contain the pulse configuration, including the "name" key.
        :param kwargs: Additional keyword arguments to be passed to the pulse
            generation function. These will override values specified using a str, or Dict
        """
        # todo: should we still allow this to take `pulse` with a string type? Do we still have a use case for that?

        # Determine the memory to be loaded
        memory_name = None
        # Pulses will not be cached if there is no memory name
        if memory is None:
            memory = list(self.get_config("pulses").keys())[0]
        if isinstance(memory, (str, dict)):
            wfm = self.get_waveform_memory(memory)
            memory_name = memory if isinstance(memory, str) else memory["name"]
        elif isinstance(memory, WaveformMemory):
            wfm = memory
        else:
            raise TypeError("`load_pulse` requires either a WaveformMemory"
                            " object that can be loaded, or a string or a dict that"
                            " specifies a pulse. If it is a string, it should match the name of a pulse"
                            " in the configuration. If it is a dict, it should contain a pulse configuration.")

        # Construct the dictionary to be passed to compute pulse
        if pulse is None:
            if isinstance(memory, (str, dict)):
                # In this case, we can still specifiy the pulse
                # If it is a string, it looks up the YAML file
                # If it is a dict, the dict should contain the pulse configuration
                pulse = memory

            else:
                raise TypeError("You should be knowing what you are doing. memory is a WaveformMemory, and pulse is None. ¯\_(ツ)_/¯")

        if isinstance(pulse, (str, dict)):
            pulse = self.get_pulse_config(pulse).copy()
        else:
            if isinstance(pulse, (float, complex)):
                # If it is a pulse, convert it into an array
                pulse = np.ones(memory.size, dtype=np.complex128) * pulse
            pulse = {"data": pulse}
            # The pulse name is either None, or the name of the memory
            pulse["name"] = memory_name
            # Need to remember length of memory if pulse is a float or complex
        pulse.update(kwargs)

        # `compute_pulse` return samples as integer pairs that have been converted using `complex_to_samples`
        # so we can directly copy it to the memory
        samples = self.compute_pulse(pulse, return_raw=True)
        np.copyto(wfm.array, samples)
        
    def compute_pulse(self, 
                        pulse: Union[str,dict,None] = None, 
                        return_raw = True,
                        **kwargs) -> NDArray:
        """
        This method computes a pulse waveform based on the configuration. This method 
        also caches the computed pulse in the `_pulse_cache` dictionary, so that 
        it can be reused later without recomputing it.
        :param pulse: The name of the pulse in the configuration, or a dictionary
            containing the configuration of the pulse. If None, the first pulse in
            the configuration is used.
        :param return_raw: When True, return the pulse samples as integers that can be directly loaded to
            the memory via `np.copyto`
        :param kwargs: Additional keyword arguments to be passed to the pulse
            generation function. These can include parameters such as `scale`,
            `detune`, `ramp`, `flat`, `use_stretch`, and `memory_length`. The values
            provided here will override the values in the pulse configuration.
        :return: A numpy array containing the computed pulse waveform.
        """
        
        # Get the pulse configuration and its hash
        pulse_config = self.get_pulse_config(pulse).copy()
        pulse_config.update(kwargs)
        scale = pulse_config.get("scale", None)
        phase = pulse_config.get("phase", None)
        detune = pulse_config.get("detune", None)
        use_stretch = pulse_config.get("use_stretch", False)
        if use_stretch and detune is not None:
            raise ValueError("Detune and use_stretch cannot be used together. "
                            "Please use either one or the other.")
        
        if pulse_config["name"] is not None:
            # Look in the cache for the pulse
            pulse_hash = self.prepare_pulse_params(pulse_config)
            waveform_container = self._pulse_cache[pulse_config["name"]]['waveforms'][pulse_hash]
            complex_samples = waveform_container.get("complex_samples")
            raw_samples = waveform_container.get("raw_samples")

            if complex_samples is None:
                # This means the pulse hasn't already been computed and cached
                if scale is not None or detune is not None:
                    # compute waveform_0 (one with no scale or detuning)
                    # so that pulse computation is not repeated
                    # when changing just scale and detuning
                    waveform_0_config = pulse_config.copy()
                    waveform_0_config["scale"] = None
                    waveform_0_config["phase"] = None
                    waveform_0_config["detune"] = None
                    complex_samples = self.compute_pulse(waveform_0_config, return_raw=False)
            
                elif isinstance(pulse_config["data"], str):
                    # If the pulse is a string, we need to compute it
                    ramp_frac = waveform_container["ramp_frac"]
                    flat_frac = waveform_container["flat_frac"]
                    memory_length = waveform_container["memory_length"]
                    num_samples = self._acadia.seconds_to_cycles(memory_length) * self._samples_per_cycle
                    complex_samples = np.zeros(num_samples, dtype = np.complex128)
                    func_kwargs = {k: v for k, v in pulse_config.items() if k not in KWARG_EXCLUDE_LIST}
                    InputOutputWaveforms.flattop_generator(complex_samples, getattr(InputOutputWaveforms, pulse_config["data"]), ramp_frac, flat_frac, **func_kwargs)
                else: # we just use the data directly
                    complex_samples = pulse_config["data"]
                
                complex_samples = InputOutputWaveforms.scale_detune_pulse(complex_samples, scale, phase, detune, self._channel.interface_sample_frequency)
                # cache complex samples for scaling and detuning
                self._pulse_cache[pulse_config["name"]]['waveforms'][pulse_hash]["complex_samples"] = complex_samples

            if raw_samples is None:    
                raw_samples = complex_to_sample(complex_samples)
                # cache raw samples for directly loading to memory
                self._pulse_cache[pulse_config["name"]]['waveforms'][pulse_hash]["raw_samples"] = raw_samples

        else:
            # Pulse without name will not be cached
            complex_samples = pulse_config["data"]
            complex_samples = InputOutputWaveforms.scale_detune_pulse(complex_samples, scale, phase, detune, self._channel.interface_sample_frequency)
            raw_samples = complex_to_sample(complex_samples)
        
        if return_raw:
            return raw_samples
        else:
            return complex_samples
    

    def duplicate_pulse(self, old_pulse:Union[str, dict], new_pulse_name:str=None, 
                        create_memory:bool=False, duplicate_waveforms:bool=False):
        """
        Duplicate an existing pulse configuration and its waveform memory.

        :param old_pulse: The pulse to duplicate, specified by name or config dictionary.
        :param new_pulse_name: The name of the new duplicated pulse. If None, "_copy{idx}" is appended to the old name.
        :param duplicate_waveforms: If True, create the waveform memory for this pulse. 
            Usually this can be False, and `create_waveform_memory` will be called at the compile time.
        :param duplicate_waveforms: If True, duplicate the cached waveform data as well.
        """
        new_config = self.get_pulse_config(old_pulse).copy()
        old_name = new_config.get("name")
        if old_name is None:
            raise ValueError("Cannot duplicate unnamed pulse.")

        # If new pulse name is None, apped _copy (and index)
        if new_pulse_name is None:
            base_name = old_name + "_copy"
            new_pulse_name = base_name
            i = 1
            while new_pulse_name in self._config.get("pulses", {}):
                new_pulse_name = f"{base_name}{i}"
                i += 1
        new_config["name"] = new_pulse_name

        # Insert into config tree (modifies in-place)
        if "pulses" not in self._config:
            self._config["pulses"] = {}
        self._config["pulses"][new_pulse_name] = new_config

        # Create WaveformMemory for this pulse
        if create_memory:
            self.get_waveform_memory(new_pulse_name)

        # Duplicate waveform entries (by copying the waveform container dicts)
        if duplicate_waveforms:
            for h, val in self._pulse_cache.get(old_name, {}).get("waveforms", {}).items():
                self._pulse_cache[new_pulse_name]["waveforms"][h] = val.copy()
        
        return new_pulse_name


    def schedule_pulse(self, pulse: Union[str, Dict, WaveformMemory] = None,
              stretch_length:Union[float, ManagedResource] = None) -> None:
        """
        Schedule a pulse from the IO channel.
        """
        if not self._channel.is_dac:
            raise TypeError(f"`pulse` can only be called from a DAC channel, got {self._channel}")
        if pulse is None:
            pulse = list(self.get_config("pulses").keys())[0]
        
        wfm = self.get_waveform_memory(pulse)
        if stretch_length is None:
            stretch_length = getattr(wfm, "_stretch_length", None)
        else:
            use_stretch = self.get_pulse_config(pulse).get("use_stretch", False)
            if not use_stretch and stretch_length != 0:
                logger.warning(f"{self._make_pulse_id_message(pulse)} had 'use_stretch = True' in the configuration, "
                               f"but a non-zero stretch length: '{stretch_length}' is used in `schedule_pulse`."
                                "This may lead to unexpected results.")
      
        self._acadia.schedule_waveform(wfm, stretch_length=stretch_length)

    # ------------ ADC specific methods ------------
    def capture_cmacc(self, capture_waveform_memory: Union[str, dict, WaveformMemory],
                      kernel: Union[str, dict, WaveformMemory] = None, cmacc_offset: Tuple[int,int]=None,
                      write_mode: Literal["upper", "lower"] = "upper", reset_fifo: bool = False):
        """
        Schedule ADC capture with CMACC (Complex Multiply and Accumulate).

        :param capture_waveform_memory: The capture waveform memory to be used.
        :param kernel: Can be either a string (name of the window in the config), 
            or a dict that specifies the configuration of a window, or a pre-allocated memory for the kernel.
        :param cmacc_offset: Preloaded offset for the CMACC kernel, as a tuple of two integers.
            If not None, this will overwrite the value in the config dict.
        :param write_mode: The output port of the CMACC is driven by a multiplexer,
            which allows the user to decide which data is written into memory. 
            The output port has 32 bits per quadrature, but the internal accumulator
            has 48 bits per quadrature, and therefore it is left to the user to
            decide whether the upper or lower 32 bits are presented to the output.
            Using the upper 32 bits reduces the precision of the output but reduces
            the probability of overflow.
        :type write_mode: str, one of "upper", "lower".
        """

        if self._channel.is_dac:
            raise TypeError(f"`capture_trace` can only be called from an ADC channel, got {self._channel}")

        capture_waveform = self.get_waveform_memory(capture_waveform_memory)

        # If kernel is specifyed with a key in yaml or just a dict, allocate the memory anew 
        # We'll interpret the type of the argument slightly differently,
        # than configure_cmacc expects; if the window argument is a float,
        # we'll interpret this as the amplitude of a boxcar window, in contrast
        # to configure_cmacc interpreting a float argument as the length in seconds
        # that the window should be. The behavior here will be different as this is a
        # much more likely use case and will be much more intuitive in the config file
        # The actual amplitude/data of the window will be set when calling `window_memory.load()`
        if isinstance(kernel, (str, dict)):
            window_config = self.get_config("windows", kernel) if isinstance(kernel, str) else kernel
            window_data = window_config["data"]
            kernel_arg = None if isinstance(window_data, float) else window_data
            if cmacc_offset is None:
                cmacc_offset = window_config.get("offset", (0, 0))

        else: # we will let stream_cmacc to handle the other types 
            kernel_arg = kernel
        
        # Because the offset is just a number (and not a thing that needs
        # to be allocated), we can just get it from the config every time
        offset_converted = [int(np.int32(q).astype(np.uint32)) for q in cmacc_offset]


        # if kernel is None or a boxcar, the accumulation length will be set to capture length
        if kernel_arg is None or (hasattr(kernel_arg, "size") and kernel_arg.size == 1):
            length = self.get_config("memories", capture_waveform_memory, "length")
        else: # otherwise, stream_cmacc will use kernel length as the accumulation length
            length = None

        stream, window_mem = self._acadia.stream_cmacc(self._channel, capture_waveform,
                                                                      kernel=kernel_arg, length=length,
                                                                      preload=offset_converted,
                                                                      write_mode=write_mode, reset_fifo=reset_fifo,
                                                                      last_only=True, accumulator_done=False)    
        
        return stream, window_mem

    def capture_trace(self, capture_waveform_memory: Union[str, dict, WaveformMemory], decimation:int=None):
        """
        Capture a IQ trace from the ADC channel into the specified waveform memory.

        :param capture_waveform_memory: Waveform memory to store the captured trace,
            can be a string (name of the memory), a dict (configuration of the memory),
            or a `WaveformMemory` object.
            If a string or dict is provided, it should match the name of a memory in the
            configuration under the "memories" section.

        :param decimation: When not None, this value overrides the decimation factor
            specified in the waveform memory configuration.
            If `capture_waveform_memory` is a `WaveformMemory` object, this parameter must be specified, because
            the decimation factor is not stored in the `WaveformMemory` object.
        """
        if self._channel.is_dac:
            raise TypeError(f"`capture_trace` can only be called from an ADC channel, got {self._channel}")
        
        if isinstance(capture_waveform_memory, str):
            deci = self.get_config("memories", capture_waveform_memory, "decimation")
            decimation = decimation if decimation is not None else deci
        elif isinstance(capture_waveform_memory, dict):
            deci = capture_waveform_memory.get("decimation")
            decimation = decimation if decimation is not None else deci
        
        if decimation is None:
            raise ValueError("Decimation must be secified (Either in the yaml, input dict, or input to this function)"
                            f"for capture trace with {capture_waveform_memory}")

        src = self._channel
        dst  = self.get_waveform_memory(capture_waveform_memory)

        dsp_stream_config = self._acadia.configure_dsp(src=src, decimation=decimation, reset=True)

        if decimation == 0:
            raise ValueError("Decimation factor cannot be zero for capture trace.")

        length_cycles = dst._shape[0]
        output_size_bytes = length_cycles * (2*dst.itemsize) 

        if decimation > 1:# the decimation % 4 check should have already been done in `create_wageform_memory`
            # when decimate, the capture cycles need to be scaled to capture the full trace
            length_cycles = length_cycles * (decimation //4)

        capture_dict = self.get_waveform_memory("readout_trace").__dict__
        logger.info(f"output_size_bytes {output_size_bytes}, capture_dict: {capture_dict}, length_cycles: {length_cycles}")

        self._acadia.stream(configuration=dsp_stream_config, 
                    dst=dst,
                    length_cycles=length_cycles,
                    output_size_bytes=output_size_bytes,
                    memory_input=(src if dsp_stream_config.input_source == "memory" else None))


    # ------------Generic IO methods ------------
    def get_waveform_memory(self, waveform_memory: Union[str, dict, WaveformMemory] = None) -> WaveformMemory:
        """
        A shortcut function for getting waveform for a specific pulse.
        Serves as a shortcut for creating waveform memory objects based on the
        parameters defined in ``config["pulses"]``.
        If no name is provided, the first memory in the list of available
        memories is used.
        :param waveform_memory: Name of the waveform memory defined under the
            "pulses" sub-dictionary for the given channel
        :return: Allocated WaveformMemory object
        """
        # Just return the input if we were directly given a memory
        # This is useful behavior for writing objects later that, when calling this
        # function to retrieve a waveform, can be directly passed either a name or
        # an actual memory

        if isinstance(waveform_memory, WaveformMemory):
            return waveform_memory
        if waveform_memory is not None and not isinstance(waveform_memory, (str, dict)):
            raise TypeError(f"Waveform memories must be specified by string or dictionary"
                            f" when not directly provided or inferred;"
                            f" received memory specified of type"
                            f" {type(waveform_memory)}")

        if self._channel.is_dac: # If it is a dac, check in the pulse cache
            if waveform_memory is None:
                waveform_memory = list(self.get_config("pulses").keys())[0]
            
            wfm_name = waveform_memory["name"] if isinstance(waveform_memory, dict) else waveform_memory
            if (self._pulse_cache.get(wfm_name) is None) or (self._pulse_cache[wfm_name].get("memory") is None):
                pulse_config = self.get_pulse_config(waveform_memory)
                pulse_hash = self.prepare_pulse_params(pulse_config)
                memory_length = self._pulse_cache[wfm_name]['waveforms'][pulse_hash]["memory_length"]
                stretch_length = self._pulse_cache[wfm_name]['waveforms'][pulse_hash]["stretch_length"]
                wfm = self._acadia.create_waveform_memory(self._channel, length = memory_length)
                setattr(wfm, "_stretch_length", stretch_length)
                self._pulse_cache[wfm_name]["memory"] = wfm
            else:
                wfm = self._pulse_cache[wfm_name]["memory"]
        else: # If it is an adc, check in allocated memories
            if waveform_memory is None:
                waveform_memory = list(self.get_config("memories").keys())[0]
            if isinstance(waveform_memory, dict):
                raise TypeError("Waveform memory must be specified by string or WaveformMemory object"
                                " for ADC channels; received a dictionary.")
            if waveform_memory not in self._allocated_memories:
                wf_cgf = self.get_config("memories", waveform_memory)
                wf_cgf = {k: v for k, v in wf_cgf.items() if k != "stretch_length"}
                self._allocated_memories[waveform_memory] = self._acadia.create_waveform_memory(self._channel, **wf_cgf)

            wfm = self._allocated_memories[waveform_memory]
        
        return wfm
    
    def set_frequency(self, frequency: float, sync: bool = True):
        """
        Update the frequency of the IO channel and optionally trigger a synchronization.
        """

        self.set_nco_frequency(frequency)
        self.reset_nco_phase()
        if sync:
            self._acadia.update_ncos_synchronized()

    def dwell(self, length: Union[float, Symbol, Operation] = None, length_is_minus_one: bool = False) -> None:
            """
            Schedule a dwell on a channel's DMA.

            :param channel: Channel to stream
            :type channel: :class:`Channel`
            :param length: The length of the dwell.
                When a `float` is provided, this is the number of seconds for the dwell.
                When a sequencer source is provided, no conversion is performed; it is
                assumed that the source contains the length in units of cycles.
            :type length: float, Register, DSP
            :param length_is_minus_one: If ``True``, the provided length is understood
                to be one less than the actual length to be included in the command.
            :type length_is_minus_one: bool
            """
            self._acadia.dwell(self.channel, length, length_is_minus_one)

    @property
    def channel(self) -> Channel:
        return self._channel

    def __getattr__(self, name: str):
        """
        For any attribute not defined here, access the underlying channel object
        so that we can call Channel methods directly on the InputOutput object
        """

        return getattr(self._channel, name)
     
class IOConfig:
    """
    This class is intended to be a placeholder provided in type hints for 
    QMsmtRuntime subclasses; behavior is described in its docstring. 
    """
    pass

class QMsmtRuntime(Runtime):
    """
    A base class that provides shortcut functions that simplifies the 
    configuration of analog channels and waveform memories based on the 
    channel configurations in provided fields (where fields are defined
    in the definition of :class:`Runtime`). This will be the base class 
    of other runtime classes that use configuration based on a yaml file.

    To indicate that a field of the Runtime dataclass is a channel configuration,
    it must be a have the type hint :class:`IOConfig`. This indicates that the
    corresponding channel should be configured when the Runtime is instructed
    to configure all channels, and also provides some flexibility in how 
    channel configurations are provided to the runtime:

        - If the value of the field is a dict, this dict is provided to the
        InputOutput object for configuring the channel.

        - If the value of the field is a string, it is interpreted as the name
        of a top-level section in a YAML file. By default, this file is assumed to 
        be "config.yaml" in the current working directory. This default can be 
        overridden by specifying  `yaml_path` as a keyword argument when initializing
        the child runtime. The configuration for the channel is then retrieved by 
        loading the file and accessing the corresponding section.

        - If the value of the field is a tuple, it is expected that the first
        element is the name of a top-level section in a yaml file whose path 
        is provided in the second element. In this case, the `yaml_path` provided
        in the child runtime will be ignored for this io.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.acadia = Acadia()  

        # Store the names and config dicts for all the 
        # members identified as channel configurations
        self._ios: Dict[str,InputOutput] = {}
        self._yaml_paths = {} # collect yaml files for copying to local data directory 
        for name,type_hint in self._get_fields().items():
            if type_hint is IOConfig:
                value = getattr(self, name)
                
                # --- IOConfig is specified via a YAML key ---
                if isinstance(value, (tuple, str)): 
                    from acadia_qmsmt.utils.yaml_editor import load_yaml
                    if isinstance(value, tuple):
                        yaml_key, yaml_path  = value
                    else:
                        yaml_key = value
                        # Allow child classes to pass `yaml_path=None` and still get the default "config.yaml"
                        yaml_path = kwargs.get("yaml_path")  or "config.yaml"

                    config_dict = load_yaml(yaml_path)[yaml_key]

                    # record the yaml file’s *absolute* path and original channel key so yaml update functions 
                    # can still work for reloaded runtimes (because these will be saved in kwargs.json)
                    config_dict["__yaml_path__"] = str(Path(yaml_path).absolute())
                    config_dict["__yaml_key__"] = yaml_key

                    # collect yaml files for copying to local data directory 
                    self._yaml_paths[name] = (config_dict["__yaml_path__"], yaml_key)                    

                # ---  IOConfig is provided as a dict directly ---
                elif isinstance(value, dict):
                    config_dict = value

                else:
                    raise TypeError(f"Unable to interpret value of type"
                                    f" {type(value)} as an IO configuration")

                self._ios[name] = InputOutput(name, self.acadia, config_dict)

        
    def _dump_fields(self, fields: dict = None):
        """
        Dump all the arguments to a JSON file as in the parent class, but replace
        IOConfig arguments with their config dicts, and also save the yaml file(s) 
        to the data folder.
        """ 
        fields = {}
        for name,type_hint in self._get_fields().items():
            if type_hint is IOConfig:
                fields[name] = self._ios[name]._config
            else:
                fields[name] = getattr(self, name)

        self._copy_yamls()

        super()._dump_fields(fields)


    def _copy_yamls(self):
        """
        Copy over YAML files to the local data directory.

        We need to be careful in cases where different IOs use YAML files from different paths BUT with the
        same filename. To avoid overwriting, we add subfix when copying such files.

        Since information like `io._config['yaml_path']` will be gone when loading back a finished experiment from
        the data folder, an extra `yaml_key_index.json` file is created in the local directory to look up for yaml
        files and keys used for initializing each IO.
        """

        copied_files = {}  # {`original_yaml_path` : `local_yaml_name`}
        io_yaml_index = {}  # {`io_name`: (`local_yaml_name`, `key_in_yaml`)}
        dest_dir = Path(self.local_directory)

        for io_name, (yaml_path, yaml_key) in self._yaml_paths.items():
            if yaml_path is None:
                continue

            base_name = Path(yaml_path).name
            local_yaml_name = base_name

            # Check for filename conflict and resolve with suffix
            counter = 1
            while (local_yaml_name in copied_files) and (yaml_path != copied_files[local_yaml_name]):
                stem = Path(base_name).stem
                suffix = Path(base_name).suffix
                local_yaml_name = f"{stem}_{counter}{suffix}"
                counter += 1

            # Copy if not already copied
            if local_yaml_name not in copied_files:
                shutil.copy2(yaml_path, dest_dir / local_yaml_name)
                copied_files[local_yaml_name] = yaml_path

            io_yaml_index[io_name] = (local_yaml_name, yaml_key)

        with open(dest_dir / "yaml_key_index.json", "w") as f:
            json.dump(io_yaml_index, f, indent=4)


    def _prepare_files(self, files, runtime_module, log_level, finalization_time):
        """
        Gather the additional files needed to send to the host, which are defined to be
        the file defining QMsmtRuntime and the file defining the subclass
        """

        # We will automatically determine the runtime module to use and yell 
        # at the user if they provide their own
        if runtime_module is not None:
            raise ValueError(f"Runtime module must not be provided for a QMsmtRuntime")
        
        # The local directory exists already so we can just copy the needed files into it
        # Copy the runtime file and rename it to runtime.py
        runtime_file = sys.modules[self.__class__.__module__].__file__
        shutil.copy(runtime_file, os.path.join(self.local_directory, "runtime.py"))

        # Copy this file (that is, the one defining QMsmtRuntime) and rename it to 
        # acadia_qmsmt.py so that it can be imported on the target without installing the 
        # package
        shutil.copy(__file__, os.path.join(self.local_directory, "acadia_qmsmt.py"))

        # This will create the directory on the target as well as prepare everything else
        # We specify the module name "runtime" because below we'll copy the file defining
        # the runtime and rename it to "runtime.py"
        super()._prepare_files(files, "runtime", log_level, finalization_time)


    def configure_channels(self, 
                        nco_update_event_source: Literal["sysref", "immediate"] = "sysref",
                        reset_phases: bool = True, 
                        align_tiles: bool = True):
        """
        Automatically configures NCO (Numerically Controlled Oscillator) parameters for channels in the `configs` dict.

        :param nco_update_event_source: The source of the NCO update event. Should be either:
                                        - "sysref" (default): Updates are synchronized to a system reference.
                                        - "immediate": Updates occur immediately without synchronization.
        :param reset_phases: When True, reset NCO phases to 0 after configuration.
        :param align_tile: When True, align tile latencies before configuration
        """
        if align_tiles:
            self.acadia.align_tile_latencies()
        for name,io in self._ios.items():
            # Configure analog parameters for each channel
            io.channel.set(nco_update_event_source=nco_update_event_source, **io.get_config("channel_config"))
            if reset_phases:
                self.acadia.reset_nco_phase(io.channel)
                self.acadia.update_nco_phase(io.channel, 0)
            if nco_update_event_source == "immediate":
                io.channel.nco_immediate_update_event()
                 # some how this is needed to ensure the update event
                io.channel.nco_immediate_update_event()

        if nco_update_event_source == "sysref":
            self.acadia.update_ncos_synchronized()


    def io(self, channel_name: str) -> InputOutput:
        return self._ios[channel_name]

    def update_ioconfig(self, ioconfig_name: str, config_field: str, value: Any):
        """
        Update the value of a field in the config file for the specified IOConfig.
        If the IOConfig was populated with a dict, no files are modified, but the
        dict will be updated.
        """

        ioconfig = getattr(self, ioconfig_name)
        if isinstance(ioconfig, dict):
            # Update the dict
            # Split the provided path, populating a list with a sequence
            # of keys which we need to use to recusrsively index into the 
            # config dict of the IO
            index_paths = config_field.split(".")

            # Walk the index tree of the dict until we get to the lowest level 
            # of the provided field path
            cfg = self._ios[ioconfig_name]._config
            while len(index_paths) > 1:
                cfg = cfg[index_paths.pop(0)]

            # Finally, actually assign the value
            cfg[index_paths[0]] = value
        elif isinstance(ioconfig, str):
            # Update the yaml
            self.update_io_yaml_field(ioconfig_name, config_field, value)
        else:
            raise TypeError(f"Unable to update IO config when provided as type {type(ioconfig)}")

    def get_io_yaml_key(self, ioconfig_name: str) -> str:
        """
        Get the YAML key that identifies the IO channel configuration, if it was initialized
        from a YAML entry. This is useful when reloading a runtime from a completed
        experiment's data folder.

        :param ioconfig_name : The attribute name of the IO configuration.
        """
        ioconfig = getattr(self, ioconfig_name)

        if isinstance(ioconfig, str):
            return ioconfig
        elif isinstance(ioconfig, dict):
            if "__yaml_key__" not in ioconfig:
                raise KeyError(f"Missing '__yaml_key__' in IO config dict for '{ioconfig_name}'")
            return ioconfig["__yaml_key__"]
        else:
            raise TypeError(f"Can't find the yaml key for '{ioconfig_name}', it was probably not"
                            f"initialized with a key in a yaml file")
        
    def update_io_yaml_field(self, ioconfig_name: str, config_field: str, value: Any, verbose=False):
        """
        Update the value of a field in the yaml config file for the specified IOConfig.
        """
        from acadia_qmsmt.utils.yaml_editor import update_yaml
        yaml_path = self._ios[ioconfig_name]._config["__yaml_path__"]
        yaml_key = self.get_io_yaml_key(ioconfig_name)

        new_cfg = update_yaml(yaml_path, {f"{yaml_key}.{config_field}": value}, verbose=verbose)
        logger.info(f"!! updated yaml file `{yaml_path}`: {yaml_key}.{config_field}: {value}")
        return new_cfg

    
    def wait_for_deploy_completion(self, suppress_data_sync_warnings: bool = True):
        """
        wait for the deployment to complete by joining the event loop.
        :param suppress_data_sync_warnings: If True, suppress warnings related to data synchronization.
        """
        from acadia_qmsmt.utils import suppress_data_sync_messages
        try:
            with suppress_data_sync_messages(suppress_data_sync_warnings):
                self._event_loop.join()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught. Stopping...")
            self.stop()
            self.finalize()
            logger.info("Cleanup complete.")

    def deploy(self, *args, no_backup: bool = False, **kwargs):
        """
        Call Runtime.deploy with the same arguments and then create a
        '.no_backup_flag' file in the resulting local data directory.
        If `no_backup` is True, this file will be created to indicate that
        This is useful for runs that should not be backed up.
        """

        super().deploy(*args, **kwargs)

        # Create the flag file in the local data directory
        if no_backup:
            try:
                data_dir = Path(self.local_directory)
                flag_path = data_dir / ".no_backup_flag"
                # Write a simple message to the file
                flag_path.write_text("Exclude this run from backups.\n", encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to create .no_backup_flag in %s: %s", getattr(self, "local_directory", "?"), e)

        


class MeasurableResonator:
    """
    A collection of functions that are useful for interacting with a resonator
    whose response can be directly measured (as opposed to something like a
    storage cavity, which can be driven but not directly measured).
    """

    def __init__(self, stimulus: InputOutput, capture: InputOutput):
        self._stimulus = stimulus
        self._capture = capture
        self._stream = None
        self._windows = {}
        self._classifiers = {}
 
    def measure(self, 
                stimulus_waveform_memory: Union[str, WaveformMemory] = None, 
                capture_waveform_memory: Union[str, WaveformMemory] = None,
                window_name: str = None,
                cmacc_offset: Tuple[int,int]=None,
                capture_delay:float=0,
                write_mode: Literal["upper", "lower"] = "upper",
                reset_fifo: bool = False
                ):
        """
        Schedules the measurement of accumulated IQ points. 
        This function should be called inside of a channel synchronizer.

        :param stimulus_waveform_memory: The pulse waveform memory to be played.
        :param capture_waveform_memory: The capture waveform memory to be used.
        :param window_name: Name of the window to be used for the CMACC kernel.
        :param cmacc_offset: Preloaded offset for the CMACC kernel, as a tuple of two integers.
            If not None, this will overwrite the value in the config dict.
        :param capture_delay: Delaytime between scheduling the pulse and starting the capture, in seconds.
        :param write_mode: The output port of the CMACC is driven by a multiplexer,
            which allows the user to decide which data is written into memory. 
            The output port has 32 bits per quadrature, but the internal accumulator
            has 48 bits per quadrature, and therefore it is left to the user to
            decide whether the upper or lower 32 bits are presented to the output.
            Using the upper 32 bits reduces the precision of the output but reduces
            the probability of overflow.
        :type write_mode: str, one of "upper", "lower".

        """
        if window_name is None:
            window_name = list(self._capture.get_config("windows").keys())[0]

        if window_name in self._windows:
            # Regardless of what the window is, if we have it already,
            # we'll pass in its WaveformMemory so that we don't re-allocate it
            kernel_arg = self._windows[window_name]
        else:
            kernel_arg = window_name
        
        if cmacc_offset is None:
            cmacc_offset = self._capture.get_config("windows", window_name).get("offset", (0, 0))

        # ------------ schedule drive pulse ------------------------------
        self._stimulus.schedule_pulse(stimulus_waveform_memory)
        if capture_delay > 0:
            self._capture.dwell(capture_delay)

        # ----------- configure and schedule ADC capture with CMACC --------------
        self._stream, window_mem = self._capture.capture_cmacc(capture_waveform_memory, kernel_arg, cmacc_offset,
                                                              write_mode=write_mode, reset_fifo=reset_fifo)
        
        # # Cache the relevant stream and window
        # If we had already allocated and cached these, this will just store 
        # them back with the same values
        self._windows[window_name] = window_mem


    def measure_trace(self, 
                stimulus_waveform_memory: Union[str, WaveformMemory] = None, 
                capture_waveform_memory: Union[str, WaveformMemory] = None,
                capture_delay:float=0):
        """
        Schedules the measurement of raw IQ traces. 
        This function should be called inside of a channel synchronizer.
        
        :param stimulus_waveform_memory: The pulse waveform memory to be played.
        :param capture_waveform_memory: The capture waveform memory to be used.
        :param capture_delay: Delaytime between scheduling the pulse and starting the capture, in seconds.

        """

        # ------------ schedule drive pulse ------------------------------
        self._stimulus.schedule_pulse(stimulus_waveform_memory)
        if capture_delay > 0:
            self._capture.dwell(capture_delay) 

        # ------------ configure and schedule capture stream ---------------------
        self._capture.capture_trace(capture_waveform_memory)


    def get_measurement(self, 
                        classifier: Literal["quadrant", "re_sign", "im_sign"]="quadrant") -> Union[int, Operation]:
        """
        Retrieves a measurement from the CMACC in the given format.

        The CMACC can provide its result in various formats, and it is up to the 
        user to decide which format is most useful for a given sequence. For 
        example, in some situations one might want to know which quadrant an 
        accumulated point lies in, and in other situations, one might want to 
        know simply whether a single quadrature is positive or negative. Alternatively,
        one might want a full 32-bit accumulator quadrature value.
        While these all may seem very similar, the hardware provides different ways 
        of accessing these values in order to minimize latency, and this requires reading 
        the result from different CMACC registers.

        The argument ``format`` should specify the type of measurement data to retrieve 
        to use. The allowed values for ``format`` are:

        - ``"quadrant"``: This method will return one of Acadia.CMACC_QUADRANT_1/2/3/4 
            depending on the quadrant of the integrated complex number.

        - ``"re_sign"``: This method will return 0 when the value of the real quadrature 
            is positive, and 0x80000000 when it's negative.

        - ``"im_sign"``: This method will return 0 when the value of the imaginary
            quadrature is positive, and 0x80000000 when it's negative.

        This blocks until the measurement is complete, as indicated by the CMACC
        completion status bit.
        """

        # All of the acadia methods here will wait until the CMACC is complete
        if classifier == 'quadrant':
            return self._capture._acadia.cmacc_get_quadrant(self._stream)
        elif classifier == 're_sign':
            return self._capture._acadia.cmacc_get_quadrature(self._stream) & 0x80000000
        elif classifier == 'im_sign':
            return self._capture._acadia.cmacc_get_quadrature(self._stream, imag=True) & 0x80000000
        else:
            raise ValueError(f"Unknown measurement classifier {classifier}")

    def classify_measurement(self, data: NDArray, classifier_name: str = None) -> NDArray:
        """
        Classifies measurement data.

        Classifiers are algorithms for converting analog measurements 
        into discrete results. Unlike in the sequencer where the only available 
        real-time classifiers are those implemented in the CMACC, offline classifiers 
        may be arbitrarily complicated.

        Each available classifier is specified as an element in the readout capture's 
        "classifiers" section. If the provided classifier name is ``None``, the first 
        available classifier is used. If there are no classifiers, a 
        :class:`MaximalVarianceAxisClassifier` is created.
        """

        from acadia_qmsmt.analysis.measurement_classifiers import create_classifier_from_config

        # First we'll check if we can just load a default
        if classifier_name is None and "classifiers" in self._capture._config:
            classifier_name = list(self._capture.get_config("classifiers").keys())[0]

        if classifier_name in self._classifiers:
            # If we already loaded this classifier, use it
            classifier = self._classifiers[classifier_name]

        elif "classifiers" in self._capture._config:
            # We haven't loaded the classifier but there is a classifier configuration
            classifier_config = self._capture.get_config("classifiers", classifier_name)
            classifier = create_classifier_from_config(classifier_config)
            self._classifiers[classifier_name] = classifier
            
        else:
            # We haven't specified a classifier and there aren't any in the config
            # By default we'll create a MaximalVarianceAxisClassifier, and we won't
            # cache it because it's unsupervised so it needs to be retrained every
            # time
            logger.warning(f"`classifiers` is not specified in either MeasurableResonator or"
                           f" the config of capture channel `{self._capture._name}`. "
                           f"Using `MaximalVarianceAxisClassifier`")
            classifier_config = {"type": "MaximalVarianceAxisClassifier"}
            classifier = create_classifier_from_config(classifier_config)

        return classifier.classify(data)


    def load_windows(self) -> None:
        """
        Load the CMACC window memory with the values specified in the configuration.
        
        This should occur after the sequence has been compiled and the relevant
        Acadia object attached.
        """

        for window_name, window_memory in self._windows.items():
            window_memory.load(self._capture.get_config("windows", window_name, "data"))

    def set_frequency(self, frequency: float, sync: bool = True):
        """
        The frequencies of both the stimulus and the capture are updated and 
        optionally synchronized.
        """

        self._stimulus.set_nco_frequency(frequency)
        self._capture.set_nco_frequency(frequency)
        self._stimulus.reset_nco_phase()
        self._capture.reset_nco_phase()
        if sync:
            self._stimulus._acadia.update_ncos_synchronized()
            self._stimulus._acadia.update_ncos_synchronized()# somehow this is needed to ensure that the nco update is actaully finished, and simply wating for extra 5us does not work

    def wait_until_measurement_done(self):
        """
        block the sequencer until the measurement is complete.
        :return:
        """
        a = self._capture._acadia
        with a.sequencer().repeat_until(a.cmacc_done(self._stream)):
            pass

    def load_pulse(self, pulse_name:str):
        """
        Load a pulse into the stimulus.
        """
        self._stimulus.load_pulse(pulse_name)

class Qubit:
    """
    A collection of functions that are useful for manipulating and measuring a qubit.
    """

    def __init__(self, stimulus: InputOutput, readout_resonator: MeasurableResonator = None):
        """
        Runtime class that contains experiment specific functions, including sub_sequence functions
        """
        self._stimulus = stimulus
        self.readout_resonator = readout_resonator

    def set_frequency(self, frequency: float, sync: bool = True):
        """
        Shortcut function for InputOutput.set_frequency
        """

        self._stimulus.set_frequency(frequency, sync)

    def make_rotation_pulse(self, polar_deg:float, azim_deg:float, 
                            reference_pi_pulse:Union[str, dict]="R_x_180",
                            pulse_name:str=None, create_memory=False):

        # duplicate the pi pulse config
        stimulus = self._stimulus
        new_config = stimulus.get_pulse_config(reference_pi_pulse).copy()

        # scale the pulse
        original_scale = new_config["scale"]
        scale = original_scale * polar_deg/180 * np.exp(1j*azim_deg/180*np.pi)
        new_config["scale"] = scale

        # set pulse name
        if pulse_name is None:
            pulse_name = f"rotation_{polar_deg}_{azim_deg}"
        base_name = pulse_name
        i=1
        while pulse_name in stimulus._config.get("pulses", {}):
            pulse_name = f"{base_name}_{i}"
            i += 1
        new_config["name"] = pulse_name


        # Insert into config tree (modifies in-place)
        stimulus._config["pulses"][pulse_name] = new_config

        # Create WaveformMemory for this pulse
        if create_memory:
            stimulus.get_waveform_memory(pulse_name)
        
        return pulse_name

    def make_selective_pulse(self, cavity_state:int, chi: float, 
                             chi_prime: float=0.0,
                             reference_pi_pulse:str="R_x_180_selective", 
                             pulse_name:str=None,
                             create_memory:bool=False):
        # duplicate the pi pulse config
        stimulus = self._stimulus
        new_config = stimulus.get_pulse_config(reference_pi_pulse).copy()

        # scale the pulse
        new_config["detune"] = cavity_state * chi + 0.5*cavity_state*(cavity_state-1)*chi_prime

        # set pulse name
        if pulse_name is None:
            pulse_name = f"selective_{cavity_state}"
        base_name = pulse_name
        i=1
        while pulse_name in stimulus._config.get("pulses", {}):
            pulse_name = f"{base_name}_{i}"
            i += 1
        new_config["name"] = pulse_name


        # Insert into config tree (modifies in-place)
        stimulus._config["pulses"][pulse_name] = new_config

        # Create WaveformMemory for this pulse
        if create_memory:
            stimulus.get_waveform_memory(pulse_name)
        
        return pulse_name
    
    def prepare(self, 
                state_quadrant: Literal[1,2,3,4],
                measurement_resonator: MeasurableResonator = None,
                pulse_waveform_memory: str = None,
                measurement_stimulus_waveform_memory: str = None,
                measurement_capture_waveform_memory: str = None,
                measurement_cmacc_window: str = None,
                measurement_post_delay: float = 2e-6,
                state_register = None) -> None:
        """
        A subsequence that prepares the qubit in a given state by measuring and
        conditionally applying a waveform. All mneasurement names are passed 
        directly to :meth:`InputOutput.get_waveform_memory`; see documentation therein 
        for argument behavior.

        :param state_quadrant: The quadrant (in [1,2,3,4]) where the target state blob fall in.
        :param measurement_resonator: The resonator to be used for measurement
            and conditional operations.
        :param pulse_waveform_memory: The name of the waveform memory to be played when 
            the qubit is measured to be in any other state than the target. 
        :param measurement_stimulus_waveform_memory: The name of the measurement
            resonator stimulus waveform.
        :param measurement_capture_waveform_memory: The name of the measurement
            resonator capture waveform.
        :param measurement_cmacc_window: CMACC window for the measuremnt 
        :param measurement_post_delay: delay after measurement before starting the next pulse sequnece, in seconds

        """
        # raise NotImplementedError("This function currently appears to be buggy. To avoid unexpected behavior, use a long `run_delay` for now")
        
        measurement_resonator = measurement_resonator or self.readout_resonator
        if measurement_resonator is None:
            raise ValueError("measurement resonator must be provided for qubit.prepare")

        a = self._stimulus._acadia
        quadrant_reg_value = getattr(a, f"CMACC_QUADRANT_{state_quadrant}")
        reg = a.sequencer().Register() if state_register is None else state_register


        # do an initial msmt
        with a.channel_synchronizer():
            measurement_resonator.measure(measurement_stimulus_waveform_memory,
                                        measurement_capture_waveform_memory,
                                        measurement_cmacc_window)
        
        reg.load(measurement_resonator.get_measurement())

        # wait for readout to empty
        with a.channel_synchronizer():
            self._stimulus.dwell(measurement_post_delay)


        ## Measure + conditional flip, until we get the target state
        # todo: try getting the number of msmts in this loop using a counter register
        with a.sequencer().repeat_until(reg == quadrant_reg_value):
            with a.channel_synchronizer():
                self._stimulus.schedule_pulse(pulse_waveform_memory)
                # self._stimulus.dwell(100e-9)
                a.barrier()
                measurement_resonator.measure(measurement_stimulus_waveform_memory,
                                              measurement_capture_waveform_memory,
                                              measurement_cmacc_window)
            with a.channel_synchronizer():
                self._stimulus.dwell(measurement_post_delay)

            reg.load(measurement_resonator.get_measurement())
        
        return reg
    
    def conditional_pulse(self, 
            state_quadrant: Literal[1,2,3,4],
            measurement_resonator: MeasurableResonator = None,
            qubit_pulse_if_true: Union[str, Dict, WaveformMemory] = None,
            qubit_pulse_if_false: Union[str, Dict, WaveformMemory] = None,
            measurement_pulse: Union[str, Dict, WaveformMemory] = None,
            measurement_capture_waveform_memory: str = None,
            measurement_cmacc_window: str = None,
            measurement_post_delay: float = 2e-6,
            state_register = None) -> None:
        
        """
        A subsequence that applies a pulse on the qubit conditional on a measurement

        :param state_quadrant: The quadrant (in [1,2,3,4]) where the target state blob falls in.
            If it falls in the target quadrant, no pulse is applied.
        :param measurement_resonator: The resonator to be used for measurement
            and conditional operations.
        :param qubit_pulse_if_true: The pulse to be played on the qubit when the measurement
            is in the target quadrant.
        :param qubit_pulse_if_false: The pulse to be played on the qubit when the measurement
            is not in the target quadrant.
        :param measurement_pulse: The pulse to be played on the measurement resonator.
        :param measurement_capture_waveform_memory: The name of the measurement
            resonator capture waveform. Note that the result of the measurement
            can be obtained from here for further processing.
        :param measurement_cmacc_window: CMACC window for the measuremnt 
        :param measurement_post_delay: delay after measurement before starting the next pulse sequnece, in seconds
        :param state_register: The register to store the measurement result. If None, a new register will be created.

        returns: The register containing the final measurement result.
        """

        measurement_resonator = measurement_resonator or self.readout_resonator
        if measurement_resonator is None:
            raise ValueError("measurement resonator must be provided for qubit.prepare")

        a = self._stimulus._acadia
        quadrant_reg_value = getattr(a, f"CMACC_QUADRANT_{state_quadrant}")
        reg = a.sequencer().Register() if state_register is None else state_register

        # do an initial msmt
        with a.channel_synchronizer():
            measurement_resonator.measure(measurement_pulse,
                                        measurement_capture_waveform_memory,
                                        measurement_cmacc_window)
        
        reg.load(measurement_resonator.get_measurement())

        # wait for readout to empty
        with a.channel_synchronizer():
            self._stimulus.dwell(measurement_post_delay)
    
        # Apply no pulse if in specified quadrant
        # Note that both the speculations are set to True
        # This can help ensure that both branches take similar time
        with a.sequencer().test(reg == quadrant_reg_value):
            with a.channel_synchronizer():
                self._stimulus.schedule_pulse(qubit_pulse_if_true)
                
        # Apply pulse if not in specified quadrant
        with a.sequencer().test(reg != quadrant_reg_value):
            with a.channel_synchronizer():
                self._stimulus.schedule_pulse(qubit_pulse_if_false)

        return reg

    def schedule_pulse(self, waveform_memory: Union[str, WaveformMemory] = None,
                stretch_length:Union[float, ManagedResource] = None, **kwargs) -> None:
        """
        Shortcut function for InputOutput.schedule_pulse
        """

        self._stimulus.schedule_pulse(waveform_memory, stretch_length, **kwargs)

    def load_pulse(self, pulse_name:str, **kwargs):
        """
        Load a pulse into the stimulus.
        """
        self._stimulus.load_pulse(pulse_name, **kwargs)

    def dwell(self, length: float):
        """
        Dwell for a specified duration.
        """
        self._stimulus.dwell(length)

class QubitQmCooler:
    """
    A collection of functions that are useful for cooling a qubit and QM(s)
    # todo: expand to multi-QM cooling
    """

    def __init__(self, qubit: Qubit, readout: MeasurableResonator, beamsplitting: InputOutput):
        self.qubit = qubit
        self.readout = readout
        self.beamsplitter = beamsplitting

    def setup(self):
        pass

    def cool(self,
                state_quadrant: Literal[1,2,3,4] = 1,
                qubit_pulse_name: str = None,
                ro_pulse_name: str = None,
                ro_capture_mem: str = None,
                ro_cmacc_window: str = None,
                bs_pulse_name: str = None,
                qm_cooling_rounds=1,
                measurement_post_delay: float = 100e-9,
                state_register = None) -> None:
        """
        cools things
        """
        a = self.qubit._stimulus._acadia

        reg = a.sequencer().Register() if state_register is None else state_register

        ## Step 1: cool qubit
        self.qubit.prepare(state_quadrant, self.readout, qubit_pulse_name,
                           ro_pulse_name, ro_capture_mem, ro_cmacc_window, measurement_post_delay, reg)
        

        ## Step 2: swap Qm photon to qubit and cool qubit again
        for i in range(qm_cooling_rounds): # in case the first swap failed.
            with a.channel_synchronizer():
                self.beamsplitter.schedule_pulse(bs_pulse_name)
            self.qubit.prepare(state_quadrant, self.readout, qubit_pulse_name,
                               ro_pulse_name, ro_capture_mem, ro_cmacc_window, measurement_post_delay, reg)
            
        return reg


class TwoQubit:
    def __init__(self, qubit1: Qubit, qubit2: Qubit):
        """
        Container for running two-qubit measurement routines (e.g. joint
        tomography) on a pair of qubit objects that share the same Acadia backend.
        """
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self._acadia = self.qubit1._stimulus._acadia

    def _tomo_with_pulse(self, pulse1:str, pulse2:str, readout1_pulse_name:str, capture1_memory_name:str, capture1_window_name:str,
                        readout2_pulse_name:str, capture2_memory_name:str, capture2_window_name:str):

        """
        Schedule two-qubit tomography along a single "direction".

        This:
        - schedules ``pulse1`` on ``qubit1`` and ``pulse2`` on ``qubit2``
        - idles each readout pulse+capture for the qubit pulse duration of the corresponding qubit
        - plays the readout pulses and start capture

        """

        a = self._acadia
        with a.channel_synchronizer():
            pulse1_config = self.qubit1._stimulus.get_pulse_config(pulse1)
            pulse1_len = pulse1_config['flat'] + pulse1_config['ramp']
            self.qubit1._stimulus.schedule_pulse(pulse1)
            self.qubit1.readout_resonator._stimulus.dwell(pulse1_len)
            self.qubit1.readout_resonator._capture.dwell(pulse1_len)
            self.qubit1.readout_resonator.measure(readout1_pulse_name, capture1_memory_name, capture1_window_name)
            
            pulse2_config = self.qubit2._stimulus.get_pulse_config(pulse2)
            pulse2_len = pulse2_config['flat'] + pulse2_config['ramp']
            self.qubit2._stimulus.schedule_pulse(pulse2)
            self.qubit2.readout_resonator._stimulus.dwell(pulse2_len)
            self.qubit2.readout_resonator._capture.dwell(pulse2_len)
            self.qubit2.readout_resonator.measure(readout2_pulse_name, capture2_memory_name, capture2_window_name)

    def _make_tomo_pulses(self, qubit1_pi_pulse_name, qubit2_pi_pulse_name):
        """
        Precompute and cache the single-qubit tomography rotation pulses used for 
        measurement along different Pauli axes, based on the reference pi pulses.

        For unsymmetrized measurements, only the ``*p`` (positive) pulses are used.  
        For symmetrized measurements, each tomography direction is measured along both the 
        nominal axis and its anti-parallel counterpart (``*m``), with the latter's data 
        weighted by a negative sign to symmetrize measurement errors.

        The "Zp" tomography case effectively corresponds to an empty pulse.
        Here it is implemented as a 0-amplitude pulse of the same nominal length as the other
        tomography pulses for simplicity. In principle, this could be replaced
        with a dwell to save memory, but the current way just makes programming easier...

        """
        X1p_tomo_pulse = self.qubit1.make_rotation_pulse(-90, 90, qubit1_pi_pulse_name)
        X1m_tomo_pulse = self.qubit1.make_rotation_pulse(90, 90, qubit1_pi_pulse_name)
        Y1p_tomo_pulse = self.qubit1.make_rotation_pulse(90, 0, qubit1_pi_pulse_name)
        Y1m_tomo_pulse = self.qubit1.make_rotation_pulse(-90, 0, qubit1_pi_pulse_name)
        Z1p_tomo_pulse = self.qubit1.make_rotation_pulse(0, 0, qubit1_pi_pulse_name)
        Z1m_tomo_pulse = self.qubit1.make_rotation_pulse(180, 0, qubit1_pi_pulse_name)
        
        X2p_tomo_pulse = self.qubit2.make_rotation_pulse(-90, 90, qubit2_pi_pulse_name)
        X2m_tomo_pulse = self.qubit2.make_rotation_pulse(90, 90, qubit2_pi_pulse_name)
        Y2p_tomo_pulse = self.qubit2.make_rotation_pulse(90, 0, qubit2_pi_pulse_name)
        Y2m_tomo_pulse = self.qubit2.make_rotation_pulse(-90, 0, qubit2_pi_pulse_name)
        Z2p_tomo_pulse = self.qubit2.make_rotation_pulse(0, 0, qubit2_pi_pulse_name)
        Z2m_tomo_pulse = self.qubit2.make_rotation_pulse(180, 0, qubit2_pi_pulse_name)
        
        tomo_pulse_dict = { "X1p": X1p_tomo_pulse, "X1m": X1m_tomo_pulse, 
                            "Y1p": Y1p_tomo_pulse, "Y1m": Y1m_tomo_pulse,
                            "Z1p": Z1p_tomo_pulse, "Z1m": Z1m_tomo_pulse,
                            "X2p": X2p_tomo_pulse, "X2m": X2m_tomo_pulse, 
                            "Y2p": Y2p_tomo_pulse, "Y2m": Y2m_tomo_pulse,
                            "Z2p": Z2p_tomo_pulse, "Z2m": Z2m_tomo_pulse}
        self.tomo_pulse_dict = tomo_pulse_dict
        return tomo_pulse_dict


    def full_2q_tomo(self, tomo_cache, core:callable, qubit1_pi_pulse_name:str, qubit2_pi_pulse_name:str,
                    readout1_pulse_name:str, capture1_memory_name:str, capture1_window_name:str,
                    readout2_pulse_name:str, capture2_memory_name:str, capture2_window_name:str,
                    tomo_reg=None, symmetrize=True):
        """
        Build a full 2-qubit tomography program on the sequencer.

        High-level idea:
        - We create / reuse tomography rotation pulses for each qubit.
        - For each tomography "direction" (XX, XY, ..., ZZ, plus sign flips if
          ``symmetrize=True``), we:
          1. run the user's `core(a)` sequence (to prepare the 2q state),
          2. apply the appropriate tomography rotations on each qubit,
          3. measure both qubits.

        The directions are indexed and wrapped in
        ``with a.sequencer().test(tomo_reg == direction_idx):``
        so the hardware can branch per-iteration.

        ``symmetrize=True`` means we also flip +/- signs on each axis so you can
        do simple sign-averaging of readout bias. That turns 9 Pauli axes
        into 36 runs (each axis with four sign combos ``pp, pm, mp, mm``).
        

        :param tomo_cache:
            Pre-built sequencer program/data to load into ``tomo_reg`` before branching.
            ``tomo_cache[0]`` is loaded into the register at the start. User need to define
            and sweep over the cache value to perform the tomography in `main()`

        :param core:
            Callable of the form ``core(acadia)``. This is the state-prep body you
            want to tomography after. It will be called once per tomography
            direction inside the branching block.
        
            
        :param tomo_reg:
            Optional sequencer register. If ``None``, a fresh register is created.
        :param bool symmetrize:
            If ``True``, include +/- axis flips for both qubits, giving 36 total
            branches instead of 9. If ``False``, only the normal +axes are used.
        """
        
        a = self._acadia
        tomo_pulse_dict = self._make_tomo_pulses(qubit1_pi_pulse_name, qubit2_pi_pulse_name)

        if tomo_reg is None:
                tomo_reg = a.sequencer().Register()

        tomo_reg.load(tomo_cache[0])
        
        num_runs = 36 if symmetrize else 9

        for direction_idx in range(num_runs):
            if symmetrize:
                pauli_str = _idx_to_pauli_str(direction_idx//4, 2)
                direction_sign1 = ['p', 'm'][(direction_idx % 4) // 2]
                direction_sign2 = ['p', 'm'][(direction_idx % 4) % 2]
            else:
                pauli_str = _idx_to_pauli_str(direction_idx, 2)
                direction_sign1 = direction_sign2 = "p"

            with a.sequencer().test(tomo_reg == direction_idx):
                core(a) # run the core part, then do tomo among one direction.
                self._tomo_with_pulse(tomo_pulse_dict[pauli_str[0] + f"1{direction_sign1}"], 
                                    tomo_pulse_dict[pauli_str[1] + f"2{direction_sign2}"],
                                    readout1_pulse_name, capture1_memory_name, capture1_window_name,
                                    readout2_pulse_name, capture2_memory_name, capture2_window_name)


    def load_tomo_pulses(self, symmetrize=True):
        """
        Load the tomography rotation pulses that were generated by
        :meth:`_make_tomo_pulses` into each qubit's stimulus memory.

        ** Remember to call this before calling `acadia.run()` **

        """
        for k, v in self.tomo_pulse_dict.items():
            if k[1] == '1' and (symmetrize or k.endswith("p")):
                self.qubit1._stimulus.load_pulse(v)
            elif k[1] == '2' and (symmetrize or k.endswith("p")):
                self.qubit2._stimulus.load_pulse(v)




# ----------- generic helper functions -----------------------------
@functools.cache
def _idx_to_pauli_str(index:int, n_qubits:int) -> str:
    """
    Convert an integer index to an N-qubit Pauli string, using base-3 encoding.

    Examples
    --------
    >>> idx_to_pauli_str(0, 3)
    'XXX'
    >>> idx_to_pauli_str(5, 3)  # 5 in base 3 → '012'
    'XYZ'
    """
    base3 = np.base_repr(index, base=3).zfill(n_qubits)
    return ''.join('XYZ'[int(d)] for d in base3)