import sys
import shutil
import os
from typing import Callable, Literal, Dict, List, Any, Union, Literal, Tuple
from pathlib import Path
import json
import logging

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import hann as scipy_hann

from acadia import Acadia, Channel, Runtime, WaveformMemory, WaveformMemory, Operation
from acadia.compiler import ManagedResource, Symbol

##############################################################
# Todo: IN ACADIA_QMSMT
# Bug: stretch_length does not seem to work if your waveform memory is less than 3 cycles, ie less than 15 ns
# Bug: if ramp = 0 and flat is an odd multiple of clock_sample_time, the first sample is zero
# Todo: Implement a gaussian pulse
# Todo: Change Qubit, MeasurableResonator, and QMsmtRuntime to use InputOutput
# Todo: Should we allow get_waveform_memory to take a dictionary? 
# think about creation of a long saturation pulse/temporary pulse which we don't put in the YAML file)

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
KWARG_EXCLUDE_LIST = ["data", "scale", "detune", "ramp", "flat", "use_stretch", "memory_length", "name"]

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
    def scale_detune_pulse(pulse: NDArray, scale: complex = None,
                    detune: float = None, sample_frequency: float = 8e8) -> NDArray:
        ''' Scale and detune a pulse'''
        if scale is not None:
            pulse = np.multiply(pulse, scale)
        if detune is not None:
            t = np.arange(pulse.size, dtype = np.float64) / sample_frequency
            pulse = np.multiply(pulse, np.exp(2 * np.pi * 1j* detune * t))
        return pulse
            
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
            if not k in current_level:
                raise KeyError(f"Missing key '{k}' in config dict under "
                               f"'{'.'.join([self._name] + list(cfg_path[:i]))}'")
            
            current_level = current_level[k]
        
        return current_level

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
        samples_per_cycle = self._acadia._firmware["rfdc"]["dac"]["channel_interface_width"][self._channel.num()] // 32
        stretch_length = None # Initialize stretch length to None
        
        if use_stretch: # This part computes 'f' appropriately if use_stretch is True
            if f is None or f == 0:
                logger.warning("Warning: flat is None, setting it to minimum value of %.3g seconds.", clock_sample_time)
                f = clock_sample_time
            # Find the closest multiple of clock_sample_time to f
            stretch_factor = np.ceil(f / clock_sample_time)
            # Remember the stretch length
            stretch_length = (stretch_factor -  1) * clock_sample_time
            # Passing 0 to the stretch_length variable causes errors
            stretch_length = stretch_length if stretch_length > 0 else None
            if (f/clock_sample_time) % 1 > 1e-15:
                logger.warning("Warning: flat is not a multiple of clock_sample_time"
                                "Because use_stretch = True, the actual flat played will be %.3g seconds.", clock_sample_time * stretch_factor)
            f = clock_sample_time # The flat part is always set to clock_sample_time when using stretch

        # If the pulse data is a string, we compute the ramp and flat fractions and         
        if isinstance(pulse_config["data"], str): 
            r = pulse_config["ramp"]
            memory_length = pulse_config.get("memory_length", None)
            ml = memory_length # Create new variable to be modified later
            
            if use_stretch:
                if r is None or r == 0:
                    logger.warning("Warning: ramp is None, and use_stretch = True."
                                   "Your pulse will be zero padded with 5 ns on either side")
                    r = 1e-12
                # Find the padding necessary to ensure that there is a flat cycle in the middle of the pulse
                tau_pad = np.ceil(r / (2 * clock_sample_time)) * 2 * clock_sample_time - r
                # Adjust memory length
                ml = tau_pad + r + f
                if memory_length is not None:
                    logger.warning("Warning: memory_length can't be set when using stretch, the value used is %.3g.", ml)
            else:
                if memory_length is None:
                    # Find the closest multiple of clock_sample_time to r + f
                    ml = np.ceil((r + f) / clock_sample_time) * clock_sample_time
                elif (memory_length - r - f) < -1e-12:
                    ml = memory_length
                    logger.warning("Warning: memory_length %s is less than ramp + flat %.3g. Truncating the pulse.", memory_length, r + f)
                elif (memory_length - r - f) > 1e-12:
                    ml = memory_length
                    logger.warning("Warning: memory_length %s is greater than ramp + flat %.3g. Zero padding the pulse with %.3g seconds.",
                                   memory_length, r + f, ml - r - f)

            # Calculate the ramp and flat lengths as fractions of the memory length
            ramp_frac = r / ml
            flat_frac = f / ml

        else:
            ml = len(pulse_config["data"]) * clock_sample_time/samples_per_cycle
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
        
        pulse_config = self.get_pulse_config(pulse_config)
        pulse_name = pulse_config["name"]
        pulse_hash = make_hash(pulse_config)
        # Don't populate cache if pulse_name is None
        if pulse_name is None:
            return
        # If pulse doesn't exist, create a new entry in the cache
        if pulse_name not in self._pulse_cache:
            self._pulse_cache[pulse_name] = {'waveforms': {}, 'memory': None}
        # If pulse_hash already exists, return
        if self._pulse_cache[pulse_name]['waveforms'].get(pulse_hash) is not None:
            return
        else: # If pulse_hash doesn't exist, compute the pulse parameters
            waveform_container = {}
            ramp_frac, flat_frac, memory_length, stretch_length = self._compute_ramp_flat_memlen_stretchlen(pulse_config)
            waveform_container["ramp_frac"] = ramp_frac
            waveform_container["flat_frac"] = flat_frac
            waveform_container["memory_length"] = memory_length
            waveform_container["stretch_length"] = stretch_length
            self._pulse_cache[pulse_name]['waveforms'][pulse_hash] = waveform_container

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
                
                self.prepare_pulse_params(waveform_memory)
                wfm_name = waveform_memory["name"] if isinstance(waveform_memory, dict) else waveform_memory
                if self._pulse_cache[wfm_name].get("memory") is None:
                    pulse_config = self.get_pulse_config(waveform_memory)
                    pulse_hash = make_hash(pulse_config)
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
            raise TypeError("`load_waveform` requires either a WaveformMemory"
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
            pulse = self.get_pulse_config(pulse)
        else:
            if isinstance(pulse, (float, complex)):
                # If it is a pulse, convert it into an array
                pulse = np.ones(memory.size, dtype=np.complex128) * pulse
            pulse = {"data": pulse}
            # The pulse name is either None, or the name of the memory
            pulse["name"] = memory_name
            # Need to remember length of memory if pulse is a float or complex
        pulse.update(kwargs)
        samples = self.compute_pulse(pulse)
        
        wfm.load(samples)
        
    def compute_pulse(self, 
                        pulse: Union[str,dict,None] = None, 
                        **kwargs) -> NDArray:
        """
        This method computes a pulse waveform based on the configuration. This method 
        also caches the computed pulse in the `_pulse_cache` dictionary, so that 
        it can be reused later without recomputing it.
        :param pulse: The name of the pulse in the configuration, or a dictionary
            containing the configuration of the pulse. If None, the first pulse in
            the configuration is used.
        :param kwargs: Additional keyword arguments to be passed to the pulse
            generation function. These can include parameters such as `scale`,
            `detune`, `ramp`, `flat`, `use_stretch`, and `memory_length`. The values
            provided here will override the values in the pulse configuration.
        :return: A numpy array containing the computed pulse waveform.
        """
        
        # Create useful variables
        samples_per_cycle = self._acadia._firmware["rfdc"]["dac"]["channel_interface_width"][self._channel.num()] // 32
            
        # Get the pulse configuration and its hash
        pulse_config = self.get_pulse_config(pulse)
        pulse_config.update(kwargs)
        pulse_hash = make_hash(pulse_config)
        scale = pulse_config.get("scale", None)
        detune = pulse_config.get("detune", None)
        use_stretch = pulse_config.get("use_stretch", False)
        if use_stretch and detune is not None:
            raise ValueError("Detune and use_stretch cannot be used together. "
                            "Please use either one or the other.")
        
        if pulse_config["name"] is not None:
            # Look in the cache for the pulse
            self.prepare_pulse_params(pulse_config)
            waveform_container = self._pulse_cache[pulse_config["name"]]['waveforms'][pulse_hash]
            samples = waveform_container.get("samples")
                
            if samples is None:
                # This means the pulse hasn't already been computed and cached
                if scale is not None or detune is not None:
                    # compute waveform_0 (one with no scale or detuning)
                    # so that pulse computation is not repeated
                    # when changing just scale and detuning
                    waveform_0_config = pulse_config.copy()
                    waveform_0_config["scale"] = None
                    waveform_0_config["detune"] = None
                    samples = self.compute_pulse(waveform_0_config)
            
                elif isinstance(pulse_config["data"], str):
                    # If the pulse is a string, we need to compute it
                    ramp_frac = waveform_container["ramp_frac"]
                    flat_frac = waveform_container["flat_frac"]
                    memory_length = waveform_container["memory_length"]
                    num_samples = self._acadia.seconds_to_cycles(memory_length) * samples_per_cycle
                    samples = np.zeros(num_samples, dtype = np.complex128)
                    func_kwargs = {k: v for k, v in pulse_config.items() if k not in KWARG_EXCLUDE_LIST}
                    InputOutputWaveforms.flattop_generator(samples, getattr(InputOutputWaveforms, pulse_config["data"]), ramp_frac, flat_frac, **func_kwargs)
                else: # we just use the data directly
                    samples = pulse_config["data"]
                
                samples = InputOutputWaveforms.scale_detune_pulse(samples, scale, detune, self._channel.interface_sample_frequency)
                self._pulse_cache[pulse_config["name"]]['waveforms'][pulse_hash]["samples"] = samples
        else:
            # No caching pulses without names
            samples = pulse_config["data"]
            samples = InputOutputWaveforms.scale_detune_pulse(samples, scale, detune, self._channel.interface_sample_frequency)
        
        return samples
    
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
      
        self._acadia.schedule_waveform(wfm, stretch_length=stretch_length)

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
                    from acadia_qmsmt.helpers.yaml_editor import load_yaml
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

    def update_io_yaml_field(self, ioconfig_name: str, config_field: str, value: Any, verbose=False):
        """
        Update the value of a field in the yaml config file for the specified IOConfig.
        """
        from acadia_qmsmt.helpers.yaml_editor import update_yaml
        yaml_path = self._ios[ioconfig_name]._config["__yaml_path__"]
        
        ioconfig = getattr(self, ioconfig_name)
        if type(ioconfig) == str:
            yaml_key = ioconfig
        elif type(ioconfig) == dict:
            yaml_key = ioconfig["__yaml_key__"]
        new_cfg = update_yaml(yaml_path, {f"{yaml_key}.{config_field}": value}, verbose=verbose)
        logger.info(f"!! updated yaml file `{yaml_path}`: {yaml_key}.{config_field}: {value}")
        return new_cfg

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
 
    def measure(self, 
                stimulus_waveform_memory: Union[str, WaveformMemory] = None, 
                capture_waveform_memory: Union[str, WaveformMemory] = None,
                window_name: str = None,
                write_mode: Literal["upper", "lower"] = "upper",
                reset_fifo: bool = True
                ):
        """
        Schedules the measurement of accumulated IQ points. 
        This function should be called inside of a channel synchronizer.

        :param write_mode: The output port of the CMACC is driven by a multiplexer,
            which allows the user to decide which data is written into memory. 
            The output port has 32 bits per quadrature, but the internal accumulator
            has 48 bits per quadrature, and therefore it is left to the user to
            decide whether the upper or lower 32 bits are presented to the output.
            Using the upper 32 bits reduces the precision of the output but reduces
            the probability of overflow.
        :type write_mode: str, one of "upper", "lower".

        """
        capture_waveform = self._capture.get_waveform_memory(capture_waveform_memory)

        # ------------ schedule drive pulse ------------------------------
        self._stimulus.schedule_pulse(stimulus_waveform_memory)


        # ------------ configure and perform capture ---------------------
        # First, we'll determine what window we need
        # Here we'll cache allocated windows. If we previously used a window, 
        # we'll retrieve its memory, otherwise we'll allocate it fresh
        if window_name is None:
            window_name = list(self._capture.get_config("windows").keys())[0]
            
        if window_name in self._windows:
            # Regardless of what the window is, if we have it already,
            # we'll pass in its WaveformMemory so that we don't re-allocate it
            kernel_arg = self._windows[window_name]

        else:
            # Allocate the memory anew 
            # We'll interpret the type of the argument slightly differently 
            # than configure_cmacc expects; if the window argument is a float,
            # we'll interpret this as the amplitude of a boxcar window, in contrast
            # to configure_cmacc interpreting a float argument as the length in seconds
            # that the window should be. The behavior here will be different as this is a
            # much more likely use case and will be much more intuitive in the config file
            window_config = self._capture.get_config("windows", window_name, "data")
            kernel_arg = None if isinstance(window_config, float) else window_config

        # Because the offset is just a number (and not a thing that needs
        # to be allocated), we can just get it from the config every time
        cmacc_offset = self._capture._config["windows"][window_name].get("offset", (0,0))
        offset_converted = [int(np.int32(q).astype(np.uint32)) for q in cmacc_offset]


        # if kernel is None or a boxcar, the accumulation length will be set to capture length
        if kernel_arg is None or (hasattr(kernel_arg, "size") and kernel_arg.size == 1):
            length = self._capture.get_config("memories", capture_waveform_memory, "length")
        else: # otherwise, stream_cmacc will use kernel length as the accumulation length
            length = None

        self._stream, window_mem = self._capture._acadia.stream_cmacc(self._capture._channel, capture_waveform,
                                                                      kernel=kernel_arg, length=length,
                                                                      preload=offset_converted,
                                                                      write_mode=write_mode, reset_fifo=reset_fifo,
                                                                      last_only=True, accumulator_done=False)    
        
        # Cache the relevant stream and window
        # If we had already allocated and cached these, this will just store 
        # them back with the same values
        self._windows[window_name] = window_mem


    def measure_trace(self, 
                stimulus_waveform_memory: Union[str, WaveformMemory] = None, 
                capture_waveform_memory: Union[str, WaveformMemory] = None,
                reset_fifo: bool = True):
        """
        Schedules the measurement of raw IQ traces. 
        This function should be called inside of a channel synchronizer.

        """
        # todo: allow configurable decimation. Currently this must be 4

        capture_waveform = self._capture.get_waveform_memory(capture_waveform_memory)

        # ------------ schedule drive pulse ------------------------------
        self._stimulus.schedule_pulse(stimulus_waveform_memory)


        # ------------ configure and perform capture ---------------------
        length = self._capture.get_config("memories", capture_waveform_memory, "length")
        self._capture._acadia.stream_cmacc(self._capture._channel, capture_waveform,
                                           length=length, kernel=None, preload=(0, 0),
                                            write_mode="input", reset_fifo=reset_fifo,
                                            last_only=False, accumulator_done=False)    


    def get_measurement(self) -> Operation:
        """
        Blocks until the measurement is complete, as indicated by the CMACC
        completion status bit.
        """

        # `Acadia.cmacc_get_quadrant` will wait until cmacc is done
        return self._capture._acadia.cmacc_get_quadrant(self._stream)


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

    def wait_until_measurement_done(self):
        """
        block the sequencer until the measurement is complete.
        :return:
        """
        a = self._capture._acadia
        with a.sequencer().repeat_until(a.cmacc_done(self._stream)):
            pass

class Qubit:
    """
    A collection of functions that are useful for manipulating and measuring a qubit.
    """

    def __init__(self, stimulus: InputOutput):
        """
        Runtime class that contains experiment specific functions, including sub_sequence functions
        """
        self._stimulus = stimulus

    def set_frequency(self, frequency: float, sync: bool = True):
        """
        The frequencies of both the stimulus and the capture are updated and 
        optionally synchronized.
        """

        self._stimulus.set_nco_frequency(frequency)
        self._stimulus.reset_nco_phase()
        if sync:
            self._stimulus._acadia.update_ncos_synchronized()

    def prepare(self, 
                state_quadrant: Literal[1,2,3,4],
                measurement_resonator: MeasurableResonator,
                pulse_waveform_memory: str = None,
                measurement_stimulus_waveform_memory: str = None,
                measurement_capture_waveform_memory: str = None,
                measurement_cmacc_window: str = None,
                measurement_post_delay: float = 2e-6) -> None:
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

        a = self._stimulus._acadia
        quadrant_reg_value = getattr(a, f"CMACC_QUADRANT_{state_quadrant}")
        reg = a.sequencer().Register()


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
                a.barrier()
                measurement_resonator.measure(measurement_stimulus_waveform_memory,
                                              measurement_capture_waveform_memory,
                                              measurement_cmacc_window)
            with a.channel_synchronizer():
                self._stimulus.dwell(measurement_post_delay)

            reg.load(measurement_resonator.get_measurement())

class DriveChannel:
    """
    A collection of functions that are useful for generic drive pulse
    """

    def __init__(self, stimulus: InputOutput):
        """
        Runtime class that contains experiment specific functions, including sub_sequence functions
        """
        self._stimulus = stimulus

    def set_frequency(self, frequency: float, sync: bool = True):
        """
        The frequencies of both the stimulus and the capture are updated and 
        optionally synchronized.
        """

        self._stimulus.set_nco_frequency(frequency)
        self._stimulus.reset_nco_phase()
        if sync:
            self._stimulus._acadia.update_ncos_synchronized()

    def pulse(self, waveform_memory: Union[str, WaveformMemory] = None, 
                stretch_length:Union[float, ManagedResource] = None) -> None:
        """
        Apply a beamsplitting pulse. The waveform memory parameter is passed directly to 
        :meth:`InputOutput.get_waveform_memory`; see documentation therein for 
        argument behavior.
        """

        self._stimulus.schedule_pulse(waveform_memory, stretch_length)

class QubitQmCooler:
    """
    A collection of functions that are useful for cooling a qubit and QM(s)
    # todo: expand to multi-QM cooling
    """

    def __init__(self, qubit: Qubit, readout: MeasurableResonator, beamsplitting: DriveChannel):
        self.qubit = qubit
        self.readout = readout
        self.beamsplitter = beamsplitting

    def setup(self):
        pass

    def cool(self,
                state_quadrant: Literal[1,2,3,4] = 1,
                qubit_pulse_mem: str = None,
                ro_stimulus_mem: str = None,
                ro_capture_mem: str = None,
                ro_cmacc_window: str = None,
                bs_pulse_mem: str = None,
                qm_cooling_rounds=1) -> None:
        """

        :param state_quadrant:
        :param qubit_pulse_mem:
        :param ro_stimulus_mem:
        :param ro_capture_mem:
        :param ro_cmacc_window:
        :param bs_pulse_mem:
        :param qm_cooling_rounds:
        :return:
        """
        a = self.qubit._stimulus._acadia

        ## Step 1: cool qubit
        self.qubit.prepare(state_quadrant, self.readout, qubit_pulse_mem,
                           ro_stimulus_mem, ro_capture_mem, ro_cmacc_window)

        ## Step 2: swap Qm photon to qubit and cool qubit again
        for i in range(qm_cooling_rounds): # in case the first swap failed.
            with a.channel_synchronizer():
                self.beamsplitter.pulse(bs_pulse_mem)
            self.qubit.prepare(state_quadrant, self.readout, qubit_pulse_mem,
                               ro_stimulus_mem, ro_capture_mem, ro_cmacc_window)
