import sys
import shutil
import os
from typing import Callable, Literal, Dict, List, Any, Union, Literal
from pathlib import Path
import json

import numpy as np
from numpy.typing import NDArray
from scipy.signal.windows import hann as scipy_hann

from acadia import Acadia, Channel, Runtime, ChannelWaveformMemory, WaveformMemory, Operation

__all__ = ["InputOutput", "InputOutputWaveforms", "QMsmtRuntime", "MeasurableResonator", "MeasurableQubit"]

class InputOutputWaveforms:
    """
    A class containing methods for all of the waveform shapes that can be 
    referenced by string in a configuration file.
    """

    @staticmethod
    def hann(output: NDArray) -> None:
        """
        Calculate a Hann (or raised-cosine) function. The entire output is filled.
        """
        phase_per_sample = 2 * np.pi / output.size
        output[:] = 0.5 * (1 - np.cos(phase_per_sample * np.arange(output.size)))

    @staticmethod
    def hann_precise(output: NDArray, length: float = 1, offset: float = 0) -> None:
        """
        Calculate a Hann (or raised-cosine) function. The length of the pulse 
        is continuously variable, as is the starting time.

        :param length: The length of the pulse, expressed as a fraction of the 
            total size of the output
        :type length: float
        :param offset: The starting time of the pulse, expressed as a fraction 
            of the total size of the output.
        :type offset: float
        """

        sample_times = np.arange(output.size) / output.size
        output[:] = 0.5 - 0.5 * np.cos(2 * np.pi * (sample_times - offset) / length)
        output[:] *= np.logical_and(sample_times >= offset, sample_times < offset+length)
        

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
        self._allocated_memories: Dict[str,ChannelWaveformMemory] = {}

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

    def get_waveform_memory(self, waveform_memory: Union[str, WaveformMemory] = None) -> ChannelWaveformMemory:
        """
        A shortcut function for allocating waveform memory for a specified 
        channel and waveform using pre-defined configurations.

        Serves as a shortcut for creating waveform memory objects based on the 
        parameters defined in ``config["memories"][waveform_memory]``.

        If no name is provided, the first memory in the list of available
        memories is used.

        :param waveform_memory: Name of the waveform memory defined under the 
            "memories" sub-dictionary for the given channel
        :return: Allocated WaveformMemory object
        """

        # Just return the input if we were directly given a memory
        # This is useful behavior for writing objects later that, when calling this
        # function to retrieve a waveform, can be directly passed either a name or
        # an actual memory
        if isinstance(waveform_memory, WaveformMemory):
            return waveform_memory

        if waveform_memory is None:
            waveform_memory = list(self.get_config("memories").keys())[0]

        if not isinstance(waveform_memory, str):
            raise TypeError(f"Waveform memories must be specified by string"
                            f" when not dirfectly provided or inferred;"
                            f" received memory specified of type"
                            f" {type(waveform_memory)}")

        # If we haven't yet allocated memory for the waveform, do so
        if waveform_memory not in self._allocated_memories:
            wf_cgf = self.get_config("memories", waveform_memory)
            self._allocated_memories[waveform_memory] = self._acadia.create_waveform_memory(self._channel, **wf_cgf)

        return self._allocated_memories[waveform_memory]

    def blank_waveform_generator(self, reference_waveform: str = None) -> Callable:
        """
        Factory function for creating blank waveform functions for DAC or ADC channels.

        This is written as a factory function so that the user can easily call the returned function as needed to
        make blank waveforms of desired lengths for any channel.

        :param acadia: Acadia instance for creating waveforms.
        :param channel_name: name of the DAC or ADC channel
        :param reference_waveform: for ADC channel, the name of a reference waveform needs to provided to decide the
            `decimation` and `region`
        :returns: A function that creates a blank waveform when called with a length.
        """
        waveform_args = {"blank": True}

        # get decimation and wf region for blank wf in adc channels
        if not self._channel.is_dac:
            if reference_waveform is None:
                # check if all the waveform mem configuration in the channel has the same settings
                for i, wf_cfg in enumerate(self.get_config("memories").values()):
                    d, r = wf_cfg.get("decimation", None), wf_cfg.get("region", None)
                    if i == 0:
                        decimation, region = d, r
                    if decimation != d or region != r:
                        raise ValueError(
                            f"for blank waveform in an ADC channel, a reference waveform name must be provided")
            else:
                ref_wf_cfg = self.get_config("memories", reference_waveform)
                decimation = ref_wf_cfg.get("decimation", None)
                region = ref_wf_cfg.get("region", None)

            if decimation == 0:
                decimation = 4  # for cmacc 
            waveform_args.update({"decimation": decimation, "region": region})

        def blank_wf_gen(length: float) -> ChannelWaveformMemory:
            blank_wf = self._acadia.create_waveform_memory(self._channel, length=length, **waveform_args)
            return blank_wf

        return blank_wf_gen

    def load_waveforms(self, **kwargs):
        """
        Populate the memory of previously allocated waveforms. The names of 
        the keyword arguments should correspond to waveform memory names in the 
        ``"memories"`` section of the configuration, and the values should be 
        either strings that correspond to waveform names in the ``"waveforms"`` 
        section of the configuration or a value to be directly passed to 
        :meth:`WaveformMemory.load`. 
        
        This is a shortcut for calling :meth:`load_waveform` for all allocated
        memories and loading them with their specified configurations. To 
        temporarily override values from the configuration, use 
        :meth:`load_waveform`.
        """

        for memory_name, waveform in kwargs.items():
            self.load_waveform(memory_name, waveform)

    def load_waveform(self,
                        memory: Union[str,WaveformMemory] = None,
                        waveform: Union[str,NDArray,float,complex,None] = None,
                        scale: complex = None,
                        **kwargs) -> None:
        """
        Load a wavefrom either from a configuration section or from sample data.

        The waveform memory to populate can be specified either by a string, 
        in which case it refers to a memory in this configuration's "memories" 
        section, or the :class:`WaveformMemory` object to load can be provided 
        directly. If ``None``, the first entry in the configuration's "memories"
        section is inferred.

        The data loaded into the memory is determined by the type of the 
        ``waveform`` parameter:

            - If ``waveform`` is a string, it must correspond to a section
            in the "waveforms" configuration; this function is then a shorthand for
            calling :meth:`compute_waveform` and passing the floating-point output 
            back into this function. If ``scale`` is ``None``, it will attempt to be
            retrieved from the configuration.

            - If ``waveform`` is a dict, it will be passed directly into 
            :meth:`compute_waveform` as the configuration.

            - If ``None``, the first entry in the configuration's "waveforms" 
            section is inferred.

            - For any other type, it will be directly passed to 
            :meth:`WaveformMemory.load`. Any additional keyword arguments will 
            cause an exception to be raised.

        The scale of the waveform being loaded depends on the type of the ``scale`` parameter:

            - If a complex or a float, that value is used.

            - If a string, the corresponding section of the configuration is used.

            - If `None`, the scale is retrieved from the configuration (whether that 
                is a section in the configuration file or a directly-provided dict).
                If it cannot be found, an exception is raised.

            
        """
        # Determine the memory to be loaded
        if memory is None or isinstance(memory, str):
            memory = self.get_waveform_memory(memory)
        elif not isinstance(memory, WaveformMemory):
            raise TypeError(f"`load_waveform` requires either a WaveformMemory"
                            " object that can be loaded, or a string that"
                            " specifies one from the configuration.")

        # Determine the samples to load into the waveform
        if waveform is None or isinstance(waveform, (str, dict)):
            samples = self.compute_waveform(memory.size, waveform, **kwargs)
        else:
            if len(kwargs) != 0:
                raise ValueError(f"Keyword arguments may not be provided when"
                                 f" directly providing sample data.")
            samples = waveform

        # Determine the scale
        # If we've provided a scale, that should override anything else; 
        # otherwise, if we specified the waveform by name, check to see if its
        # configuration provides a scale
        if scale is None:
            if waveform is None:
                scale = list(self._config["waveforms"].values())[0]["scale"]
            elif isinstance(waveform, str):
                scale = self.get_config("waveforms", waveform)["scale"]
            elif isinstance(waveform, dict):
                scale = waveform["scale"]
            else:
                raise KeyError(f"Unable to infer scale for waveform of type {type(waveform)}")
        elif isinstance(scale, str):
            scale = self.get_config("waveforms", scale)["scale"]
        elif not isinstance(scale, (int, float, complex)):
            raise TypeError(f"Unable to derive scale from argument of type {scale}")

        memory.load(samples, scale)
            
    def compute_waveform(self, 
                        size: Union[str,WaveformMemory,int] = None, 
                        waveform: Union[str,dict,None] = None, 
                        **kwargs) -> np.ndarray:
        """
        Compute the samples of a waveform from a configuration section.

        The size of the waveform can be specified in a handful of ways, 
        depending on the type of the ``size`` parameter:
        
            - If an int, this is the number of samples to compute.

            - If a string, this refers to a memory in this configuration's 
            "memories" section. 
            
            - If a :class:`WaveformMemory` object, its ``size`` property is used.
            
            - If ``None``, the first entry in the configuration's "memories" 
            section is inferred.

        The waveform is specified by a string that refers to a section in the 
        configuration's "waveforms" section. If ``None``, the first entry in 
        the "waveforms" section is used. The configuration may also be directly 
        provided as a dict.

        Within the specified waveform configuration, there must be a key "data", 
        which specifies how the waveform is to be computed depending on the type:
        
            - If the value of the "data" key is a string, it must refer to a member 
            function of :class:`InputOutputWaveforms` which will then be used to 
            compute the samples. In this case, all other key-value pairs in the 
            configuration section will be provided as keyword arguments to the 
            function, which may be overridden by providing them as keyword arguments
            to this function. 

            - If the value of the "data" key is a scalar or numpy array, it will be
            directly returned. If there are any other keyword arguments in the 
            configuration or passed to this function (except for "scale"), then
            an exception is raised.
        """
        # Determine the number of samples needed
        if size is None:
            size = self.get_waveform_memory(list(self._config["memories"].keys())[0]).size
        elif isinstance(size, WaveformMemory):
            size = size.size
        elif isinstance(size, str):
            if size not in self._allocated_memories:
                raise ValueError(f"WaveformMemory {size} requested but was never allocated.")
            size = self._allocated_memories[size].size
        elif not isinstance(size, int):
            raise TypeError(f"Unable to specify waveform size with object of "
                            f"type {type(size)}")

        # Determine the samples that will be loaded in
        if waveform is None:
            waveform_config = list(self._config["waveforms"].values())[0]
        elif isinstance(waveform, dict):
            waveform_config = waveform
        elif isinstance(waveform, str):
            waveform_config = self.get_config("waveforms", waveform)
        else:
            raise TypeError(f"Unable to specify waveform with object of type {type(waveform)}")
       
        
        if "data" not in waveform_config:
            raise KeyError(f"Waveform configuration missing required \"data\" key.")

        if isinstance(waveform_config["data"], str):  
            samples = np.empty(size, dtype=np.complex128)
            func_kwargs = {k:v for k,v in waveform_config.items() if k != "data" and k != "scale"}
            func_kwargs.update(kwargs)
            getattr(InputOutputWaveforms, waveform_config["data"])(samples, **func_kwargs)
            return samples
        else:
            if len(kwargs) != 0:
                raise ValueError("Keyword arguments cannot be provided to "
                                "`compute_waveform` when the configuration "
                                "specifies sample data.")
            return waveform_config["data"]


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
        self._yaml_paths = {}
        for name,type_hint in self._get_fields().items():
            if type_hint is IOConfig:
                value = getattr(self, name)
                if isinstance(value, tuple):
                    from acadia_qmsmt.helpers.yaml_editor import load_yaml
                    config_dict = load_yaml(value[1])[value[0]]
                    config_dict["yaml_path"] = value[1]
                    self._yaml_paths[name] = (config_dict["yaml_path"], value[0])
                elif isinstance(value, str):
                    from acadia_qmsmt.helpers.yaml_editor import load_yaml
                    yaml_path = kwargs.get("yaml_path") 
                    # Allow child classes to pass `yaml_path=None` and still get the default "config.yaml"
                    yaml_path = "config.yaml" if yaml_path is None else yaml_path 
                    config_dict = load_yaml(yaml_path)[value]
                    config_dict["yaml_path"] = yaml_path
                    self._yaml_paths[name] = (config_dict["yaml_path"], value)
                elif isinstance(value, dict):
                    config_dict = value
                    config_dict["yaml_path"] = None
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
            from acadia_qmsmt.helpers.yaml_editor import update_yaml
            yaml_path = self._ios[ioconfig_name]._config["yaml_path"]
            update_yaml(yaml_path, {f"{ioconfig}.{config_field}": value})
        else:
            raise TypeError(f"Unable to update IO config when provided as type {type(ioconfig)}")


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

    def prepare_cmacc(self, 
                window_name: str = None,
                output_type: Literal["upper", "lower", "input", None] = "upper",
                output_last_only: bool = True,
                reset_fifo: bool = True):
        """
        Perform a measurement of the resonator. 

        The length and decimation of the capture are retrieved from the provided 
        capture configuration. Because a CMACC is always used for the measurement, 
        the decimation of the specified capture configuration must be a multiple of 4. 
        A value of 0 may also be provided, in which case
        the decimation factor will automatically be chosen so that the 
        entire captured trace is decimated into a single point.

        
        :param window_name: The name of the window to be used (and newly 
            allocated, if necessary). The name should be a key in the 
            "windows" section of the configuration for the provided capture.
            The value of the window (as provided in the configuration's ``data`` 
            key may be a float, in which case a boxcar window
            is allocated. Note that this does not actually populate the window
            memory with that float, but simply specifies that a boxcar window
            will be used so that only a single point in window memory is allocated.
            The value of the window may also be an array, in which case a window 
            of the appropriate size is allocated. If the name is ``None``, the 
            first configuration entry is used.
        :type window_name: str
        :param output_type: The output port of the CMACC is driven by a multiplexer,
            which allows the user to decide which data is written into memory. 
            The output port has 32 bits per quadrature, but the internal accumulator
            has 48 bits per quadrature, and therefore it is left to the user to
            decide whether the upper or lower 32 bits are presented to the output.
            Using the upper 32 bits reduces the precision of the output but reduces
            the probability of overflow. Alternatively, the stream of data entering
            the accumulator may be passed to the output port rather than the 
            accumulated data, or the memory write may be cancelled altogether.
        :type output_type: str, one of "upper", "lower", "input", or None
        :param output_last_only: The accumulator processes streams of data present
            at its input and creates a stream at its output. However, there are many
            situations in which only the final value is required, such as when only
            the complete sum of a trace is required and the partial sums created while
            the trace is being captured are not. In these situations, the output port
            of the CMACC should only write a single value at the end of the trace.
            This behavior is specified via this flag. If `False`, a value is written 
            to the output port every decimation cycle.
        :type output_last_only: bool
        """

        # First, we'll determine what window we need
        # Here we'll cache allocated windows. If we previously used a window, 
        # we'll retrieve its memory, otherwise we'll allocate it fresh
        if window_name is None:
            window_name = list(self._capture.get_config("windows").keys())[0]
            
        if window_name in self._windows:
            # Regardless of what the window is, if we have it already,
            # we'll pass in its WaveformMemory so that we don't re-allocate it
            kernel_arg = self._windows[window_name]
            window_cache_key = window_name

            # Because the offset is just a number (and not a thing that needs
            # to be allocated), we can just get it from the config every time
            cmacc_offset = self._capture._config["windows"][window_name].get("offset", (0,0))
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
            window_cache_key = window_name
            cmacc_offset = self._capture._config["windows"][window_name].get("offset", (0,0))

        # If we've used this measurement before, retrieve the stream, otherwise create a new one
        src_arg = self._capture._channel if self._stream is None else self._stream

        # Configure the CMACC and retrieve the allocated stream and window memory
        stream, window_mem = self._capture._acadia.configure_cmacc(
            src=src_arg,
            kernel=kernel_arg,
            write_mode=output_type,
            last_only=output_last_only,
            reset_fifo=reset_fifo,
            accumulator_done=False
        )

        # Cache the relevant stream and window
        # If we had already allocated and cached these, this will just store 
        # them back with the same values
        self._stream = stream
        self._windows[window_cache_key] = window_mem

        # Convert the offset appropriately so that negative numbers are 
        # accepted (constants in the compiler must be unsigned)
        offset_converted = [int(np.int32(q).astype(np.uint32)) for q in cmacc_offset]
        self._capture._acadia.cmacc_load(stream, offset_converted)

    def measure(self, 
                stimulus_waveform_memory: Union[str, WaveformMemory] = None, 
                capture_waveform_memory: Union[str, WaveformMemory] = None):
        """
        Schedules the measurement. This function should be called inside of a 
        channel synchronizer, and prepare_cmacc must have been called before 
        this. 
        """
        self._capture_waveform = self._capture.get_waveform_memory(capture_waveform_memory)
        self._stimulus_waveform = self._stimulus.get_waveform_memory(stimulus_waveform_memory)
        self._stimulus._acadia.schedule_waveform(self._stimulus_waveform)
        self._capture._acadia.stream(self._stream, self._capture_waveform)

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

    def pulse(self, waveform_memory: Union[str, WaveformMemory] = None) -> None:
        """
        Apply a pulse to the qubit. The waveform memory parameter is passed directly to 
        :meth:`InputOutput.get_waveform_memory`; see documentation therein for 
        argument behavior.
        """

        self._stimulus._acadia.schedule_waveform(self._stimulus.get_waveform_memory(waveform_memory))

    def prepare(self, 
                state_quadrant: Literal[1,2,3,4],
                measurement_resonator: MeasurableResonator,
                pulse_waveform_memory: str = None,
                measurement_stimulus_waveform_memory: str = None,
                measurement_capture_waveform_memory: str = None,
                measurement_cmacc_window: str = None) -> None:
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

        """

        a = self._stimulus._acadia
        quadrant_reg_value = getattr(a, f"CMACC_QUADRANT_{state_quadrant}")
        reg = a.sequencer().Register()
        
        ## Step 1: Do a first msmt and store the result in a register
        measurement_resonator.prepare_cmacc(measurement_cmacc_window)
        with a.channel_synchronizer():
            self.pulse(pulse_waveform_memory)
            a.barrier()
            measurement_resonator.measure(measurement_stimulus_waveform_memory,
                                          measurement_capture_waveform_memory)

        reg.load(measurement_resonator.get_measurement())

        ## Step 2: Measure + conditional flip, until we get the target state
        # todo: try getting the number of msmts in this loop using a counter register
        with a.sequencer().repeat_until(reg == quadrant_reg_value):
            measurement_resonator.prepare_cmacc(measurement_cmacc_window)

            with a.channel_synchronizer():
                self.pulse(pulse_waveform_memory)
                a.barrier()
                measurement_resonator.measure(measurement_stimulus_waveform_memory,
                                              measurement_capture_waveform_memory)

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

    def pulse(self, waveform_memory: Union[str, WaveformMemory] = None) -> None:
        """
        Apply a pulse to the qubit. The waveform memory parameter is passed directly to 
        :meth:`InputOutput.get_waveform_memory`; see documentation therein for 
        argument behavior.
        """

        self._stimulus._acadia.schedule_waveform(self._stimulus.get_waveform_memory(waveform_memory))
