import sys
import shutil
import os
from typing import Callable, Literal, Dict, List, Any

from acadia import Acadia, Channel, Runtime, ChannelWaveform, Waveform, Operation

__all__ = ["InputOutput", "QMsmtRuntime", "MeasurableResonator", "MeasurableQubit"]

class InputOutput:
    """
    A base class for abstracting input and output channels. In some
    sense this abstracts Acadia's `Channel` object, but at a higher level
    so that certain patterns (such as for allocating memory) can be carried
    out with full knowledge of the configuration as provided by a dictionary.
    """

    def __init__(self, acadia: Acadia, config: Dict[str,Any]):
        self._config: Dict[str,Any] = config
        self._acadia: Acadia = acadia
        self._channel: Channel = acadia.channel(config["channel"])
        self._allocated_waveforms: Dict[str,ChannelWaveform] = {}


    def get_waveform(self, waveform_name: str = None) -> ChannelWaveform:
        """
        A shortcut function for allocating waveform memory for a specified 
        channel and waveform using pre-defined configurations.

        Serves as a shortcut for creating waveform objects based on the 
        parameters defined in config["waveforms"][`waveform_name`].

        If no name is provided, the first waveform in the list of available
        waveforms is used.

        :param waveform_name: Name of the waveform defined under the 
            "waveforms" sub-dictionary for the given channel
        :return: Allocated Waveform object
        """
        if waveform_name is None:
            waveform_name = self._config["waveforms"].keys()[0]

        # If we haven't yet allocated memory for the waveform, do so
        if waveform_name not in self._allocated_waveforms:
            wf_cgf = self._config["waveforms"][waveform_name]
            self._allocated_waveforms[waveform_name] = self._acadia.create_waveform(self._channel, **wf_cgf)

        return self._allocated_waveforms[waveform_name]


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
                for i, wf_cfg in enumerate(self._config["waveforms"].values()):
                    d, r = wf_cfg.get("decimation", None), wf_cfg.get("region", None)
                    if i == 0:
                        decimation, region = d, r
                    if decimation != d or region != r:
                        raise ValueError(
                            f"for blank waveform in an ADC channel, a reference waveform name must be provided")
            else:
                ref_wf_cfg = self._config["waveforms"][reference_waveform]
                decimation = ref_wf_cfg.get("decimation", None)
                region = ref_wf_cfg.get("region", None)

            if decimation == 0:
                decimation = 4  # for cmacc 
            waveform_args.update({"decimation": decimation, "region": region})

        def blank_wf_gen(length: float) -> ChannelWaveform:
            blank_wf = self.acadia.create_waveform(self._channel, length=length, **waveform_args)
            return blank_wf

        return blank_wf_gen

    def load_memory(self, **kwargs):
        """
        Populate the memory of previously allocated waveforms. The names of 
        the keyword arguments should correspond to waveform names, and the 
        values should be either strings that correspond to signal names in the
        config dictionary or a value to be directly passed to :meth:`Waveform.set`.
        """

        for waveform_name, signal in kwargs.items():
            if waveform_name not in self._allocated_waveforms:
                raise ValueError(f"Waveform {waveform_name} requested to be "
                                    "loaded but was never allocated.")
            set_value = self._config["signals"][signal] if isinstance(signal, str) else signal
            if isinstance(set_value, dict):
                self._allocated_waveforms[waveform_name].set(**set_value)
            else:
                self._allocated_waveforms[waveform_name].set(set_value)

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
    configuration of analog channels and pulse waveforms based on the 
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

        - If the value of the field is a string, it is expected that the execution
        directory of the script contains a file called "config.yaml", and that this
        file contains a top-level section with a name matching the field value. The
        configuration for the channel is then retrieved by loading the file and
        accessing the corresponding section.

        - If the value of the field is a tuple, it is expected that the first
        element is the name of a top-level section in a yaml file whose path 
        is provided in the second element. 

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.acadia = Acadia()  

        # Store the names and config dicts for all the 
        # members identified as channel configurations
        self._ios: Dict[str,InputOutput] = {}
        for name,type_hint in self._get_fields().items():
            if type_hint is IOConfig:
                value = getattr(self, name)
                if isinstance(value, tuple):
                    from acadia_qmsmt.helpers.yaml_editor import load_yaml
                    config_dict = load_yaml(value[1])[value[0]]
                    config_dict["yaml_path"] = value[1]
                elif isinstance(value, str):
                    from acadia_qmsmt.helpers.yaml_editor import load_yaml
                    config_dict = load_yaml("config.yaml")[value]
                    config_dict["yaml_path"] = "config.yaml"
                elif isinstance(value, dict):
                    config_dict = value
                    config_dict["yaml_path"] = None
                else:
                    raise TypeError(f"Unable to interpret value of type"
                                    f" {type(value)} as an IO configuration")

                self._ios[name] = InputOutput(self.acadia, config_dict)
        
    def _dump_fields(self, fields: dict = None):
        
        # Dump all the arguments to a JSON file as in the parent class, but replace
        # IOConfig arguments with their config dicts
        fields = {}
        for name,type_hint in self._get_fields().items():
            if type_hint is IOConfig:
                fields[name] = self._ios[name]._config
            else:
                fields[name] = getattr(self, name)

        super()._dump_fields(fields)

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
            io.channel.set(nco_update_event_source=nco_update_event_source, **io._config["channel_config"])
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

        if isinstance(getattr(self, ioconfig_name), dict):
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
        else:
            # Update the yaml
            from acadia_qmsmt.helpers.yaml_editor import update_yaml
            update_yaml(self._ios[ioconfig_name]._config["yaml_path"], {config_field: value})


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
        self._kernels = {}

    def prepare_cmacc(self, 
                kernel_name: str = None,
                output_type: Literal["upper", "lower", "input", None] = "upper",
                output_last_only: bool = True):
        """
        Perform a measurement of the resonator. 

        The length and decimation of the capture are retrieved from the provided 
        capture configuration. Because a CMACC is always used for the measurement, 
        the decimation of the specified capture configuration must be a multiple of 4. 
        A value of 0 may also be provided, in which case
        the decimation factor will automatically be chosen so that the 
        entire captured trace is decimated into a single point.

        
        :param kernel_name: The name of the kernel to be used (and newly 
            allocated, if necessary). The name should be a key in the 
            "kernels" section of the configuration for the provided capture.
            The value of the kernel may be a float, in which case a boxcar kernel
            is allocated. Note that this does not actually populate the kernel
            memory with that float, but simply specifies that a boxcar kernel
            will be used so that only a single point in kernel memory is allocated.
            The value of the kernel may also be an array, in which case a kernel 
            of the appropriate size is allocated.
        :type kernel_name: str
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

        # First, we'll determine what kernel we need
        # Here we'll cache allocated kernels. If we previously used a kernel, 
        # we'll retrieve its memory, otherwise we'll allocate it fresh
        if kernel_name is None:
            if "_default" in self.kernels:
                # If we already allocated a boxcar kernel, use it
                kernel_arg = self._kernels["_default"]
                kernel_cache_key = "_default"
            else:
                # we'll create a boxcar kernel by passing None
                kernel_arg = None
                kernel_cache_key = "_default"
            cmacc_offset = 0
        elif kernel_name in self._kernels:
            # Regardless of what the kernel is, if we have it already,
            # we'll pass in its Waveform so that we don't re-allocate it
            kernel_arg = self._kernels[kernel_name]
            kernel_cache_key = kernel_name

            # Because the offset is just a number (and not a thing that needs
            # to be allocated), we can just get it from the config every time
            cmacc_offset = self._capture._config["kernels"][kernel_name].get("offset", 0)
        else:
            # Allocate the waveform anew 
            # We'll interpret the type of the argument slightly differently 
            # than configure_cmacc expects; if the kernel argument is a float,
            # we'll interpret this as the amplitude of a boxcar kernel, in contrast
            # to configure_cmacc interpreting a float argument as the length in seconds
            # that the kernel should be. The behavior here will be different as this is a
            # much more likely use case and will be much more intuitive in the config file
            kernel_config = self._capture._config["kernels"][kernel_name]["data"]
            kernel_arg = None if isinstance(kernel_config, float) else kernel_config
            kernel_cache_key = kernel_name
            cmacc_offset = self._capture._config["kernels"][kernel_name].get("offset", 0)

        # If we've used this measurement before, retrieve the stream, otherwise create a new one
        src_arg = self._capture._channel if self._stream is None else self._stream

        # Configure the CMACC
        stream, kernel = self._capture._acadia.configure_cmacc(
            src=src_arg,
            kernel=kernel_arg,
            write_mode=output_type,
            last_only=output_last_only,
            reset_fifo=True,
            accumulator_done=False
        )

        # Cache the relevant stream and kernel
        # If we had already allocated and cached these, this will just store 
        # them back with the same values
        self._stream = stream
        self._kernels[kernel_cache_key] = kernel

        self._capture._acadia.cmacc_load(stream, cmacc_offset)

    def measure(self, 
                stimulus_waveform_name: str = None, 
                capture_waveform_name: str = None):
        """
        Schedules the measurement. This function should be called inside of a 
        channel synchronizer, and prepare_cmacc must have been called before 
        this. 
        """
        self._capture_waveform = self._capture.get_waveform(capture_waveform_name)
        self._stimulus_waveform = self._stimulus.get_waveform(stimulus_waveform_name)
        self._stimulus._acadia.schedule_waveform(self._stimulus_waveform)
        self._capture._acadia.stream(self._stream, self._capture_waveform)

    def get_measurement(self) -> Operation:
        """
        Blocks until the measurement is complete, as indicated by the CMACC
        completion status bit.
        """

        # `Acadia.cmacc_get_quadrant` will wait until cmacc is done
        return self._capture._acadia.cmacc_get_quadrant(self._stream)


    def load_kernels(self) -> None:
        """
        Load the kernel memory with the values specified in the configuration.
        
        This should occur after the sequence has been compiled and the relevant
        Acadia object attached.
        """

        for kernel_name, kernel_memory in self._kernels.items():
            kernel_memory.set(self._capture._config["kernels"][kernel_name]["data"])


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

    def pulse(self, waveform_name: str = None) -> None:
        """
        Apply a pulse to the qubit. The waveform name is passed directly to 
        :meth:`InputOutput.get_waveform`; see documentation therein for 
        argument behavior.
        """

        waveform = self._stimulus.get_waveform(waveform_name)
        self._stimulus._acadia.schedule_waveform(waveform)

    def prepare(self, 
                state_target_name: str,
                measurement_resonator: MeasurableResonator,
                pulse_waveform_name: str = None,
                measurement_stimulus_waveform_name: str = None,
                measurement_capture_waveform_name: str = None,
                measurement_cmacc_kernel: str = None) -> None:
        """
        A subsequence that prepares the qubit in a given state by measuring and
        conditionally applying a waveform. All mneasurement names are passed 
        directly to :meth:`InputOutput.get_waveform`; see documentation therein 
        for argument behavior.

        :param state_target_name: The name of the target state to prepare. This
            should be a key in the `state_quadrants` section of the capture
            configuration, whose value is the quadrant number for a measurement
            result in the given state. For example, if a measurement of a system
            in state "g" would result in a value in the 4th quadrant of the IQ
            plane, the line `"g": 4` should be in the capture configuration,
            and state_target_name should be set to `"g"` to prepare that state.
        :type state_target_name: str 
        :param measurement_resonator: The resonator to be used for measurement
            and conditional operations.
        :type measurement_resonator: :class:`MeasurableResonator`
        :param pulse_waveform_name: The name of the waveform to be played when 
            the qubit is measured to be in any other state than the target. 
        :type pulse_waveform_name: str
        :param measurement_stimulus_waveform_name: The name of the measurement
            resonator stimulus waveform.
        :type measurement_stimulus_waveform_name: str
        :param measurement_stimulus_waveform_name: The name of the measurement
            resonator capture waveform.
        :type measurement_capture_waveform_name: str
        """

        a = self._stimulus._acadia
        target_quadrant = measurement_resonator._capture._config["state_quadrants"][state_target_name]
        quadrant_reg_value = getattr(a, f"CMACC_QUADRANT_{g_quadrant}")
        reg = a.sequencer().Register()
        
        ## Step 1: Do a first msmt and store the result in a register
        measurement_resonator.prepare_cmacc(measurement_cmacc_kernel)
        with a.channel_synchronizer():
            measurement_resonator.measure()

        reg.load(measurement_resonator.get_measurement())

        ## Step 2: Measure + conditional flip, until we get the ground state
        # todo: try getting the number of msmts in this loop using a counter register
        with a.sequencer().repeat_until(reg == quadrant_reg_value):
            measurement_resonator.prepare_cmacc(measurement_cmacc_kernel)

            with a.channel_synchronizer():
                self.pulse(pulse_waveform_name)
                a.barrier()
                measurement_resonator.measure()

            reg.load(measurement_resonator.get_measurement())
