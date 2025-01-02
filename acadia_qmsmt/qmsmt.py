import sys
import shutil
import os
from typing import Callable, Literal, Dict, List, Any

from acadia import Acadia, Runtime, ChannelWaveform, Waveform

__all__ = ["InputOutput", "QMsmtRuntime", "MeasurableResonator", "MeasurableQubit"]

class InputOutput:
    """
    A base class for abstracting input and output channels. In some
    sense this abstracts Acadia's `Channel` object, but at a higher level
    so that certain patterns (such as for allocating memory) can be carried
    out with full knowledge of the configuration as provided by a dictionary.
    """

    def __init__(self, acadia: Acadia, config: Dict[str,Any]):
        self._config = config
        self._acadia = acadia
        self._channel = acadia.channel(config["channel"])
        self._allocated_waveforms: Dict[str,ChannelWaveform] = {}


    def get_waveform(self, waveform_name: str = None) -> ChannelWaveform:
        """
        A shortcut function for allocating waveform memory for a specified channel and waveform
        using pre-defined configurations.

        Serves as a shortcut for creating waveform objects based on the parameters defined in
        config["waveforms"][`waveform_name`].

        If no name is provided, the first waveform in the list of available waveforms is used.

        :param waveform_name: Name of the waveform defined under the "waveforms" sub-dictionary for the given channel
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
    def channel(self):
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
        self._ios = {}
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
        # Store a list of used 
        self._capture_waveforms = []
        self._stimulus_waveforms = []

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
        Schedules the measurement in the channel synchronizer. Note that 
        prepare_cmacc must have been called before this.
        """
        self._capture_waveform = self._capture.get_waveform(capture_waveform_name)
        self._stimulus_waveform = self._stimulus.get_waveform(stimulus_waveform_name)
        self._stimulus._acadia.schedule_waveform(self._stimulus_waveform)
        self._capture._acadia.stream(self._stream, self._capture_waveform)


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


class MeasurableQubit:
    """
    A collection of functions that are useful for manipulating and measuring a qubit
    with a dispersive readout resonator.
    """

    def __init__(self, **config_dict: dict):
        """
        Runtime class that contains experiment specific functions, including sub_sequence functions
        """
        super().__init__(**config_dict)

    def subsqeuence_cool_qubit(self, acadia:Acadia, capture_cfg:dict, capture_dst:Waveform,
                               ro_drive_wf:ChannelWaveform, capture_blank_wf:ChannelWaveform,
                               q_pi_pulse_wf:ChannelWaveform, q_blank_wf:ChannelWaveform):
        """
        A sub-sequence that cools the qubit to its ground state using measurement and conditional flips.

        :param acadia: Acadia instance where the sequence belongs to
        :param capture_cfg: configuration dict for the  ADC capture channel
        :param capture_dst: destination memory for the captured measurement data
        :param ro_drive_wf: readout drive waveform for the measurement pulse
        :param capture_blank_wf: blank waveform for the delay between ro drive and capture
        :param q_pi_pulse_wf: qubit pi_pulse waveform for flipping the qubit when measured in e
        :param q_blank_wf: qubit blank waveform between qubit flip and measurement
        :return:
        """

        capture_delay = capture_cfg.get("capture_delay", 0)
        cmacc_offset = capture_cfg.get("cmacc_offset", 0)
        kernel_wf = capture_cfg.get("kernel_wf", [0.1])
        g_quadrant = capture_cfg["state_quadrants"][0]

        a = acadia

        ## Step 1: Do a first msmt
        capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel_wf,
                                                   reset_fifo=True, accumulator_done=False)

        a.cmacc_load(capture_stream, cmacc_offset)

        with a.channel_synchronizer():
            if capture_delay != 0:
                a.schedule_waveform(capture_blank_wf)
            a.schedule_waveform(ro_drive_wf)
            a.stream(capture_stream, capture_dst)

        # put the first msmt result in a register
        reg = a.sequencer().Register()
        reg.load(a.cmacc_get_quadrant(capture_stream))

        ## Step 2: Measure + conditional flip, until we get the ground state
        # todo: try getting the number of msmts in this loop using a counter register
        with a.sequencer().repeat_until(reg == getattr(a, f"CMACC_QUADRANT_{g_quadrant}")):
            capture_stream, kernel = a.configure_cmacc(self.channel_objs["ro_capture"], kernel=kernel,
                                                       reset_fifo=True, accumulator_done=False)

            a.cmacc_load(capture_stream, cmacc_offset)  # todo: extra bias can be added here.

            with a.channel_synchronizer():
                a.schedule_waveform(q_pi_pulse_wf)
                a.schedule_waveform(q_blank_wf)
                a.barrier()
                if capture_delay != 0:
                    a.schedule_waveform(capture_blank_wf)
                a.schedule_waveform(ro_drive_wf)
                # this will keep overwriting the re_pts1 waveform
                # the final value will always be in the g state quadrant since that's the condition for exiting the loop
                a.stream(capture_stream, capture_dst)

            # `cmacc_get_quadrant` will wait until cmacc is done
            reg.load(a.cmacc_get_quadrant(capture_stream))
