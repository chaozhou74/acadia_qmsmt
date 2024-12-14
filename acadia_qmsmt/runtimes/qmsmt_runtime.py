import sys
from typing import Callable, Literal, Dict, List

from acadia.waveforms import ChannelWaveform
from acadia import Acadia

class QMsmtRuntimeBase():
    """
    A base class that provides shortcut functions that simplifies the configuration of analog channels 
    and pulse waveforms based on the provided configuration dict.

    This will be the base class of other runtime classes that uses configuration based on a yaml file.

    """

    def __init__(self, **channel_configs: dict):
        """
        Gather the configuration dict and necessary python modules that will be send to the board.
        
        This should be called in the `__post_init__` of the child runtime dataclass, to gather the modules used on the
        host.

        :param channel_configs:
        """

        self.config_dict = channel_configs
        self._gather_files()


    def _gather_files(self) -> List[str]:
        """
        Gather the file paths of the python modules used to define the child 
        Runtime dataclass.

        :return: list of paths to the gathered files
        """
        files = []
        for cls in self.__class__.__mro__:
            if cls is QMsmtRuntimeBase:  # Stop at this base class
                break
            module_name = cls.__module__
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if hasattr(module, "__file__"):
                    files.append(module.__file__)
        
        self.FILES = [*set(files)] # remove duplicate files
        return files
    

    def init_system(self) -> Acadia:
        """
        Create an instance of acadia, and obtain the DAC and ADC channels that will be used.
        
        This should be called at the beginning of the `main()` function of the child runtime class,
        so this will only run on the board.

        :return:
        """
        self.acadia = Acadia()

        # Obtain all acadia channels used in the configs dict and put them in a collection.
        self.channel_objs = {}
        for ch_name, config in self.config_dict.items():
            self.channel_objs[ch_name] = self.acadia.channel(config["channel"])

        return self.acadia


    def allocate_waveform_mem(self, channel_name: str, waveform_name: str) -> ChannelWaveform:
        """
        A shortcut function for allocating waveform memory for a specified channel and waveform
        using pre-defined configurations.

        Serves as a shortcut for creating waveform objects based on the parameters defined in
        configs[`channel_name`]["waveforms"][`waveform_name`].

        :param channel_name: Name of the channel where the waveform will be used
        :param waveform_name: Name of the waveform defined under the "waveforms" sub-dictionary for the given channel
        :return: Allocated Waveform object
        """
        channel_obj = self.channel_objs[channel_name]
        wf_cgf = self.config_dict[channel_name]["waveforms"][waveform_name]
        return self.acadia.create_waveform(channel_obj, **wf_cgf)


    def configure_ncos(self, nco_update_event_source: Literal["sysref", "immediate"] = "sysref",
                       reset_phases=True, align_tile=True):
        """
        Automatically configures NCO (Numerically Controlled Oscillator) parameters for channels in the `configs` dict.

        :param nco_update_event_source: The source of the NCO update event. Should be either:
                                        - "sysref" (default): Updates are synchronized to a system reference.
                                        - "immediate": Updates occur immediately without synchronization.
        :param reset_phases: When True, reset NCO phases to 0 after configuration.
        :param align_tile: When True, align tile latencies before configuration
        """
        if align_tile:
            self.acadia.align_tile_latencies()
        for ch_name, config in self.config_dict.items():
            channel_obj = self.channel_objs[ch_name]
            # Configure  analog parameters for each channel
            channel_obj.set(nco_update_event_source=nco_update_event_source, **config["nco_config"])
            if nco_update_event_source == "immediate":
                channel_obj.nco_immediate_update_event()
            if reset_phases:
                self.acadia.reset_nco_phase(channel_obj)
                self.acadia.update_nco_phase(channel_obj, 0)

        if nco_update_event_source == "sysref":
            self.acadia.update_ncos_synchronized()


    def blank_waveform_generator(self, channel_name: str, reference_waveform: str = None) -> Callable:
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
        channel_obj = self.channel_objs[channel_name]
        waveform_args = {"blank": True}

        # get decimation and wf region for blank wf in adc channels
        if not channel_obj.is_dac:
            if reference_waveform is None:
                # check if all the waveform mem configuration in the channel has the same settings
                for i, wf_cfg in enumerate(self.config_dict[channel_name]["waveforms"].values()):
                    d, r = wf_cfg.get("decimation", None), wf_cfg.get("region", None)
                    if i == 0:
                        decimation, region = d, r
                    if decimation != d or region != r:
                        raise ValueError(
                            f"for blank waveform in an ADC channel, a reference waveform name must be provided")
            else:
                ref_wf_cfg = self.config_dict[channel_name]["waveforms"][reference_waveform]
                decimation = ref_wf_cfg.get("decimation", None)
                region = ref_wf_cfg.get("region", None)

            if decimation == 0:
                decimation = 4  # for cmacc 
            waveform_args.update({"decimation": decimation, "region": region})

        def blank_wf_gen(length: float) -> ChannelWaveform:
            blank_wf = self.acadia.create_waveform(channel_obj, length=length, **waveform_args)
            return blank_wf

        return blank_wf_gen


    def __getitem__(self, channel_name:str) -> Dict:
        """
        Directly get the config dict of a channel in the overall config dict

        :param channel_name: name of a channel that exist in the yaml config file.
        :return: Config dict of the required channel
        """
        return self.config_dict[channel_name]


class SingleQubitRuntime(QMsmtRuntimeBase):
    def __init__(self, **config_dict: dict):
        super().__init__(**config_dict)

    # todo: more experiment specific functions can be added here. 
    #   e.g. sub_sequence functions
