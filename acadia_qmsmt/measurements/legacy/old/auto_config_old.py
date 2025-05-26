from copy import deepcopy
from typing import Callable, Literal, Dict

from acadia import Acadia
from acadia.waveforms import ChannelWaveformMemory
from acadia.rfdc import Channel




# todo: integrate this to acadia so we don't need to send this to the board every time.
#  but this also means things here need to be more generic (other than being specific to a project)


class AutoConfigMixin:
    """
    takes the overall config dict and automatically applies some configurations
    """
    FILE = __file__
    
    def obtain_channels(self, acadia: Acadia, **configs) -> Dict[str, Channel]:
        """
        Obtain all acadia channels used in the configs dict and put them in a collection.

        :param acadia: Acadia instance for creating channels.
        :param configs: Config dicts for each channel. #todo: make a schema
        :return:
        """
        self.channel_configs = configs
        self.channel_objs = {}
        for ch_name, config in configs.items():
            self.channel_objs[ch_name] = acadia.channel(config["channel"])

        return self.channel_objs


    def allocate_waveform_mem(self, acadia: Acadia, channel_name: str, waveform_name: str) -> ChannelWaveformMemory:
        """
        A shortcut function for allocating waveform memory for a specified channel and waveform
        using pre-defined configurations.

        Serves as a shortcut for creating waveform objects based on the parameters defined in
        configs[`channel_name`]["waveforms"][`waveform_name`].

        :param acadia: Acadia instance for creating waveforms.
        :param channel_name: Name of the channel where the waveform will be used
        :param channel_name: Name of the waveform defined under the "waveforms" sub-dictionary for the given channel
        :return: Allocated WaveformMemory object
        """
        channel_obj = self.channel_objs[channel_name]
        wf_cgf = self.channel_configs[channel_name]["waveforms"][waveform_name]
        return acadia.create_waveform_memory(channel_obj, **wf_cgf)


    def allocate_all_waveform_mems(self, acadia: Acadia, **configs):
        """
        Automatically allocates all the waveforms listed in the configs dict, and put the waveform objects in
        `self.channel_waveforms` dict.

        For individual waveform, use `allocate_waveform_mem` instead.

        todo: This function was intended to reduce boilerplate. However, since the created waveform objects will be used
            in multiple places later in the runtime code, the nested dicts created here also makes them cumbersome to
            access...

        :param acadia:
        :param configs:
        :return:
        """
        # make a nested dict of wave form objects for each channel for later access
        self.channel_waveforms = {}  # ["channel_name"]["wf_mem_name"] = wf_mem_obj
        for ch_name, config in configs.items():
            self.channel_waveforms[ch_name] = {}
            for wf_name, wf_cgf in config["waveforms"].items():
                self.channel_waveforms[ch_name][wf_name] = self.allocate_waveform_mem(acadia, ch_name, wf_name)


    def auto_config_ncos(self, acadia: Acadia, nco_update_event_source: Literal["sysref", "immediate"] = "sysref",
                         reset_phases=True, align_tile=True, **configs):
        """
        Automatically configures NCO (Numerically Controlled Oscillator) parameters for channels in the `configs` dict.

        :param acadia: Acadia instance
        :param nco_update_event_source: The source of the NCO update event. Should be either:
                                        - "sysref" (default): Updates are synchronized to a system reference.
                                        - "immediate": Updates occur immediately without synchronization.
        :param reset_phases: When True, reset NCO phases to 0 after configuration.
        :param align_tile: When True, align tile latencies before configuration
        :param configs: Config dicts for each channel
        """
        if align_tile:
            acadia.align_tile_latencies()
        for ch_name, config in configs.items():
            channel_obj = self.channel_objs[ch_name]
            # Configure  analog parameters for each channel
            channel_obj.set(nco_update_event_source=nco_update_event_source, **config["nco_config"])
            if nco_update_event_source == "immediate":
                channel_obj.nco_immediate_update_event()
            if reset_phases:
                acadia.reset_nco_phase(channel_obj)
                acadia.update_nco_phase(channel_obj, 0)

        if nco_update_event_source == "sysref":
            acadia.update_ncos_synchronized()


    def blank_waveform_generator(self, acadia: Acadia, channel_name: str, reference_waveform: str = None) -> Callable:
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
                for i, wf_cfg in enumerate(self.channel_configs[channel_name]["waveforms"].values()):
                    d, r = wf_cfg.get("decimation", None), wf_cfg.get("region", None)
                    if i == 0:
                        decimation, region = d, r
                    if decimation != d or region != r:
                        raise ValueError(
                            f"for blank waveform in an ADC channel, a reference waveform name must be provided")
            else:
                ref_wf_cfg = self.channel_configs[channel_name]["waveforms"][reference_waveform]
                decimation = ref_wf_cfg.get("decimation", None)
                region = ref_wf_cfg.get("region", None)

            if decimation == 0:
                decimation = 4  # for cmacc # todo: need to test
            waveform_args.update({"decimation": decimation, "region": region})

        def blank_wf_gen(length: float) -> ChannelWaveformMemory:
            blank_wf = acadia.create_waveform_memory(channel_obj, length=length, **waveform_args)
            return blank_wf

        return blank_wf_gen
