from copy import deepcopy

from acadia import Acadia


FILE = __file__


#todo: this should be moved to a separate submodule... But need to figure out a easy way to send this to the board

# todo: needs to also configure have separate signals.


class AutoConfigMixin:
    """
    takes the overall config dict and automatically applies some configurations
    """

    def auto_config_channels(self, acadia: Acadia, **configs):
        """
        Automatically configure channels with given configurations dicts and add them to a collection.
        """
        self.channel_configs = configs
        self.channel_objs = {}
        for ch_name, config in configs.items():
            self.channel_objs[ch_name] = acadia.channel(config["channel"])


    def auto_config_waveform_mems(self, acadia: Acadia, **configs):
        """

        :param acadia:
        :param configs:
        :return:
        """
        # make a nested dict of wave form objects for each channel for later access
        self.channel_waveforms = {} # ["channel_name"]["wf_mem_name"] = wf_mem_obj
        for ch_name, config in configs.items():
            self.channel_waveforms[ch_name] = {}
            channel_obj = self.channel_objs[ch_name]
            # config each waveform in the channel
            for wf_name, wf_cgf in config["waveforms"].items():
                wf_obj = acadia.create_waveform(channel_obj, **wf_cgf)
                self.channel_waveforms[ch_name][wf_name] = wf_obj

    def auto_config_ncos(self, acadia: Acadia, nco_update_event_source="sysref", 
                         reset_phases=True, align_tile=True, **configs):
        """

        :param acadia:
        :param nco_update_event_source:
        :param reset_phases:
        :param configs:
        :return:
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



    def blank_waveform_generator(self, acadia: Acadia, channel:str, reference_waveform:dict=None):
        """function for making blank waveforms for dac or adc channels

        :param acadia: 
        :param channel: name of the DAC or ADC channel
        :param reference_waveform: for ADC channel, a reference waveform needs to provided to decide the
            `decimation` and `region`
        """
        channel_obj = self.channel_objs[channel]
        waveform_args = {"blank": True}

        # get decimation and wf region for blank wf in adc channels
        if not channel_obj.is_dac:
            if reference_waveform is None:
                # check if all the waveform mem configuration in the channel has the same settings
                for i, wf_cfg in enumerate(self.channel_configs[channel]["waveforms"].values()):
                    d, r = wf_cfg.get("decimation", None), wf_cfg.get("region", None)
                    if i == 0:
                        decimation, region = d, r
                    if decimation != d or region != r:
                        raise ValueError(f"for blank waveform in an ADC channel, a reference waveform name must be provided")
            else:
                ref_wf_cfg = self.channel_configs[channel]["waveforms"][reference_waveform]
                decimation = ref_wf_cfg.get("decimation", None)
                region = ref_wf_cfg.get("region", None)
            
            waveform_args.update({"decimation": decimation, "region": region})
        
        def blank_wf_gen(length:float):
            blank_wf = acadia.create_waveform(channel_obj, length=length, **waveform_args)
            return blank_wf
        
        return blank_wf_gen
        