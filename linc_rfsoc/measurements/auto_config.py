


FILE = __file__


#todo: this should be moved to a separate submodule... But need to figure out a easy way to send this to the board

class AutoConfigMixin:
    """
    takes the overall config dict and automatically applies some configurations
    """

    def auto_config_channels(self, acadia, **configs):
        """
        Automatically configure channels with given configurations dicts and add them to a collection.
        """
        self.channel_objs = {}
        for ch_name, config in configs.items():
            self.channel_objs[ch_name] = acadia.channel(config["channel"])


    def auto_config_waveform_mems(self, acadia, **configs):
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

    def auto_config_ncos(self, acadia, nco_update_event_source="sysref", reset_phases=True, **configs):
        """

        :param acadia:
        :param nco_update_event_source:
        :param reset_phases:
        :param configs:
        :return:
        """
        acadia.align_tile_latencies() # todo: might be removed?
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



    # todo: should have auto config for 'channel', 'datapath', ('waveform', and 'signal')
    #  consider renaming those also.....