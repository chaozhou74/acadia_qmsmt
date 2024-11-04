import matplotlib.pyplot as plt
import numpy as np

from linc_rfsoc.runtimes.readout import ReadoutRuntime

from IPython.core.getipython import get_ipython
get_ipython().run_line_magic("matplotlib", "widget")

if __name__ == "__main__":
    stimulus: dict = {
        "channel": "DAC2",

        "datapath": {
            "vop": 30000,
            "mix_reconstruction": True,
            "nco_frequency": 9.03e9
        },

        "waveform": {
            "length": 200e-9, 
            "fixed_length": 200e-9 
        },
        
        "signal": {
            "data": ("scipy", "hann"),
            "scale": 0.7
        }
    }

    capture: dict = {
        "channel": "ADC0",

        "datapath": {
            "nco_frequency": -9.03e9
        },

        "waveform": {
            "length": 12*1.25*100e-9,
            "decimation": 1,
            "region": "plddr"
        }
    }

    plot = True
    iterations = 100000


    rt = ReadoutRuntime(stimulus, capture, plot=plot, iterations=iterations)
    rt.deploy("10.66.3.198", "readout", files=[rt.FILE])    
    rt.display()

    # some ad hoc processing
    rt._event_loop.join()

    data = rt.data["traces"].records().astype(float)
    avg_pwr = np.mean(data[:,:,0]**2+data[:,:,1]**2, axis=0)
    
    plt.figure()
    plt.plot(rt.time_axis*1e9, avg_pwr)
    plt.xlabel("time (ns)")
    
    print("SigPwr/NoisePwr", np.mean(avg_pwr[600:1400])/np.mean(avg_pwr[2000:]))