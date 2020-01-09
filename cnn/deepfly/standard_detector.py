import numpy as np
from .utils import lowpass, highpass


class SimpleReichardt:

    @staticmethod
    def get_standard_parameters():
        return {
            "tau_hp": 0.25,
            "tau_lp": 0.05,
            "tau_lipetz": 0.75,
            "dt": 0.01,
        }

    def __init__(self, tau_hp, tau_lp, tau_lipetz, dt):
        self.tau_hp = tau_hp
        self.tau_lp = tau_lp
        self.tau_lipetz = tau_lipetz
        self.dt = dt
    
    def process(self, inp):
        
        # Phototransduction:
        if self.tau_lipetz is not None:
            lsignal = lowpass(inp, self.tau_lipetz, self.dt)
            inp = inp / (inp + lsignal)
        
        inp = highpass(inp, self.tau_hp, self.dt)
        filtered = lowpass(inp, self.tau_lp, self.dt)
        
        a = inp[1:, :, :] * filtered[:-1, :, :]        
        b = inp[:-1, :, :] * filtered[1:, :, :]
        c = inp[:, 1:, :] * filtered[:, :-1, :]
        d = inp[:, :-1, :] * filtered[:, 1:, :]

        return a, b, c, d
