import torch
from torch import nn


# Basic filters

class FilterConv(nn.Module):
    """Implementation of a causal convolutional filter block"""

    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        
    @staticmethod
    def causal_conv(inp, filt):
        filt = filt.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.conv3d(inp, filt, padding=(0, 0, filt.shape[-1]-1))[..., :inp.shape[-1]]
        return out
    
    def _generate_ir(self, x):
        raise NotImplementedError
        
    def forward(self, x):
        filt = self._generate_ir(x)
        x = self.causal_conv(x, filt)
        return x
    
    
class LowpassConv(FilterConv):
    """Implementation of first-order low-pass in PyTorch"""

    def __init__(self, tau, dt, filter_factor=10.0):
        super().__init__(dt)
        self.tau = torch.nn.Parameter(torch.FloatTensor([tau]))
        self.ff = filter_factor
        
    def _generate_ir(self, x):
        time = torch.arange(torch.round(self.ff * self.tau / self.dt).data[0], -1, -1, device=x.device) * self.dt
        ir_lp = torch.exp(-1 * time / self.tau)
        return ir_lp / ir_lp.sum()

    
class HighpassConv(LowpassConv):
    """Implementation of first-order high-pass in PyTorch"""

    def __init__(self, tau, dt, filter_factor=10.0):
        super().__init__(tau, dt, filter_factor)
        
    def _generate_ir(self, x):
        ir_lp = super()._generate_ir(x)
        probe = torch.zeros_like(ir_lp)
        probe[-1] = 1.0
        return probe - ir_lp
