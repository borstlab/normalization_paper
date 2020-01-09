import torch
from torch import nn

# Model definitions for biologically plausible convolutional filters & motion detection


class FilterBlock(nn.Module):
    """Spatiotemporally separable, causal convolution block."""

    def __init__(self, shape, c_in, c_fan):
        """Parameters:
            shape - 3-element list of shape in azimuth x elevation x timepoints
            c_in - number of incoming channels
            c_fan - factor by which channels fan out"""

        super().__init__()

        fl = shape[2]
        c_out = c_fan * c_in

        pad1, pad2 = self._calculate_pad(shape[0]), self._calculate_pad(shape[1])
        self.layer_padding = nn.ConstantPad3d((fl - 1, fl - 1, pad2, pad2, pad1, pad1), 0.0)
        self.layer_spatial = nn.Conv3d(c_in, c_out, (shape[0], shape[1], 1), bias=False, groups=c_in)
        self.layer_temporal = nn.Conv3d(c_out, c_out, (1, 1, fl), bias=False, groups=c_out)

    def forward(self, x):
        keep = x.shape[-1]
        x = self.layer_padding(x)
        x = self.layer_spatial(x)
        x = self.layer_temporal(x)
        x = x[..., :keep]
        return x

    def postprocess(self):
        pass

    @staticmethod
    def _calculate_pad(x):
        return (x - 1) // 2


class NormalizationBlock(nn.Module):
    """Implementation of a neuro-inspired tanh-based normalization layer."""

    def __init__(self, shape, c_in, normalization_mode):
        """normalization_mode can be one of the following:
            -- none: No non-linearity is applied
            -- static: The non-linearity is fixed
            -- dynamic: The non-linearity is a function of local contrast"""

        assert normalization_mode in ["none", "static", "dynamic"]

        super().__init__()
        self.normalization_mode = normalization_mode

        if normalization_mode in ["dynamic"]:
            self.layer_rf = FilterBlock(shape=(shape[0], shape[1], 1), c_in=c_in, c_fan=1)
            self.layer_rf.layer_spatial.weight.data.fill_(0.0001)
            self.layer_rf.layer_temporal.weight.data.fill_(1.0)
            self.layer_rf.layer_temporal.weight.requires_grad = False

        if normalization_mode in ["static", "dynamic"]:
            self.squeeze_factor = nn.Parameter(torch.FloatTensor([1.0]))

    def postprocess(self):
        if self.normalization_mode in ["dynamic"]:
            self.layer_rf.layer_spatial.weight.data = torch.clamp(self.layer_rf.layer_spatial.weight.data, min=0.0)

        if self.normalization_mode in ["static", "dynamic"]:
            self.squeeze_factor.data = torch.clamp(self.squeeze_factor.data, min=0.001)

    def forward(self, x):
        if self.normalization_mode == "none":
            pass
        elif self.normalization_mode == "static":
            nf = self.squeeze_factor
            x = torch.tanh(x / nf)
        elif self.normalization_mode == "dynamic":
            nf = self.layer_rf(torch.abs(x)) + self.squeeze_factor
            x = torch.tanh(x / nf)

        return x


class SimpleModel(nn.Module):
    """Implementation of a multi-layer motion detection network with a fixed local multiplication stage"""

    def __init__(self, normalization_mode="none"):
        super().__init__()

        self.layer_input1 = FilterBlock(shape=(3, 3, 30), c_in=1, c_fan=2)
        self.layer_norm1 = NormalizationBlock(shape=(11, 11), c_in=2, normalization_mode=normalization_mode)
        self.factor_amplification = nn.Parameter(torch.FloatTensor([1.0]))

    def postprocess(self):
        self.layer_input1.postprocess()
        self.layer_norm1.postprocess()

    def forward(self, x):
        # Linear:
        x = self.layer_input1(x)
        x = self.layer_norm1(x)

        # Fixed RD part:
        f1 = x[:, 0:1, ...]
        f2 = x[:, 1:2, ...]
        rd1 = f1[:, :, 1:, :, :] * f2[:, :, :-1, :, :]
        rd2 = f2[:, :, 1:, :, :] * f1[:, :, :-1, :, :]

        # Linear combination:
        rd = (rd1 - rd2).mean(dim=2).mean(dim=2)
        output = self.factor_amplification * rd

        return output.squeeze()
