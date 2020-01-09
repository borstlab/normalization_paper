import numpy as np
import cv2
import matplotlib.pyplot as plt


class SimpleOptics:

    @staticmethod
    def get_standard_parameters():
        return {
            "fwhm": 5.7,
            "dphi": 5.0,
            "ds": 1.0,
        }

    def __init__(self, fwhm, dphi, ds):
        self.sigma = fwhm / 2.355
        self.dphi = dphi
        self.ds = ds
        
        self.dphi_pix = int(self.dphi / self.ds)
        
    def process(self, inp, debug=False):
        az, el = inp.shape[0] * self.ds, inp.shape[1] * self.ds
        
        # Gaussian filtering:
        filtered = np.zeros_like(inp)
        pix_sigma = self.sigma / self.ds
        for fidx in range(inp.shape[-1]):
            filtered[..., fidx] = cv2.GaussianBlur(inp[..., fidx], (0, 0), pix_sigma,
                                              borderType=cv2.BORDER_REPLICATE)
            
        # Subsample:
        n_az = int(az / self.dphi)
        n_el = int(el / self.dphi)
        
        rec_pos_az = np.arange(1, n_az) * self.dphi_pix
        rec_pos_el = np.arange(1, n_el) * self.dphi_pix
        
        g_el, g_az = np.meshgrid(rec_pos_el, rec_pos_az)
        
        if debug:
            plt.figure()
            plt.matshow(inp[:, :, 0].T, cmap="gray", interpolation="nearest")
            plt.scatter(g_az, g_el, color="red", s=3)
            return filtered[g_az, g_el, :], (g_az, g_el)
        
        return filtered[g_az, g_el, :]
