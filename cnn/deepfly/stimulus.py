import numpy as np
import cv2

from .utils import lowpass1d


class SineGrating:

    @staticmethod
    def get_standard_parameters():
        return {
            "azimuth": 120.0,
            "elevation": 90.0,
            "duration": 5.0,
            "wavelength": 60.0,
            "velocity": 60.0,
            "orientation": 0.0,
            "initial_phase": 0.0,
            "start": 1.0,
            "stop": 4.0,
            "dt": 0.01,
            "ds": 1.0,
        }

    def __init__(self, azimuth, elevation, duration,
                 wavelength, velocity, orientation, initial_phase, start, stop,
                 dt, ds):
        self.az = azimuth
        self.el = elevation
        self.dur = duration
        self.lam = wavelength
        self.vel = velocity
        self.ori = orientation
        self.phase = initial_phase
        self.start = start
        self.stop = stop
        self.dt = dt
        self.ds = ds
    
    def deg2pix(self, val):
        return int(val / self.ds)
    
    def t2step(self, val):
        return int(val / self.dt)
    
    def render(self):
        az = np.arange(self.deg2pix(self.az)) * self.ds
        el = np.arange(self.deg2pix(self.el)) * self.ds
        ori = np.deg2rad(self.ori)
        a, b = np.sin(ori), np.cos(ori)
        
        time = np.zeros(self.t2step(self.dur))
        start, stop = self.t2step(self.start), self.t2step(self.stop)
        time[start:stop] = self.vel
        time = time.cumsum() * self.dt
        
        g_el, g_az, g_time = np.meshgrid(el, az, time)  # NOTE: el and az are swapped here
        image = np.sin(2 * np.pi / self.lam * (a * g_el + b * g_az - g_time) + self.phase)
        
        return image


class AnnulusMask:

    @staticmethod
    def get_standard_parameters():
        return {
            "azimuth": 120.0,
            "elevation": 90.0,
            "duration": 5.0,
            "center": (60.0, 45.0),
            "radius": 10.0,
            "dt": 0.01,
            "ds": 1.0,
        }
    
    def __init__(self, azimuth, elevation, duration,
                 center, radius,
                 dt, ds):
        self.az = azimuth
        self.el = elevation
        self.dur = duration
        self.c = center
        self.r = radius
        self.dt = dt
        self.ds = ds
        
        self.radius = None
    
    def deg2pix(self, val):
        return int(val / self.ds)
    
    def t2step(self, val):
        return int(val / self.dt)
    
    def build_radius_mask(self):
        az = np.arange(self.deg2pix(self.az)) * self.ds
        el = np.arange(self.deg2pix(self.el)) * self.ds
        time = np.zeros(self.t2step(self.dur))
        
        g_el, g_az, _ = np.meshgrid(el, az, time)
        
        self.radius = np.sqrt((g_az - self.c[0])**2 + (g_el - self.c[1])**2)
    
    def render(self, vals):
        if not self.radius:
            self.build_radius_mask()
        
        mask = np.zeros_like(self.radius)

        mask[self.radius <= self.r] = vals[0]
        mask[self.radius > self.r] = vals[1]

        return mask


class TranslatingImageStimulus:
    """Base class for all translating image stimuli.
    Subclasses implement generate_velocity_profile(n_t)."""

    def __init__(self, **params):
        self.img = params["image"]
        self.dur = params["duration"]
        self.start = params["start"]
        self.stop = params["stop"]
        self.phase = params["spatial_phase"]
        self.wwidth = params["window_width"]
        self.ds = params["ds"]
        self.dt = params["dt"]

    def render(self):
        n_t, n_window = self.t2step(self.dur), self.deg2pix(self.wwidth)

        # Establish shifts:
        self.velocity_profile = self.generate_velocity_profile(n_t)
        shifts = (np.cumsum(self.velocity_profile) * self.dt + self.phase) / self.ds

        # Actual movie making!
        movie = []
        for fidx in range(n_t):
            frame = self.shift_by(self.img, (shifts[fidx], 0))[:, :n_window]
            movie.append(frame)
        movie = np.array(movie).swapaxes(0, 2)

        return movie

    def generate_velocity_profile(self, n_t):
        """Subclasses need to implement this..."""
        return np.zeros(n_t)

    def deg2pix(self, val):
        return int(val / self.ds)
    
    def t2step(self, val):
        return int(val / self.dt)

    @staticmethod
    def shift_by(image, pixels):
        m = np.array([
            [1, 0, pixels[0]],
            [0, 1, pixels[1]],
        ]).astype(np.float32)

        return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]),
                              borderMode=cv2.BORDER_WRAP, flags=cv2.INTER_LINEAR)

    
class FixedVelocityTranslatingImageStimulus(TranslatingImageStimulus):

    @staticmethod
    def get_standard_parameters():
        return {
            "image": None,
            "velocity": 100.0,
            "duration": 5.0,
            "start": 1.0,
            "stop": 4.0,
            "spatial_phase": 0.0,
            "window_width": 90.0,
            "ds": 0.225,
            "dt": 0.01,
        }
    
    def __init__(self, **params):
        super().__init__(**params)
        self.vel = params["velocity"]
        
    def generate_velocity_profile(self, n_t):
        shifts = np.zeros(n_t)
        a, b = self.t2step(self.start), self.t2step(self.stop)
        shifts[a:b] = self.vel
        return shifts

    
class NoisyTranslatingImageStimulus(TranslatingImageStimulus):

    @staticmethod
    def get_standard_parameters():
        return {
            "image": None,
            "velocity_sd": 100.0,
            "velocity_tau": 0.2,
            "velocity_bias": 0.0,
            "velocity_order": 1,
            "duration": 5.0,
            "start": 1.0,
            "stop": 4.0,
            "spatial_phase": 0.0,
            "window_width": 90.0,
            "ds": 0.225,
            "dt": 0.01,
        }
    
    def __init__(self, **params):
        super().__init__(**params)
        self.vel_sd = params["velocity_sd"]
        self.vel_tau = params["velocity_tau"]
        self.vel_bias = params["velocity_bias"]
        self.vel_order = params["velocity_order"]
        
    def generate_velocity_profile(self, n_t):
        a, b = self.t2step(self.start), self.t2step(self.stop)

        noise = np.random.normal(size=n_t)
        noise[:a] = 0.0

        for _ in range(self.vel_order):
            noise = lowpass1d(noise, self.vel_tau, self.dt)

        noise[b:] = 0.0
        noise = noise / noise[a:b].std()
        
        shifts = self.vel_sd * noise + self.vel_bias
        
        return shifts


class NoiseStimulus:

    @staticmethod
    def get_standard_parameters():
        return {
            "azimuth": 120.0,
            "elevation": 90.0,
            "pixel_size": 5.0,
            "duration": 9.0,
            "velocity": 100.0,
            "start": 1.5,
            "stop": 7.5,
            "background_width": 120.0,
            "foreground_width": 20.0,
            "shield_width": 0.0,
            "speed_factor": 1.0,
            "flatten_factor": 5.0,
            "dt": 0.01,
            "ds": 0.5,
        }
    
    def __init__(self, azimuth, elevation, pixel_size, duration,
                 velocity, start, stop,
                 background_width, foreground_width, shield_width,
                 speed_factor, flatten_factor,
                 dt, ds):
        
        self.az = azimuth
        self.el = elevation
        self.psize = pixel_size
        self.dur = duration
        self.vel = velocity
        self.start = start
        self.stop = stop
        self.bgw = background_width
        self.fgw = foreground_width
        self.sgw = shield_width
        self.sfac = speed_factor
        self.ffac = flatten_factor
        self.dt = dt
        self.ds = ds
        
        self.output_background = None
        self.output_foreground = None
        self.output_layered = None
        
        # Pre-calculate pixel shape:
        assert (self.psize % self.ds) == 0.0
        assert (self.az % self.psize) == 0.0 and (self.el % self.psize) == 0.0
        self.img_shape = (int(self.az / self.psize), int(self.el / self.psize))
    
    def render_background(self):
        target_shape = (self.img_shape[0], self.img_shape[1], 1)
        freqs = np.random.normal(loc=0.0, scale=self.sfac, size=target_shape)
        phases = np.random.uniform(low=0.0, high=2*np.pi, size=target_shape)

        n_steps = self.t2step(self.dur)
        t = (np.arange(n_steps) * self.dt)[np.newaxis, np.newaxis, :]
        t = t.repeat(self.img_shape[0], axis=0).repeat(self.img_shape[1], axis=1)
        out = np.sin(2 * np.pi * freqs * t + phases)

        out = np.sqrt((1 + self.ffac**2) / (1 + self.ffac**2 * out**2)) * out
        rep = self.deg2pix(self.psize)
        out = out.repeat(rep, axis=0).repeat(rep, axis=1)
        
        self.output_background = out
        return self.output_background
    
    def render_foreground(self):
        pattern = np.sin(2 * np.pi * np.random.rand(*self.img_shape))
        pattern = np.sqrt((1 + self.ffac**2) / (1 + self.ffac**2 * pattern**2)) * pattern
        
        rep = self.deg2pix(self.psize)
        pattern = pattern.repeat(rep, axis=0).repeat(rep, axis=1)

        n_steps = self.t2step(self.dur)
        movie = np.zeros((pattern.shape[0], pattern.shape[1], n_steps))
        
        shifts = np.zeros(n_steps)
        start, stop = self.t2step(self.start), self.t2step(self.stop)
        shifts[start:stop] = self.vel
        shifts = shifts.cumsum() * self.dt
        
        for fidx in range(movie.shape[-1]):
            movie[:, :, fidx] = self.shift_by(pattern, (0, shifts[fidx]))
            
        self.output_foreground = movie
        return self.output_foreground
    
    def render_stack(self, background_contrast, foreground_contrast):
        assert self.output_background is not None and self.output_foreground is not None
        final = np.zeros_like(self.output_background)
        
        mid = self.az / 2.0
        
        # Calculate envelopes:        
        if isinstance(background_contrast, tuple):
            background_contrast = self.construct_envelope(*background_contrast)
            
        if isinstance(foreground_contrast, tuple):
            foreground_contrast = self.construct_envelope(*foreground_contrast)
        
        # Restrict BG:
        hw = self.bgw / 2.0
        left, right = self.deg2pix(mid - hw), self.deg2pix(mid + hw)
        final[left:right, :, :] = background_contrast * self.output_background[left:right, :, :]
        
        # Restrict with frontal field:
        hw = self.sgw / 2.0
        left, right = self.deg2pix(mid - hw), self.deg2pix(mid + hw)
        final[left:right, :, :] = 0.0
        
        # Add windowed FG:
        hw = self.fgw / 2.0
        left, right = self.deg2pix(mid - hw), self.deg2pix(mid + hw)
        final[left:right, :, :] = foreground_contrast * self.output_foreground[left:right, :, :]
        
        self.output_layered = final
        return self.output_layered
    
    def deg2pix(self, val):
        return int(val / self.ds)
    
    def t2step(self, val):
        return int(val / self.dt)
    
    def construct_envelope(self, f, c):
        start, stop = self.t2step(self.start), self.t2step(self.stop)
        time = np.zeros(self.output_background.shape[-1])
        time[start:stop] = self.dt
        time = time.cumsum()
        envelope = c * np.sin(2 * np.pi * f * time)
        return envelope
    
    @staticmethod
    def shift_by(image, pixels):
        m = np.array([
            [1, 0, pixels[0]],
            [0, 1, pixels[1]],
        ]).astype(np.float32)

        return cv2.warpAffine(image, m, (image.shape[1], image.shape[0]),
                              borderMode=cv2.BORDER_WRAP, flags=cv2.INTER_LINEAR)
