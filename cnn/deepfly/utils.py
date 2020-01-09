import cv2
import numpy as np
import torch
from numba import jit


# Filters

@jit
def lowpass(i, tau, dt):
    out = np.zeros_like(i)

    alpha = dt / (dt + tau)
    out[:, :, 0] = i[:, :, 0]

    for k in range(i.shape[0]):
        for l in range(i.shape[1]):
            for t in range(1, i.shape[2]):
                out[k, l, t] = alpha * i[k, l, t] + (1 - alpha) * out[k, l, t - 1]

    return out


def highpass(i, tau, dt):
    return i - lowpass(i, tau, dt)


def lowpass1d(i, tau, dt):
    return lowpass(i[np.newaxis, np.newaxis, :], tau, dt).squeeze()


# Torch

def torchify_pattern(x):
    x = torch.FloatTensor(x)
    x = x.unsqueeze(0).unsqueeze(0)
    return x


# Movies

def render_movie(movie, filename, dt, f=30):
    # Simple logic to check for 0-1-scaled movies:
    if movie.max() <= 1.0:
        movie = 255.0 * movie

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 1.0 / dt, (movie.shape[0] * f, movie.shape[1] * f))

    movie = movie.astype(np.uint8)
    for fidx in range(movie.shape[2]):
        frame = movie[..., fidx].T
        frame = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
