import numpy as np
from typing import Union


def add_complex_noise(data: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """
    Add complex noise (Gaussian, Laplacian, and AR) to the data.

    Args:
        data (np.ndarray): Input data.
        noise_level (float): Standard deviation of the noise.

    Returns:
        np.ndarray: Noisy data.
    """
    if noise_level < 0:
        raise ValueError("Noise level must be non-negative.")
    noise_gaussian = np.random.normal(0, noise_level, data.shape)
    noise_laplacian = np.random.laplace(0, noise_level, data.shape)
    alpha = 0.8
    noise_ar = np.zeros(data.shape)
    epsilon = np.random.normal(0, noise_level, data.shape)
    noise_ar[0] = epsilon[0]
    for t in range(1, data.shape[0]):
        noise_ar[t] = alpha * noise_ar[t - 1] + epsilon[t]
    complex_noise = (noise_gaussian + noise_laplacian + noise_ar) / 3.0
    return data + complex_noise
