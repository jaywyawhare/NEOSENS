import numpy as np

def add_complex_noise(data, noise_level=0.05):
    noise_gaussian = np.random.normal(0, noise_level, data.shape)
    noise_laplacian = np.random.laplace(0, noise_level, data.shape)
    alpha = 0.8
    noise_ar = np.zeros(data.shape)
    epsilon = np.random.normal(0, noise_level, data.shape)
    noise_ar[0] = epsilon[0]
    for t in range(1, data.shape[0]):
        noise_ar[t] = alpha * noise_ar[t-1] + epsilon[t]
    complex_noise = (noise_gaussian + noise_laplacian + noise_ar) / 3.0
    return data + complex_noise
