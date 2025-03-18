import numpy as np

def create_sliding_windows(data, window_size, forecast_horizon=1):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+forecast_horizon-1])
    return np.array(X), np.array(y)
