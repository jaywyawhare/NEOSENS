import numpy as np
from typing import Tuple


def create_sliding_windows(
    data: np.ndarray, window_size: int, forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series data.

    Args:
        data (np.ndarray): Input time series data.
        window_size (int): Size of the sliding window.
        forecast_horizon (int): Steps ahead to forecast.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of input windows (X) and corresponding targets (y).
    """
    if len(data) < window_size + forecast_horizon:
        raise ValueError(
            "Data length must be greater than window_size + forecast_horizon."
        )
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size + forecast_horizon - 1])
    return np.array(X), np.array(y)
