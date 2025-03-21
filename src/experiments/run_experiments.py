import numpy as np
import matplotlib.pyplot as plt
from src.utils.sliding_window import create_sliding_windows
from src.utils.noise import add_complex_noise
from src.models import tf_models, sklearn_models, model_factory
from sklearn.metrics import mean_squared_error
import torch
import inspect
from typing import Callable, Dict, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_and_evaluate_tf(
    model_fn: Callable,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_func: Callable,
    noise_level: float,
    epochs: int,
) -> Tuple[float, float, float]:
    """
    Train and evaluate a TensorFlow model.

    Args:
        model_fn (Callable): Function to create the model.
        model_name (str): Name of the model.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test data.
        y_test (np.ndarray): Test labels.
        noise_func (Callable): Function to add noise.
        noise_level (float): Noise level.
        epochs (int): Number of training epochs.

    Returns:
        Tuple[float, float, float]: Clean loss, noisy loss, and delta.
    """
    print(f"\nTraining {model_name} ...")
    model = model_fn(X_train.shape[1:])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=0)
    loss_clean = model.evaluate(X_test, y_test, verbose=0)
    X_test_noisy = noise_func(X_test, noise_level)
    loss_noisy = model.evaluate(X_test_noisy, y_test, verbose=0)
    delta = loss_noisy - loss_clean
    print(f"{model_name} - Clean Test Loss (MSE): {loss_clean:.4f}")
    print(f"{model_name} - Noisy Test Loss (MSE): {loss_noisy:.4f}")
    print(f"{model_name} - Delta: {delta:.4f}")
    return loss_clean, loss_noisy, delta


def train_and_evaluate_sklearn(
    model: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_func: Callable,
    noise_level: float,
) -> Tuple[float, float, float]:
    """
    Train and evaluate a scikit-learn model.

    Args:
        model (Callable): The scikit-learn model.
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Test data.
        y_test (np.ndarray): Test labels.
        noise_func (Callable): Function to add noise.
        noise_level (float): Noise level.

    Returns:
        Tuple[float, float, float]: Clean loss, noisy loss, and delta.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flat = X_train.reshape((n_train, -1))
    X_test_flat = X_test.reshape((n_test, -1))
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)
    mse_clean = mean_squared_error(y_test, y_pred)
    X_test_noisy = noise_func(X_test, noise_level)
    X_test_noisy_flat = X_test_noisy.reshape((n_test, -1))
    y_pred_noisy = model.predict(X_test_noisy_flat)
    mse_noisy = mean_squared_error(y_test, y_pred_noisy)
    delta = mse_noisy - mse_clean
    print(f"SK Model - Clean Test Loss (MSE): {mse_clean:.4f}")
    print(f"SK Model - Noisy Test Loss (MSE): {mse_noisy:.4f}")
    print(f"SK Model - Delta: {delta:.4f}")
    return mse_clean, mse_noisy, delta


def run_experiment(
    dataset_name: str,
    data_func: Callable,
    window_size: int,
    forecast_horizon: int,
    noise_func: Callable,
    noise_level: float,
    epochs: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run an experiment with the given dataset and models.

    Args:
        dataset_name (str): Name of the dataset.
        data_func (Callable): Function to load the data.
        window_size (int): Size of the sliding window.
        forecast_horizon (int): Forecast horizon.
        noise_func (Callable): Function to add noise.
        noise_level (float): Noise level.
        epochs (int): Number of training epochs.

    Returns:
        Dict[str, Dict[str, Dict[str, float]]]: Results of the experiment.
    """
    print(f"\n==== Dataset: {dataset_name} ====")
    data = data_func()
    X, y = create_sliding_windows(data, window_size, forecast_horizon)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-6)
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y = (y - y_mean) / (y_std + 1e-6)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    results = {"TF": {}, "SK": {}}

    tf_models = model_factory.create_all_models(X_train.shape[1:])
    for name, model in tf_models.items():
        try:
            print(f"\nEvaluating TF model: {name}")
            mse_clean, mse_noisy, delta = train_and_evaluate_tf(
                lambda _: model, 
                name,
                X_train,
                y_train,
                X_test,
                y_test,
                add_complex_noise,
                noise_level,
                epochs,
            )
            results["TF"][name] = {
                "Clean": mse_clean,
                "Noisy": mse_noisy,
                "Delta": delta,
            }
        except Exception as e:
            print(f"Error evaluating TF model {name}: {e}")

    sk_models = sklearn_models.create_sklearn_models(y.shape[1])
    for name, model in sk_models.items():
        print(f"\nEvaluating SK model: {name}")
        mse_clean, mse_noisy, delta = train_and_evaluate_sklearn(
            model, X_train, y_train, X_test, y_test, add_complex_noise, noise_level
        )
        results["SK"][name] = {"Clean": mse_clean, "Noisy": mse_noisy, "Delta": delta}

    return results
