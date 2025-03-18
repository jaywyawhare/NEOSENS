import numpy as np
import matplotlib.pyplot as plt
from src.experiments.run_experiments import run_experiment
from src.data.datasets import (
    generate_lorenz_data,
    generate_rossler_data,
    generate_mackey_glass_data,
    generate_sine_data,
)
from src.utils.noise import add_complex_noise
from src.models import model_factory


def main():
    window_size = 50
    forecast_horizon = 1
    noise_level = 0.1
    epochs = 20

    dataset_funcs = {
        "Lorenz": generate_lorenz_data,
        "Rossler": generate_rossler_data,
        "Mackey-Glass": generate_mackey_glass_data,
        "Sine": generate_sine_data,
    }

    all_results = {}
    for ds_name, ds_func in dataset_funcs.items():
        res = run_experiment(
            ds_name,
            ds_func,
            window_size,
            forecast_horizon,
            add_complex_noise,
            noise_level,
            epochs,
        )
        all_results[ds_name] = res

    for ds_name, res in all_results.items():
        tf_names = list(res["TF"].keys())
        sk_names = list(res["SK"].keys())

        tf_clean = [res["TF"][name]["Clean"] for name in tf_names]
        tf_noisy = [res["TF"][name]["Noisy"] for name in tf_names]
        tf_delta = [res["TF"][name]["Delta"] for name in tf_names]

        sk_clean = [res["SK"][name]["Clean"] for name in sk_names]
        sk_noisy = [res["SK"][name]["Noisy"] for name in sk_names]
        sk_delta = [res["SK"][name]["Delta"] for name in sk_names]

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        x_tf = np.arange(len(tf_names))
        axs[0].bar(x_tf - 0.2, tf_clean, width=0.4, label="Clean")
        axs[0].bar(x_tf + 0.2, tf_noisy, width=0.4, label="Noisy")
        axs[0].set_xticks(x_tf)
        axs[0].set_xticklabels(tf_names, rotation=45, ha="right")
        axs[0].set_ylabel("MSE")
        axs[0].set_title(f"TF Models on {ds_name}")
        axs[0].legend()

        x_sk = np.arange(len(sk_names))
        axs[1].bar(x_sk - 0.2, sk_clean, width=0.4, label="Clean")
        axs[1].bar(x_sk + 0.2, sk_noisy, width=0.4, label="Noisy")
        axs[1].set_xticks(x_sk)
        axs[1].set_xticklabels(sk_names, rotation=45, ha="right")
        axs[1].set_ylabel("MSE")
        axs[1].set_title(f"SK Models on {ds_name}")
        axs[1].legend()

        plt.suptitle(f"Clean and Noisy Losses on {ds_name} Dataset")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].bar(x_tf, tf_delta, width=0.6, label="Δ (Noisy - Clean)")
        axs[0].set_xticks(x_tf)
        axs[0].set_xticklabels(tf_names, rotation=45, ha="right")
        axs[0].set_ylabel("MSE Delta")
        axs[0].set_title(f"TF Models on {ds_name}")
        axs[0].legend()

        axs[1].bar(x_sk, sk_delta, width=0.6, label="Δ (Noisy - Clean)")
        axs[1].set_xticks(x_sk)
        axs[1].set_xticklabels(sk_names, rotation=45, ha="right")
        axs[1].set_ylabel("MSE Delta")
        axs[1].set_title(f"SK Models on {ds_name}")
        axs[1].legend()

        plt.suptitle(f"Delta (Noisy - Clean) on {ds_name} Dataset")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    main()
