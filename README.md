# NEOSENS: Noise Evaluation of Neural Network Output Sensitivity

## Overview
Welcome to NEOSENS, a project dedicated to evaluating the sensitivity of various machine learning and deep learning algorithms, including Liquid Neural Networks (LNN), to subtle input perturbations. Our primary goal is to assess the robustness of these paradigms in real-world applications, such as surgical robotics and self-driving cars, where small input noise should not compromise attention mechanisms. Inspired by the resilience of biological systems, like the human brain, NEOSENS investigates how AI can maintain stability and performance under noisy conditions.

## Project Objectives

- **Comparative Analysis**: Conduct an extensive comparative analysis of AI architectures, including Long Short-Term Memory (LSTM), Liquid Neural Networks (LNN), and Simple Neural Networks, in practical contexts.
- **Sensitivity Evaluation**: Investigate how these architectures respond to subtle input perturbations, similar to challenges encountered in self-driving cars and surgical robotics.
- **Attention Mechanisms**: Assess the efficacy of attention mechanisms in preserving focus amidst noise, inspired by the intricate neurological attention systems.
- **Application-Oriented Insights**: Generate practical insights into which architecture demonstrates superior noise tolerance without compromising attention mechanisms, making them ideal for safety-critical applications.

## Repository Structure

- **README.md**: Project overview and instructions.
- **requirements.txt**: List of required Python packages.
- **setup.py**: Setup script for packaging the project.
- **src/**: Contains the source code.
  - **data/**: Functions for dataset generation (Lorenz, Rossler, Mackey‑Glass, Sine).
  - **models/**: Model definitions (TensorFlow/Keras and scikit‑learn).
  - **utils/**: Utility functions for noise injection and sliding window creation.
  - **experiments/**: Script to run experiments and compare model performance.

## How to Run

1. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the experiments:
    ```bash
    python -m src.experiments.run_experiments
    ```

## License

This project is licensed under the DBaJ-NC-CFL License - see the [LICENSE](./LICENCE.md) file for details.
