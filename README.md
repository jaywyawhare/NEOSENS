# NEOSENS: Noise Evaluation of Quantum and Neural Network Output Sensitivity

## Overview
Welcome to NEOSENS, a research project dedicated to evaluating the sensitivity of various machine learning and deep learning algorithms, Quantum Long Short-Term Memory (QLSTM), and Liquid Neural Networks (LNN) to subtle input perturbations. Our primary goal is to assess the robustness of these paradigms in real-world applications, including surgical robotics, self-driving cars where small input noise should not compromise attention mechanisms, taking inspiration from the resilience of biological systems, like the human brain.

## Project Objectives

- **Comparative Analysis**: Conduct an extensive comparative analysis of AI architectures, including Long Short-Term Memory (LSTM), Quantum Long Short-Term Memory (QLSTM), Liquid Neural Networks (LNN), and Simple Neural Networks, in practical contexts.
- **Sensitivity Evaluation**: Investigate how these architectures respond to subtle input perturbations, resembling challenges encountered in self-driving cars and surgical robotics.
- **Attention Mechanisms**: Assess the efficacy of attention mechanisms in preserving focus and attention amidst noise, inspired by the intricate neurological attention systems.
- **Application-Oriented Insights**: Generate practical insights into which architecture demonstrates superior noise tolerance without compromising attention mechanisms, making them ideal for safety-critical applications.


## Installation

1. Set up a virtual environment with conda 

    ```bash
    conda create -n neosens python=3.7
    conda activate neosens
    ```
1. Clone the repository

    ```bash
    git clone https://github.com/jaywyawhare/NEOSENS.git
    ```
1. Install the requirements

    ```bash
    pip install -r requirements.txt
    ```

1. Explore the codebase in the code/ directory to access data preprocessing, model implementations, and evaluation scripts.

1. To run experiments and evaluate AI architectures, refer to specific scripts and documentation within the code/ directory.

## License

This project is licensed under the DBaJ-NC-CFL License - see the [LICENSE](./LICENCE.md) file for details.