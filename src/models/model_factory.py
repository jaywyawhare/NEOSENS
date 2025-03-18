import tensorflow as tf
from .tf_models import CTRNN, ODEBlock
from typing import Tuple, Dict


def create_simple_nn(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create a simple neural network model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: Neural network model.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3),
        ]
    )


def create_lstm(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create an LSTM model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: LSTM model.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(128, input_shape=input_shape),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3),
        ]
    )


def create_gru(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create a GRU model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: GRU model.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.GRU(128, input_shape=input_shape),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3),
        ]
    )


def create_cnn(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create a CNN model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: CNN model.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3),
        ]
    )


def create_lnn(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create a LNN model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: LNN model.
    """
    cell = CTRNN(num_units=128, cell_clip=-1, global_feedback=False, fix_tau=True)
    model = tf.keras.Sequential(
        [tf.keras.layers.RNN(cell, input_shape=input_shape), tf.keras.layers.Dense(1)]
    )
    return model


def create_transformer(
    input_shape: Tuple[int], num_heads: int = 4, ff_dim: int = 64
) -> tf.keras.Model:
    """
    Create a Transformer model.

    Args:
        input_shape (Tuple[int]): Input shape.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward dimension.

    Returns:
        tf.keras.Model: Transformer model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(
        x, x
    )
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    ffn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(64),
        ]
    )
    ffn_output = ffn(x)
    x = tf.keras.layers.Add()([x, ffn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_bidirectional_lstm(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create a bidirectional LSTM model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: Bidirectional LSTM model.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128), input_shape=input_shape
            ),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3),
        ]
    )


def create_tcn(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create a TCN model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: TCN model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=2, dilation_rate=1, padding="causal", activation="relu"
    )(inputs)
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=2, dilation_rate=2, padding="causal", activation="relu"
    )(x)
    x = tf.keras.layers.Conv1D(
        filters=64, kernel_size=2, dilation_rate=4, padding="causal", activation="relu"
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_neural_ode(input_shape: Tuple[int]) -> tf.keras.Model:
    """
    Create a neural ODE model.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        tf.keras.Model: Neural ODE model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.expand_dims(x, axis=1)
    x = ODEBlock(128)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def create_all_models(input_shape: Tuple[int]) -> Dict[str, tf.keras.Model]:
    """
    Create all models from the factory.

    Args:
        input_shape (Tuple[int]): Input shape.

    Returns:
        Dict[str, tf.keras.Model]: Dictionary of model names and their instances.
    """
    return {
        "SimpleNN": create_simple_nn(input_shape),
        "LSTM": create_lstm(input_shape),
        "GRU": create_gru(input_shape),
        "CNN": create_cnn(input_shape),
        "LNN": create_lnn(input_shape),
        "Transformer": create_transformer(input_shape),
        "BidirectionalLSTM": create_bidirectional_lstm(input_shape),
        "TCN": create_tcn(input_shape),
        "NeuralODE": create_neural_ode(input_shape),
    }
