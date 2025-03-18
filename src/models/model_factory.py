import tensorflow as tf
from .tf_models import CTRNN, ODEBlock

def create_simple_nn(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

def create_lstm(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

def create_gru(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.GRU(128, input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

def create_cnn(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

def create_lnn(input_shape):
    cell = CTRNN(num_units=64, cell_clip=-1, global_feedback=False, fix_tau=True)
    model = tf.keras.Sequential([
        tf.keras.layers.RNN(cell, input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    return model

def create_transformer(input_shape, num_heads=4, ff_dim=64):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x)
    x = tf.keras.layers.Add()([x, attn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation='relu'),
        tf.keras.layers.Dense(64),
    ])
    ffn_output = ffn(x)
    x = tf.keras.layers.Add()([x, ffn_output])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_bidirectional_lstm(input_shape):
    return tf.keras.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128), input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)
    ])

def create_tcn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, dilation_rate=1, padding='causal', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, dilation_rate=2, padding='causal', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, dilation_rate=4, padding='causal', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_neural_ode(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.expand_dims(x, axis=1)
    x = ODEBlock(128)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    