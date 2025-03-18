import tensorflow as tf

class CTRNN(tf.keras.layers.AbstractRNNCell):
    def __init__(self, num_units, cell_clip=-1, global_feedback=False, fix_tau=True, **kwargs):
        super(CTRNN, self).__init__(**kwargs)
        self._num_units = num_units
        self._unfolds = 6
        self._delta_t = 0.1
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self.tau = 1.0
        self.cell_clip = cell_clip

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W_step = self.add_weight(
            name='W_step',
            shape=(input_dim, self._num_units),
            initializer='glorot_uniform')
        self.b_step = self.add_weight(
            name='b_step',
            shape=(self._num_units,),
            initializer='zeros')
        super(CTRNN, self).build(input_shape)

    def _dense(self, inputs, activation=tf.nn.tanh):
        y = tf.matmul(inputs, self.W_step) + self.b_step
        if activation is not None:
            y = activation(y)
        return y

    def call(self, inputs, states):
        state = states[0]
        if not self.global_feedback:
            input_f_prime = self._dense(inputs)
        for i in range(self._unfolds):
            if self.global_feedback:
                fused_input = tf.concat([inputs, state], axis=-1)
                input_f_prime = self._dense(fused_input)
            f_prime = -state / self.tau + input_f_prime
            state = state + self._delta_t * f_prime
            if self.cell_clip > 0:
                state = tf.clip_by_value(state, -self.cell_clip, self.cell_clip)
        return state, [state]

class ODEBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, num_steps=10, dt=0.1, **kwargs):
        super(ODEBlock, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dt = dt
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim)

    def call(self, inputs):
        h = inputs
        for _ in range(self.num_steps):
            dh = self.dense2(tf.nn.relu(self.dense1(h)))
            h = h + self.dt * dh
        return h


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