import tensorflow as tf
from typing import Tuple


class CTRNN(tf.keras.layers.AbstractRNNCell):
    """
    Continuous-Time Recurrent Neural Network (CTRNN) cell implementation.
    """

    def __init__(
        self,
        num_units: int,
        cell_clip: int = -1,
        global_feedback: bool = False,
        fix_tau: bool = True,
        **kwargs,
    ):
        super(CTRNN, self).__init__(**kwargs)
        self._num_units = num_units
        self._unfolds = 6
        self._delta_t = 0.1
        self.global_feedback = global_feedback
        self.fix_tau = fix_tau
        self.tau = 1.0
        self.cell_clip = cell_clip

    @property
    def state_size(self) -> int:
        return self._num_units

    @property
    def output_size(self) -> int:
        return self._num_units

    def build(self, input_shape: Tuple[int, ...]) -> None:
        input_dim = input_shape[-1]
        self.W_step = self.add_weight(
            name="W_step",
            shape=(input_dim, self._num_units),
            initializer="glorot_uniform",
        )
        self.b_step = self.add_weight(
            name="b_step", shape=(self._num_units,), initializer="zeros"
        )
        super(CTRNN, self).build(input_shape)

    def _dense(self, inputs: tf.Tensor, activation=tf.nn.tanh) -> tf.Tensor:
        y = tf.matmul(inputs, self.W_step) + self.b_step
        if activation is not None:
            y = activation(y)
        return y

    def call(
        self, inputs: tf.Tensor, states: Tuple[tf.Tensor]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
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
    """
    Ordinary Differential Equation (ODE) block for neural networks.
    """

    def __init__(self, hidden_dim: int, num_steps: int = 10, dt: float = 0.1, **kwargs):
        super(ODEBlock, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dt = dt
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(hidden_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        h = inputs
        for _ in range(self.num_steps):
            dh = self.dense2(tf.nn.relu(self.dense1(h)))
            h = h + self.dt * dh
        return h
