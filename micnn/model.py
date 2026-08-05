"""MI-CNN: a Mamba-inspired convolutional network for multivariate time-series
classification.

MI-CNN keeps the split-gate-project block layout of Mamba (Gu & Dao, 2024) but
replaces the selective state-space recurrence with a depthwise *causal*
convolution and a learned gate, in the spirit of Gated Convolutional Networks
(Dauphin et al., 2017). The result is a lightweight, fully parallel TensorFlow
model: no recurrent scan, no custom CUDA kernel, and no `mamba-ssm` dependency.

Shape flow for the postural-state task (T = 100 time steps, F = 1001 features):

    (B, 100, 1001)
      -> Dense(d_model=64) + LayerNorm          # input projection
      -> 4 x MambaInspiredBlock(64)             # sequence encoder
      -> GlobalAveragePooling1D                 # (B, 64)
      -> Dense(128) -> Dropout -> Dense(64) -> Dropout
      -> Dense(2, softmax)

With F = 1001 this instantiates 256,066 trainable parameters: about 1.0 MB of
float32 weights, or the ~3.07 MB reported in the paper once saved as a trained
`.keras` checkpoint that also stores the Adam optimizer state.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
)
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable(package="micnn")
class MambaInspiredBlock(Layer):
    """One MI-CNN block: split -> gated causal convolution -> project -> residual.

    The input of width ``d_model`` is projected to ``2 * d_inner`` and split into
    a *temporal* branch and a *gate* branch:

    * The temporal branch is passed through SiLU, a depthwise causal Conv1D
      (kernel 3, one filter group per channel, so channels do not mix), and SiLU
      again. Causal padding means position *t* only ever sees positions <= *t*,
      which keeps the block usable for streaming/online inference.
    * A learned per-channel gate ``sigmoid(softplus(dt_proj(x)))`` modulates the
      convolved features. This is the "state-space transform" of Figure 2 in the
      paper: a simplified selective-gating operation, not an explicit SSM
      recurrence.
    * The gated features are multiplied by ``SiLU`` of the second branch, giving
      the gated-linear-unit style multiplicative interaction.

    The product is projected back to ``d_model``, dropped out, added to the
    residual, and layer-normalised.

    Args:
        d_model: Width of the block input/output (the residual stream).
        d_state: Width of the auxiliary state projection. See the note on
            ``x_proj`` in ``build``.
        expand_factor: Inner width multiplier; ``d_inner = expand_factor * d_model``.
        dropout: Dropout rate applied to the block output before the residual add.
    """

    def __init__(self, d_model, d_state=16, expand_factor=2, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = int(expand_factor * d_model)
        self.dropout_rate = dropout

    def build(self, input_shape):
        # Project to 2 * d_inner so the result can be split into the temporal
        # branch and the gating branch.
        self.in_proj = Dense(self.d_inner * 2, use_bias=False, name="in_proj")

        # Depthwise causal convolution: groups == filters means each channel is
        # convolved independently, so this captures short-range temporal shape
        # per channel at a fraction of a dense Conv1D's cost.
        self.conv1d = Conv1D(
            filters=self.d_inner,
            kernel_size=3,
            padding="causal",
            groups=self.d_inner,
            use_bias=True,
            name="conv1d",
        )

        # State projection, retained from the Mamba-style parameterisation. It is
        # instantiated (and therefore counted in the model's 256,066 parameters)
        # but its output does not feed the block output: the selective behaviour
        # is carried entirely by the `dt_proj` gate below. It is kept here so this
        # implementation stays weight-compatible with the published checkpoints
        # and reproduces the parameter count reported in the paper.
        self.x_proj = Dense(self.d_state, use_bias=False, name="x_proj")

        # Gate parameterisation ("delta" in Mamba's notation).
        self.dt_proj = Dense(self.d_inner, use_bias=True, name="dt_proj")

        self.out_proj = Dense(self.d_model, use_bias=False, name="out_proj")
        self.norm = LayerNormalization(epsilon=1e-5, name="norm")
        self.dropout = Dropout(self.dropout_rate, name="dropout")
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Map (batch, seq_len, d_model) -> (batch, seq_len, d_model)."""
        residual = inputs

        # Split the projection into the temporal branch (x) and gate branch (res).
        x, res = tf.split(self.in_proj(inputs), 2, axis=-1)

        x = tf.nn.silu(x)
        x = self.conv1d(x)               # local, causal temporal mixing
        x = tf.nn.silu(x)

        _ = self.x_proj(x)               # see the note in `build`

        # Selective gate: softplus keeps the pre-activation positive, sigmoid
        # squashes it to (0, 1) so it acts as a soft per-channel, per-timestep
        # retain/forget weight over the convolved features.
        dt = tf.nn.softplus(self.dt_proj(x))
        y = x * tf.nn.sigmoid(dt)

        # Gated-linear-unit interaction with the untouched second branch.
        y = y * tf.nn.silu(res)

        out = self.out_proj(y)
        out = self.dropout(out, training=training)
        return self.norm(out + residual)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_state": self.d_state,
                "expand_factor": self.expand_factor,
                "dropout": self.dropout_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_micnn(
    input_shape,
    num_classes=2,
    d_model=64,
    n_layers=4,
    d_state=16,
    expand_factor=2,
    dropout=0.3,
    name="MI-CNN",
):
    """Build the MI-CNN classifier.

    Args:
        input_shape: ``(timesteps, features)`` of one window, e.g. ``(100, 1001)``.
        num_classes: Number of output classes (2 for balanced vs. imbalanced).
        d_model: Width of the residual stream.
        n_layers: Number of stacked ``MambaInspiredBlock`` layers.
        d_state: State-projection width inside each block.
        expand_factor: Inner-width multiplier inside each block.
        dropout: Dropout rate, shared by the blocks and the classification head.
        name: Keras model name.

    Returns:
        An uncompiled ``keras.Model`` producing softmax class probabilities.
    """
    inputs = Input(shape=input_shape, name="window")

    # Project the wide multimodal feature vector down to d_model before any
    # temporal modelling; LayerNorm stabilises the scale across the 1001
    # heterogeneous channels (kinematic, EMG, EDA).
    x = Dense(d_model, name="input_proj")(inputs)
    x = LayerNormalization(name="input_norm")(x)

    for i in range(n_layers):
        x = MambaInspiredBlock(
            d_model=d_model,
            d_state=d_state,
            expand_factor=expand_factor,
            dropout=dropout,
            name=f"mamba_block_{i}",
        )(x)

    # Average over time: the class evidence is distributed across the window
    # rather than concentrated at its final step, so pooling beats last-step readout.
    x = GlobalAveragePooling1D(name="gap")(x)

    x = Dense(128, activation="relu", name="head_dense_1")(x)
    x = Dropout(dropout, name="head_dropout_1")(x)
    x = Dense(64, activation="relu", name="head_dense_2")(x)
    x = Dropout(dropout, name="head_dropout_2")(x)

    outputs = Dense(num_classes, activation="softmax", name="predictions")(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


def compile_micnn(model, learning_rate=1e-3):
    """Compile with the paper's training objective (Adam + sparse CE)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
