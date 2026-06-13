
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape,
                output_dim,
                learning_rate=1e-3,
                lstm_units=64,
                dense_units=32,
                dropout=0.3,
                num_layers=1,
                loss="mse",
                optimizer="adam",
                use_attention=False,
                mc_dropout=False):
    """
    Build a BiLSTM regression model with optional attention and MC dropout.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # BiLSTM stack
    for i in range(num_layers):
        return_seq = (i < num_layers - 1 or use_attention)
        x = layers.Bidirectional(
            layers.LSTM(lstm_units,
                        return_sequences=return_seq,
                        dropout=dropout if mc_dropout else 0.0,
                        recurrent_dropout=0.0)
        )(x)

    # Attention
    if use_attention and len(x.shape) == 3:
        x = layers.Attention()([x, x])

    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim)(x)

    model = models.Model(inputs, outputs)

    # Optimizer
    if optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(optimizer=opt, loss=loss, metrics=["mae", "mse"])
    return model
