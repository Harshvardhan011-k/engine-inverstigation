import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, ops


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = ops.mean(
                ops.sum(ops.square(data - reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * ops.mean(
                ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=-1)
            )
            total_loss = reconstruction_loss + 0.01 * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        _, _, z = self.encoder(inputs)
        return self.decoder(z)


class AnomalyVAE:
    """Variational Autoencoder wrapper for time-series anomaly detection."""
    def __init__(self, seq_len: int, num_features: int, latent_dim: int = 8):
        self.seq_len = seq_len
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = 0.05
        self._build_vae()

    def _build_vae(self):
        encoder_inputs = keras.Input(shape=(self.seq_len, self.num_features))
        x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(encoder_inputs)
        x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
        x = layers.Conv1D(filters=16, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)

        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = keras.Input(shape=(self.latent_dim,))
        half_len = self.seq_len // 2
        x = layers.Dense(half_len * 16, activation="relu")(latent_inputs)
        x = layers.Reshape((half_len, 16))(x)
        x = layers.UpSampling1D(size=2)(x)
        if half_len * 2 != self.seq_len:
            x = layers.ZeroPadding1D((0, self.seq_len - half_len * 2))(x)
        x = layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
        decoder_outputs = layers.Conv1D(filters=self.num_features, kernel_size=3, padding="same", activation="sigmoid")(x)

        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

        self.model = VAEModel(self.encoder, self.decoder)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    def fit(self, X_healthy: np.ndarray, epochs: int = 15, batch_size: int = 64):
        self.model.fit(
            X_healthy,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1
        )
        healthy_scores = self.predict_anomaly_score(X_healthy)
        self.threshold = float(np.mean(healthy_scores) + 3.0 * np.std(healthy_scores))
        return self.threshold

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        reconstructed = self.model.predict(X, batch_size=128, verbose=0)
        mse = np.mean(np.square(X - reconstructed), axis=(1, 2))
        return mse

    def save(self, filepath_prefix: str):
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        self.model.save_weights(filepath_prefix + "_weights.weights.h5")
        np.save(filepath_prefix + "_threshold.npy", np.array([self.threshold]))

    def load(self, filepath_prefix: str):
        dummy = np.zeros((1, self.seq_len, self.num_features), dtype=np.float32)
        _ = self.model(dummy)
        self.model.load_weights(filepath_prefix + "_weights.weights.h5")
        if os.path.exists(filepath_prefix + "_threshold.npy"):
            self.threshold = float(np.load(filepath_prefix + "_threshold.npy")[0])
