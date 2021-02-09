import tensorflow as tf
import numpy as np
import math

class SincLayer(tf.keras.layers.Layer):
    min_mel_freq = 80
    min_hz_freq = 50
    min_hz_bandwidth = 50

    @staticmethod
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    @staticmethod
    def mel_to_hz(mel):
        return (10 ** (mel / 2595) - 1) * 700

    def __init__(self, n_filter, filter_dim, sample_rate, **kwargs):
        self.n_filter = n_filter
        self.filter_dim = filter_dim
        self.sample_rate = sample_rate
        assert self.filter_dim % 2 == 1  # Make sure it can be symetric
        super(SincLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Initialize Weights
        self.filter_low_freq = self.add_weight(
            name='filter_low_freq',
            shape=(self.n_filter,),
            initializer='uniform',
            constraint=tf.keras.constraints.NonNeg(),
            dtype=tf.float32,
            trainable=True
        )
        self.filter_bandwidth = self.add_weight(
            name='filter_bandwidth',
            shape=(self.n_filter,),
            initializer='uniform',
            constraint=tf.keras.constraints.NonNeg(),
            dtype=tf.float32,
            trainable=True
        )

        max_mel_freq = self.hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(self.min_mel_freq, max_mel_freq, self.n_filter)
        hz_points = self.mel_to_hz(mel_points)
        left_band = np.roll(hz_points, 1)
        right_band = np.roll(hz_points, -1)
        left_band[0] = 30
        right_band[-1] = (self.sample_rate / 2) - 100 # Why -100? Save space for safe padding
        self.set_weights([left_band / self.sample_rate, (right_band - left_band) / self.sample_rate])

        
    
    def call(self, x):
        # Add Safe Padding
        self.filter_low_cutoff = self.filter_low_freq + self.min_hz_freq / self.sample_rate
        self.filter_high_cutoff = self.filter_low_cutoff + self.filter_bandwidth + self.min_hz_bandwidth / self.sample_rate
        self.filter_low_cutoff = tf.reshape(self.filter_low_cutoff, [-1, 1])
        self.filter_high_cutoff = tf.reshape(self.filter_high_cutoff, [-1, 1])

        # Hamming Window
        n = np.linspace(0, self.filter_dim, self.filter_dim)
        window = 0.54 - 0.46 + np.cos(2 * math.pi * n / self.filter_dim)
        self.window = tf.constant(window.reshape([1, -1]), dtype=tf.float32)

        # Construct Kernel
        n = np.linspace(1, (self.filter_dim - 1) / 2, int((self.filter_dim - 1) / 2)).reshape((1, -1))  # Half of filter
        n = tf.constant(n / self.sample_rate, dtype=tf.float32) # Why divided by sample_rate?

        inter_low = 2 * math.pi * tf.linalg.matmul(self.filter_low_cutoff * self.sample_rate, n)
        low_low_pass = 2 * tf.multiply(tf.math.sin(inter_low) / inter_low, self.filter_low_cutoff)
        inter_high = 2 * math.pi * tf.linalg.matmul(self.filter_high_cutoff * self.sample_rate, n)
        high_low_pass = 2 * tf.multiply(tf.math.sin(inter_high) / inter_high, self.filter_high_cutoff)
        band_pass = high_low_pass - low_low_pass

        self.band_pass = tf.concat([tf.reverse(band_pass, axis=[1]), tf.zeros([self.n_filter, 1], tf.float32), band_pass], 1)
        self.band_pass = tf.multiply(self.band_pass, self.window)  # Shape: [n_filter, filter_dim]

        self.kernel = tf.reshape(tf.transpose(self.band_pass), [self.filter_dim, 1, self.n_filter]) # [filter_dim, 1, n_filter]

        return tf.nn.conv1d(
            x,
            filters=self.kernel,
            stride=1,
            padding='VALID')
    
    def compute_output_shape(self, input_shape):
        # 1D Conv with (pad=VALID, stride=1)
        output_shape = (input_shape[0], input_shape[1] - self.filter_dim + 1, self.n_filter)
        return output_shape
