import os
import random
import subprocess
import configparser
import tensorflow as tf
import numpy as np
from .model.sincnet import SincLayer

class Preprocess:
    @staticmethod
    def get_template(temp_dir):
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section('PATH')
        cfg['PATH'] = {
            '#data_list: list of <data> <label> pairs': '',
            'data_list': 'train_data.list',
            '#model_dir: output model directory': '',
            'model_dir': 'models/test',
        }
        cfg.add_section('CONFIG')
        cfg['CONFIG'] = {
            '#sample_rate: sampling rate': '',
            'sample_rate': '16000',
            # '#: label for background': '',
            # 'backgronnd_label': 'bg',
        }
        with open(temp_dir, 'w') as f:
            cfg.write(f)
    
    def __init__(self, cfg_dir):
        self._parse_cfg(cfg_dir)

    def _parse_cfg(self, cfg_dir):
        cfg = configparser.ConfigParser()
        cfg.read(cfg_dir)
        # PATH
        self._data_list = cfg['PATH'].get('data_list')
        self._model_dir = cfg['PATH'].get('model_dir')
        # CONFIG
        self._sample_rate = cfg['CONFIG'].getint('sample_rate')

        self._batch_size = 128

    def _init_data(self):
        # Read Data List
        with open(self._data_list, 'r') as f:
            lines = f.readlines()
        data = [x.strip('\n') for x in lines]
        random.shuffle(data)

        train_data = data[:int(len(data) * 0.875)]
        valid_data = data[int(len(data) * 0.875):]

        # Collect Labels
        label_dict = {}
        for i in range(len(data)):
            label = data[i].split(' ')[1]
            label_dict[label] = label_dict.get(label, 0) + 1
        labels = np.array(list(label_dict.keys()))
        self._n_label = labels.shape[0]
        self._n_total_sample = sum([label_dict[key] for key in label_dict])
        inverse_sum = sum([1 / label_dict[key] for key in label_dict])
        self._class_weight = {}
        for i in range(len(labels)):
            self._class_weight[i] = label_dict[labels[i]] / inverse_sum

        # Parse Function
        def __get_waveform_and_label(raw_data):
            content = tf.strings.split(raw_data, ' ')
            waveform, _ = tf.audio.decode_wav(tf.io.read_file(content[0]))
            label = tf.argmax(content[1] == labels)
            return waveform, label
        
        train_ds = tf.data.Dataset.from_tensor_slices(train_data)
        train_ds = train_ds.shuffle(buffer_size=50000)
        train_ds = train_ds.map(__get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.batch(self._batch_size)

        valid_ds = tf.data.Dataset.from_tensor_slices(valid_data)
        valid_ds = valid_ds.map(__get_waveform_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        valid_ds = valid_ds.batch(self._batch_size)
        return train_ds, valid_ds
 
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[self._sample_rate, 1]),
            SincLayer(16, 251, 16000),
            tf.keras.layers.MaxPooling1D(pool_size=3, strides=3),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(16, 5, 1, padding='valid'),
            tf.keras.layers.MaxPooling1D(pool_size=3, strides=3),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(16, 5, 1, padding='valid'),
            tf.keras.layers.MaxPooling1D(pool_size=3, strides=3),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(16, 5, 1, padding='valid'),
            tf.keras.layers.MaxPooling1D(pool_size=3, strides=3),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(16, 5, 1, padding='valid'),
            tf.keras.layers.MaxPooling1D(pool_size=3, strides=3),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv1D(16, 5, 1, padding='valid'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(self._n_label)
        ])
        model.summary()
        return model

    def run(self):
        train_ds, valid_ds = self._init_data()
        model = self._build_model()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        
        print(model.get_layer(index=0).get_weights())
        model.fit(train_ds, epochs=2, class_weight=self._class_weight, validation_data=valid_ds)
        print(model.get_layer(index=0).get_weights())

def process(cfg_dir):
    p = Preprocess(cfg_dir)
    p.run()

def get_template(temp_dir):
    Preprocess.get_template(temp_dir)

