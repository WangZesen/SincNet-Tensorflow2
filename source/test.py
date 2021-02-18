from datetime import datetime
import tensorflow as tf
import configparser
import numpy as np
import os
from .data.segment import assign_division
from .model.visualize import plot_kernel
from .model.sincnet import SincLayer

class TestTool:
    @staticmethod
    def get_template(temp_dir):
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section('PATH')
        cfg['PATH'] = {
            '#data_list: list of <data> <label> pairs': '',
            'data_list': 'train_data.list',
            '#model_dir: output model directory': '',
            'model_dir': 'models/test',
            '#log_dir: train log directory': '',
            'log_dir': 'logs/test/'
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
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y-%b-%d-%H-%M-%S")
        # PATH
        self._data_list = cfg['PATH'].get('data_list')
        self._model_dir = cfg['PATH'].get('model_dir')
        self._log_dir = cfg['PATH'].get('log_dir')
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_dir = self._log_dir + '/' + timestamp
        # CONFIG
        self._sample_rate = cfg['CONFIG'].getint('sample_rate')
        self._batch_size = 256

    def _init_data(self):
        # Read Data List
        with open(self._data_list, 'r') as f:
            lines = f.readlines()
        data = [x.strip('\n') for x in lines]
        train_data = []
        valid_data = []
        test_data = []
        for item in data:
            par = assign_division(item.split(' ')[0], 70, 10)
            if par == 'train':
                train_data.append(item)
            elif par == 'test':
                test_data.append(item)
            else:
                valid_data.append(item)

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
            self._class_weight[i] = (1 / label_dict[labels[i]]) / inverse_sum * len(labels)

        # Parse Function
        def __get_waveform_and_label(raw_data):
            content = tf.strings.split(raw_data, ' ')
            waveform, _ = tf.audio.decode_wav(tf.io.read_file(content[0]))
            label = tf.argmax(content[1] == labels)
            return waveform, label
        
        AUTOTUNE = tf.data.AUTOTUNE if tf.__version__.startswith('2.4') else tf.data.experimental.AUTOTUNE

        train_ds = tf.data.Dataset.from_tensor_slices(train_data)
        train_ds = train_ds.shuffle(buffer_size=50000)
        train_ds = train_ds.map(__get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        train_ds = train_ds.batch(self._batch_size).prefetch(1)

        valid_ds = tf.data.Dataset.from_tensor_slices(valid_data)
        valid_ds = valid_ds.map(__get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        valid_ds = valid_ds.batch(self._batch_size).prefetch(1)

        test_ds = tf.data.Dataset.from_tensor_slices(test_data)
        test_ds = test_ds.map(__get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.batch(self._batch_size).prefetch(1)
        return train_ds, valid_ds, test_ds
 
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[self._sample_rate, 1]),
            SincLayer(32, 401, 16000, 2),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv1D(64, 7, 3, padding='valid'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 7, 3, padding='valid'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 7, 3, padding='valid'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 7, 3, padding='valid'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(16, 7, 3, padding='valid'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(3)
        ])
        model.summary()
        return model

    def _collect_layers(self, model, n_cache_conv = 4):
        # Collect Layers Before Flatten
        before_layers = []
        print(model.layers)
        for layer in model.layers:
            print(layer.output_shape)
            if layer.name.find('dropout') == -1:
                if layer.name.find('flatten') >= 0:
                    break
                before_layers.append(layer)

        # Collect Layers After Flatten
        after_layers = []
        for layer in reversed(model.layers):
            if layer.name.find('dropout') == -1:
                if layer.name.find('flatten') >= 0:
                    break
                after_layers = [layer] + after_layers
        
        return before_layers, after_layers

    def run(self):
        # train_ds, valid_ds, test_ds = self._init_data()
        model = self._build_model()
        model.load_weights(self._model_dir)
        before_layers, after_layers = self._collect_layers(model)
        
        pad = tf.zeros([1, 16000, 1], dtype=tf.float32)
        
        x, _ = tf.audio.decode_wav(tf.io.read_file('four/002138_nohash_ftlk-snr10.wav'))
        x = tf.reshape(x, [1, -1, 1])
        x = tf.concat([pad, x, pad], axis=1)

        for layer in before_layers:
            x = layer(x)
        x = tf.signal.frame(x, 30, 1, axis=1)
        x = tf.reshape(x, [x.shape[1], -1])
        print(x.shape)
        for layer in after_layers:
            x = layer(x)
        x = tf.nn.softmax(x).numpy()

        thresholds = 0.8
        
        with open('tmp.log', 'w') as f:
            durs = [0, 0, 0]
            restart = 0
            for i in range(x.shape[0]):
                index = np.argmax(x[i])
                durs = [((durs[i] + 1) if i == index else 0) for i in range(len(durs))]
                restart += 1
                if (durs[index] > 8) and (restart * 486 >= 16000) and (index != 2):
                    print(f'<{index}> Trigger at {i * 486 / 16000}')
                    restart = 0
                print(f"[{','.join([str(y) for y in x[i]])}],", file=f)


def process(cfg_dir):
    p = TestTool(cfg_dir)
    p.run()

def get_template(temp_dir):
    TestTool.get_template(temp_dir)

