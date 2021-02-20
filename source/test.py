from datetime import datetime
import tensorflow as tf
import configparser
import numpy as np
import ast
import os
import time
from .data.segment import assign_division
from .model.visualize import plot_kernel
from .model.sincnet import SincLayer

class TestTool:
    @staticmethod
    def get_template(temp_dir):
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section('PATH')
        cfg['PATH'] = {
            '#data_list: list of audio': '',
            'data_list': 'data/fr_test.list',
            '#model_dir: output model directory': '',
            'model_dir': 'models/test',
            '#log_dir: train log directory': '',
            'log_dir': 'logs/test/'
        }
        cfg.add_section('CONFIG')
        cfg['CONFIG'] = {
            '#mode: FR (false reject test) or FA (false alarm test)': '',
            'mode': 'FR',
            '#sample_rate: sampling rate': '',
            'sample_rate': '16000',
            '#durations: options of durations for activate': '',
            'durations': '[5, 6, 7, 8]',
            '#thresholds, options of thresholds for activate': '',
            'thresholds': '[0.75, 0.8, 0.85, 0.9]',
        }
        with open(temp_dir, 'w') as f:
            cfg.write(f)
    
    def __init__(self, cfg_dir):
        self._parse_cfg(cfg_dir)
        self._pad = tf.zeros([self._sample_rate, 1], dtype=tf.float32)

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
        self._mode = cfg['CONFIG']['mode']
        self._sample_rate = cfg['CONFIG'].getint('sample_rate')
        self._thresholds = ast.literal_eval(cfg['CONFIG'].get('thresholds'))
        self._durations = ast.literal_eval(cfg['CONFIG'].get('durations'))

    def _init_data(self):
        # Read Data List
        with open(self._data_list, 'r') as f:
            lines = f.readlines()
        test_data = [x.strip('\n') for x in lines]
        self._n_total = len(test_data)

        # Parse Function
        def __get_waveform(raw_data):
            waveform, _ = tf.audio.decode_wav(tf.io.read_file(raw_data))
            waveform = tf.concat([self._pad, waveform, self._pad], axis=0)
            return waveform
        
        AUTOTUNE = tf.data.AUTOTUNE if tf.__version__.startswith('2.4') else tf.data.experimental.AUTOTUNE
        test_ds = tf.data.Dataset.from_tensor_slices(test_data)
        test_ds = test_ds.map(__get_waveform, num_parallel_calls=AUTOTUNE)
        test_ds = test_ds.batch(1).prefetch(1)
        return test_ds
 
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
            tf.keras.layers.Dense(2)
        ])
        model.summary()
        return model

    def _collect_layers(self, model, n_cache_conv = 4):
        # Collect Layers Before Flatten
        before_layers = []
        for layer in model.layers:
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
        test_ds = self._init_data()
        model = self._build_model()
        model.load_weights(self._model_dir)
        before_layers, after_layers = self._collect_layers(model)
        
        accept = [[0 for col in range(len(self._durations))] for row in range(len(self._thresholds))]

        cnt = 0
        self._len_total = 0
        for x in test_ds:
            self._len_total += x.shape[1] - 2 * self._sample_rate
            cnt += 1
            print(f'Processing {cnt}', end='\r')
            start = time.time()

            for layer in before_layers:
                x = layer(x)
            x = tf.signal.frame(x, 30, 1, axis=1)
            x = tf.reshape(x, [x.shape[1], -1])
            for layer in after_layers:
                x = layer(x)
            x = tf.nn.softmax(x).numpy()

            # print(f'Inference Time Eclipsed: {time.time() - start}')
            start = time.time()
            # with open('tmp.log', 'w') as f:
            #     for i in range(x.shape[0]):
            #         print(f'[{x[i][0]}, {x[i][1]}],', file=f)

            for threshold_index in range(len(self._thresholds)):
                for duration_index in range(len(self._durations)):
                    threshold = self._thresholds[threshold_index]
                    duration = self._durations[duration_index]

                    durs = [0, 0]
                    restart = 0
                    for i in range(x.shape[0]):
                        index = np.argmax(x[i])
                        if x[i][index] >= threshold:
                            durs = [((durs[i] + 1) if i == index else 0) for i in range(len(durs))]
                        else:
                            durs = [0 for _ in durs]
                        restart += 1
                        if (durs[index] >= duration) and (restart * 486 >= 16000) and (index != 0):
                            accept[threshold_index][duration_index] += 1
                            if self._mode == 'FR':
                                break
                            restart = 0
                            durs = [0 for _ in durs]
            # print(f'Loop Time Eclipsed: {time.time() - start}')
            len_total = self._len_total / self._sample_rate / 60 / 60
            # for threshold_index in range(len(self._thresholds)):
            #     for duration_index in range(len(self._durations)):
            #         if self._mode == 'FR':
            #             print(1 - accept[threshold_index][duration_index] / self._n_total, end='\t')
            #         else:
            #             print(accept[threshold_index][duration_index] / len_total, end='\t')
            #     print()
        self._len_total = self._len_total / self._sample_rate / 60 / 60
        
        for threshold_index in range(len(self._thresholds)):
            for duration_index in range(len(self._durations)):
                if self._mode == 'FR':
                    print(1 - accept[threshold_index][duration_index] / self._n_total, end='\t')
                else:
                    print(accept[threshold_index][duration_index] / self._len_total, end='\t')
            print()

def process(cfg_dir):
    p = TestTool(cfg_dir)
    p.run()

def get_template(temp_dir):
    TestTool.get_template(temp_dir)

