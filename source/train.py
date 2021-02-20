from datetime import datetime
from scipy import signal
import tensorflow as tf
import configparser
import numpy as np
import random
import wave
import ast
import os
from .data.segment import *
from .model.visualize import plot_kernel
from .model.sincnet import SincLayer

class TrainTool:
    _max_gain_db = 300.0

    @staticmethod
    def get_template(temp_dir):
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section('PATH')
        cfg['PATH'] = {
            '#data_list: list of <data> <label> pairs': '',
            'data_list': '../Data/Processed/data.list',
            '#model_dir: output model directory': '',
            'model_dir': 'models/test',
            '#log_dir: train log directory': '',
            'log_dir': 'logs/test/'
        }
        cfg.add_section('CONFIG')
        cfg['CONFIG'] = {
            '#online_augment: whether use online augment': '',
            'online_augment': 'True',
            '#label_mapping: map label to augment method (key <_default_> is saved for default augment method)': '',
            'label_mapping': '{"bg": "AUGMENT-BG", "_default_": "AUGMENT-POS"}',
            '### The following parameters must be the same as in data pre-processing': '',
            '#sample_rate: sampling rate': '',
            'sample_rate': '16000',
            '#window_size: sample window size in milisecond (pad zero to window size if shorter)': '',
            'window_size': '1000',
            '#: label for background': '',
            'background_label': 'bg',
            '#pos_left: positive samples on the left of true end of keyword': '',
            'pos_left': '3',
            '#pos_right: positive samples on the right of true end of keyword': '',
            'pos_right': '4',
            '#subsample_stride: stride of subsamples in milisecond (default is 5ms => positive window is 40ms)': '',
            'subsample_stride': '5',
            '#margin_left: margin samples on the left of positive window': '',
            'margin_left': '40',
            '#margin_right: margin samples on the right of positive window': '',
            'margin_right': '40',
            '#neg_left: negative samples on the left of margin window': '',
            'neg_left': '5',
            '#neg_right: negative samples on the right of margin window': '',
            'neg_right': '5',
        }
        cfg.add_section('AUGMENT-BG')
        cfg['AUGMENT-BG'] = {
            'effect-1-all': '[("unchange", 0.5), ("reverb", 0.5)]',
            'effect-1-unchange-type': 'unchange',
            'effect-1-reverb-type': 'reverb',
            'effect-1-reverb-reverb_list': 'data/reverb.list',
            'effect-1-reverb-memcache': 'True',
            'effect-2-all': '[("unchange", 0.25), ("noise_snr20", 0.25), ("noise_snr15", 0.25), ("noise_snr10", 0.25)]',
            'effect-2-unchange-type': 'unchange',
            'effect-2-noise_snr20-type': 'noise',
            'effect-2-noise_snr20-noise_list': 'data/noise.list',
            'effect-2-noise_snr20-snr': '20',
            'effect-2-noise_snr20-snr_randomness': '3',
            'effect-2-noise_snr15-type': 'noise',
            'effect-2-noise_snr15-noise_list': 'data/noise.list',
            'effect-2-noise_snr15-snr': '15',
            'effect-2-noise_snr15-snr_randomness': '2',
            'effect-2-noise_snr15-type': 'noise',
            'effect-2-noise_snr10-noise_list': 'data/noise.list',
            'effect-2-noise_snr10-snr': '10',
            'effect-2-noise_snr10-snr_randomness': '1',
            'effect-3-all': '[("gain", 1)]',
            'effect-3-gain-type': 'gain',
            'effect-3-gain-randomness': '2',
        }
        cfg.add_section('AUGMENT-POS')
        cfg['AUGMENT-POS'] = {
            'effect-1-all': '[("unchange", 0.5), ("reverb", 0.5)]',
            'effect-1-unchange-type': 'unchange',
            'effect-1-reverb-type': 'reverb',
            'effect-1-reverb-reverb_list': 'data/reverb.list',
            'effect-2-all': '[("unchange", 0.25), ("noise_snr20", 0.25), ("noise_snr15", 0.25), ("noise_snr10", 0.25)]',
            'effect-2-unchange-type': 'unchange',
            'effect-2-noise_snr20-type': 'noise',
            'effect-2-noise_snr20-noise_list': 'data/noise.list',
            'effect-2-noise_snr20-snr': '20',
            'effect-2-noise_snr20-snr_randomness': '3',
            'effect-2-noise_snr20-memcache': 'True',
            'effect-2-noise_snr15-type': 'noise',
            'effect-2-noise_snr15-noise_list': 'data/noise.list',
            'effect-2-noise_snr15-snr': '15',
            'effect-2-noise_snr15-snr_randomness': '2',
            'effect-2-noise_snr15-memcache': 'True',
            'effect-2-noise_snr10-type': 'noise',
            'effect-2-noise_snr10-noise_list': 'data/noise.list',
            'effect-2-noise_snr10-snr': '10',
            'effect-2-noise_snr10-snr_randomness': '1',
            'effect-2-noise_snr10-memcache': 'True',
            'effect-3-all': '[("gain", 1)]',
            'effect-3-gain-type': 'gain',
            'effect-3-gain-randomness': '2',
        }
        cfg.add_section('HYPERPARAMETERS')
        cfg['HYPERPARAMETERS'] = {
            'batch_size': '256',
        }
        with open(temp_dir, 'w') as f:
            cfg.write(f)
    
    def __init__(self, cfg_dir):
        self._parse_cfg(cfg_dir)
        self._window_size_in_sample = int(self._sample_rate * self._window_size / 1000)
        self._subsample_stride_in_sample = int(self._subsample_stride / 1000 * self._sample_rate)
        pos_label = np.zeros((self._pos_left + self._pos_right + self._neg_left + self._neg_right + 1, ))
        pos_label[self._neg_left:self._neg_left + self._pos_left + self._pos_right + 1] = 1
        self._pos_label = tf.constant(pos_label, dtype=tf.int32)
        self._reverb_cache = {}
        self._noise_cache = {}

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
        self._window_size = cfg['CONFIG'].getint('window_size')
        self._subsample_stride = cfg['CONFIG'].getint('subsample_stride')
        self._background_label = cfg['CONFIG'].get('background_label')
        self._pos_left = cfg['CONFIG'].getint('pos_left')
        self._pos_right = cfg['CONFIG'].getint('pos_right')
        self._neg_left = cfg['CONFIG'].getint('neg_left')
        self._neg_right = cfg['CONFIG'].getint('neg_right')
        self._margin_left = cfg['CONFIG'].getint('margin_left')
        self._margin_right = cfg['CONFIG'].getint('margin_right')
        self._online_augment = cfg['CONFIG'].getboolean('online_augment')
        # HYPERPARAMETERS
        self._batch_size = cfg['HYPERPARAMETERS'].getint('batch_size')

        if self._online_augment:
            self._parse_augment(cfg)
    
    def _parse_augment(self, cfg):
        self._label_mapping = ast.literal_eval(cfg['CONFIG'].get('label_mapping'))
        self._augments = {}
        for key in self._label_mapping:
            section = self._label_mapping[key]
            index = 1
            self._augments[section] = []
            while True:
                if f'effect-{index}-all' in cfg[section]:
                    effect = {}
                    effect['all'] = ast.literal_eval(cfg[section].get(f'effect-{index}-all'))
                    effect['param'] = []
                    prob_acc = 0.
                    for i in range(len(effect['all'])):
                        prob_acc += effect['all'][i][1]
                        effect['all'][i] = (effect['all'][i][0], prob_acc)
                        effect_type = cfg[section][f'effect-{index}-{effect["all"][i][0]}-type']
                        if effect_type == 'unchange':
                            effect['param'].append({'type':'unchange'})
                        elif effect_type == 'noise':
                            with open(cfg[section][f'effect-{index}-{effect["all"][i][0]}-noise_list'], 'r') as f:
                                noise_list = [x.strip('\n') for x in f.readlines()]
                            effect['param'].append({
                                'type': 'noise',
                                'noise_list': noise_list,
                                'snr': cfg[section].getint(f'effect-{index}-{effect["all"][i][0]}-snr'),
                                'snr_randomness': cfg[section].getint(f'effect-{index}-{effect["all"][i][0]}-snr_randomness'),
                                'memcache': cfg[section].getboolean(f'effect-{index}-{effect["all"][i][0]}-memcache'),
                            })
                        elif effect_type == 'reverb':
                            with open(cfg[section][f'effect-{index}-{effect["all"][i][0]}-reverb_list'], 'r') as f:
                                reverb_list = [x.strip('\n') for x in f.readlines()]
                            effect['param'].append({
                                'type': 'reverb',
                                'reverb_list': reverb_list,
                                'memcache': cfg[section].getboolean(f'effect-{index}-{effect["all"][i][0]}-memcache'),
                            })
                        elif effect_type == 'gain':
                            effect['param'].append({
                                'type': 'gain',
                                'randomness': cfg[section].getfloat(f'effect-{index}-{effect["all"][i][0]}-randomness'),
                            })
                    index += 1
                    self._augments[section].append(effect)
                else:
                    break

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
        label_dict[self._background_label] = 0
        for i in range(len(data)):
            label = data[i].split(' ')[1]
            label_dict[label] = label_dict.get(label, 0) + 1
            if not (label in self._label_mapping):
                self._label_mapping[label] = self._label_mapping['_default_']
        labels = np.array(list(label_dict.keys()))
        augment_mapping = [self._label_mapping[x] for x in labels]
        self._n_label = labels.shape[0]
        self._n_total_sample = sum([label_dict[key] for key in label_dict])
        inverse_sum = sum([1 / label_dict[key] for key in label_dict])
        self._class_weight = {}
        for i in range(len(labels)):
            self._class_weight[i] = (1 / label_dict[labels[i]]) / inverse_sum * len(labels)
        
        AUTOTUNE = tf.data.AUTOTUNE if tf.__version__.startswith('2.4') else tf.data.experimental.AUTOTUNE

        # Parse Function
        def __get_waveform_and_label(raw_data):
            content = tf.strings.split(raw_data, ' ')
            waveform, _ = tf.audio.decode_wav(tf.io.read_file(content[0]))
            waveform = tf.reshape(waveform, [-1, ])
            label = tf.argmax(content[1] == labels, output_type=tf.int32)
            return waveform, label
        
        def __get_waveform_window(waveform):
            waveform = tf.signal.frame(waveform, self._window_size_in_sample, self._subsample_stride_in_sample, axis=0)
            neg_left, _, pos, _, neg_right = tf.split(waveform, 
                [
                    self._neg_left,
                    self._margin_left,
                    self._pos_left + self._pos_right + 1,
                    self._margin_right,
                    self._neg_right,
                ],
                axis=0
            )
            waveform = tf.concat([neg_left, pos, neg_right], axis=0)
            waveform = tf.expand_dims(waveform, -1)
            return waveform
        
        def __get_label(raw_data):
            content = tf.strings.split(raw_data, ' ')
            label = tf.argmax(content[1] == labels, output_type=tf.int32)
            return label * self._pos_label
        
        that = self
        # Augment Function
        def __adjust_gain(wave, param):
            return wave

        def __data_augment(waveform, label):
            augment_name = augment_mapping[label]
            for effect in that._augments[augment_name]:
                rnd = random.uniform(0, 1)
                for i in range(len(effect['all'])):
                    if effect['all'][i][1] >= rnd:
                        break
                param = effect['param'][i]
                if param['type'] == 'unchanged':
                    pass
                elif param['type'] == 'reverb':
                    index = random.randint(0, len(param['reverb_list']) - 1)
                    if param['memcache'] and (param['reverb_list'][index] in self._reverb_cache):
                        reverb = self._reverb_cache[param['reverb_list'][index]]
                    else:
                        reverb, _ = tf.audio.decode_wav(tf.io.read_file(param['reverb_list'][index]))
                        reverb = reverb.numpy().reshape((-1, ))
                        max_ind = np.argmax(np.abs(reverb))
                        reverb = reverb[max_ind:]
                        if param['memcache']:
                            self._reverb_cache[param['reverb_list'][index]] = reverb
                    
                    audio_amplitude = np.max(np.abs(waveform))
                    delay_after = len(reverb)
                    waveform = signal.fftconvolve(waveform, reverb, "full")[:- delay_after + 1]
                    waveform = waveform / (np.max(np.abs(waveform)) + 1e-9) * audio_amplitude
                elif param['type'] == 'noise':
                    index = random.randint(0, len(param['noise_list']) - 1)
                    if param['memcache'] and (param['noise_list'][index] in self._noise_cache):
                        noise = self._noise_cache[param['noise_list'][index]]
                    else:
                        noise, _ = tf.audio.decode_wav(tf.io.read_file(param['noise_list'][index]))
                        noise = noise.numpy().reshape((-1, ))
                        if param['memcache']:
                            self._noise_cache[param['noise_list'][index]] = noise
                    left, right = get_boarder(waveform)
                    rms_db = get_rms_db(waveform, left, right)
                    noise_db = get_rms_db(noise, 0, noise.shape[0])
                    random_perturb = random.uniform(-param['snr_randomness'], param['snr_randomness'])
                    noise_gain_db = min(rms_db - noise_db - param['snr'] + random_perturb, TrainTool._max_gain_db)
                    time_index = random.randint(0, noise.shape[0] - waveform.shape[0])
                    noise_sample = noise[time_index:time_index + waveform.shape[0]] * (10.0 ** (noise_gain_db / 20.0))
                    waveform = waveform + noise_sample
                elif param['type'] == 'gain':
                    gain_db = random.uniform(-param['randomness'], param['randomness'])
                    waveform = waveform * (10.0 ** (gain_db / 20.0))
                    amplitude = np.max(np.abs(waveform))
                    if amplitude > 0.99:
                        waveform = waveform / amplitude * 0.99
            return waveform
        
        def __build_dataset(data):
            random.seed(321)
            random.shuffle(data)
            label = tf.data.Dataset.from_tensor_slices(data)
            label = label.map(__get_label, num_parallel_calls=AUTOTUNE).unbatch()
            wave = tf.data.Dataset.from_tensor_slices(data)
            wave = wave.map(__get_waveform_and_label, num_parallel_calls=AUTOTUNE)
            wave = wave.map(lambda wave, label: tf.py_function(__data_augment, (wave, label), Tout=tf.float32), num_parallel_calls=AUTOTUNE)
            wave = wave.map(__get_waveform_window, num_parallel_calls=AUTOTUNE).unbatch()
            dataset = tf.data.Dataset.zip((wave, label)).shuffle(10000).batch(self._batch_size)
            return dataset

        train_ds = __build_dataset(train_data)
        valid_ds = __build_dataset(valid_data)
        test_ds = __build_dataset(test_data)
        
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
            tf.keras.layers.Dense(self._n_label)
        ])
        model.summary()
        return model

    def run(self):
        train_ds, valid_ds, test_ds = self._init_data()

        # for wave, label in train_ds.take(1):
        #     tf.io.write_file('tmp.wav', tf.audio.encode_wav(wave[0], sample_rate=self._sample_rate))
        #     print(label[0])
        #     exit(0)

        model = self._build_model()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self._model_dir,
                save_best_only=True,
                save_weights_only=True
            ),
        ]

        f = open(self._log_dir + '.log', 'w')
        low_cutoff, bandwidth = model.get_layer(index=0).get_weights()
        print(low_cutoff, file=f)
        print(bandwidth, file=f)
        plot_kernel(low_cutoff.reshape((-1, 1)), bandwidth.reshape((-1, 1)), 50 / 16000, 50 / 16000, 16000, 401, self._log_dir + '_init')

        history = model.fit(
            train_ds,
            epochs=50,
            class_weight=self._class_weight,
            validation_data=valid_ds,
            callbacks=callbacks)
        # model.save_weights(self._model_dir)
        print(history.history, file=f)

        low_cutoff, bandwidth = model.get_layer(index=0).get_weights()
        print(low_cutoff, file=f)
        print(bandwidth, file=f)
        plot_kernel(low_cutoff.reshape((-1, 1)), bandwidth.reshape((-1, 1)), 50 / 16000, 50 / 16000, 16000, 401, self._log_dir + '_train')

        print(model.evaluate(test_ds), file=f)
        f.close()

def process(cfg_dir):
    p = TrainTool(cfg_dir)
    p.run()

def get_template(temp_dir):
    TrainTool.get_template(temp_dir)


