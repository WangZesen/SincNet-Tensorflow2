import os
from functools import partial
import subprocess
import configparser
import numpy as np
from .data.vad import get_interval

class Preprocess:
    @staticmethod
    def get_template(temp_dir):
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section('PATH')
        cfg['PATH'] = {
            '#data_list: list of <data> <label> pairs': '',
            'data_list': 'test.list',
            '#output_dir: output directory': '',
            'output_dir': 'out',
        }
        cfg.add_section('CONFIG')
        cfg['CONFIG'] = {
            '#window_size: sample window size in milisecond (pad zero to window size if shorter)': '',
            'window_size': '1000',
            '#window_stride: stride length in milisecond': '',
            'window_stride': '797',
            '#sample_rate: sampling rate': '',
            'sample_rate': '16000',
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
            '#background_label: label for background': '',
            'background_label': 'bg',
        }
        with open(temp_dir, 'w') as f:
            cfg.write(f)
    
    def __init__(self, cfg_dir):
        self._parse_cfg(cfg_dir)
        self._cnt = 0
        self._window_size_in_sample = int(self._sample_rate * self._window_size / 1000)
        self._window_stride_in_sample = int(self._sample_rate * self._window_stride / 1000)
        self._subsample_stride_in_sample = int(self._sample_rate * self._subsample_stride / 1000)
        extended = self._pos_left + self._margin_left + self._neg_left + self._pos_right + self._margin_right + self._neg_right
        self._bg_window_size_in_sample = self._window_size_in_sample + extended * self._subsample_stride_in_sample
        os.makedirs(self._output_dir, exist_ok=True)

    def _parse_cfg(self, cfg_dir):
        cfg = configparser.ConfigParser()
        cfg.read(cfg_dir)
        # PATH
        self._data_list = cfg['PATH'].get('data_list')
        self._output_dir = cfg['PATH'].get('output_dir')
        # CONFIG
        self._window_size = cfg['CONFIG'].getint('window_size')
        self._window_stride = cfg['CONFIG'].getint('window_stride')
        self._sample_rate = cfg['CONFIG'].getint('sample_rate')
        self._background_label = cfg['CONFIG'].get('background_label')
        self._pos_left = cfg['CONFIG'].getint('pos_left')
        self._pos_right = cfg['CONFIG'].getint('pos_right')
        self._margin_left = cfg['CONFIG'].getint('margin_left')
        self._margin_right = cfg['CONFIG'].getint('margin_right')
        self._neg_left = cfg['CONFIG'].getint('neg_left')
        self._neg_right = cfg['CONFIG'].getint('neg_right')
        self._subsample_stride = cfg['CONFIG'].getint('subsample_stride')
    
    def _normalize_audio(self, data, end):
        left = end - (self._pos_left + self._margin_left + self._neg_left) * self._subsample_stride_in_sample - self._window_size_in_sample
        if left < 0:
            pad = np.zeros((- left, ))
            data = np.concatenate((pad, data), axis=0)
            end = end + (- left)
        else:
            data = data[left:]
            end = end - left
        right = end + (self._pos_right + self._margin_right + self._neg_right) * self._subsample_stride_in_sample
        n_samples = data.shape[0]
        if n_samples >= right:
            data = data[:right]
        else:
            pad = np.zeros((right - n_samples, ))
            data = np.concatenate((pad, data), axis=0)
        return data
    
    def _split_bg_audio(self, data):
        n_samples = data.shape[0]
        if n_samples < self._bg_window_size_in_sample:
            pad = np.zeros((self._bg_window_size_in_sample - n_samples, ))
            data = np.concatenate((pad, data), axis=0)
            n_samples = self._bg_window_size_in_sample
        cnt = (n_samples - self._bg_window_size_in_sample) // self._bg_window_size_in_sample
        samples = []
        for index in range(cnt + 1):
            start = index * self._bg_window_size_in_sample
            end = self._bg_window_size_in_sample + index * self._bg_window_size_in_sample
            samples.append(data[start:end])
        return samples
    
    def _next_output_dir(self, label):
        out_file = os.path.join(self._output_dir, label, f'{str(self._cnt).zfill(6)}_nohash_.wav')
        self._cnt += 1
        return out_file
    
    def _write_summary(self, label_dict):
        summary_dir = os.path.join(self._output_dir, 'summary.txt')
        with open(summary_dir, 'w') as f:
            printf = partial(print, file=f)
            printf(f'Output Directory: {os.path.abspath(self._output_dir)}')
            printf('Label Distribution:')
            for key in label_dict:
                printf(f'\t{key}: {label_dict[key]}')

    def run(self):
        import tensorflow as tf
        tmp_file = '_temp_.wav'
        label_dict = {}
        fw = open(os.path.join(self._output_dir, 'data.list'), 'w')

        with open(self._data_list, 'r') as f:
            line = f.readline()
            while line:
                content = line.strip('\n').split(' ')
                audio_dir, label = content[0], content[1]
                subprocess.check_call(f'sox {audio_dir} -t wav -r {self._sample_rate} -b 16 {tmp_file}', shell=True)
                data, sample_rate = tf.audio.decode_wav(tf.io.read_file(tmp_file))
                assert sample_rate == self._sample_rate
                data = data.numpy().reshape((-1))
                if not (label in label_dict):
                    label_dict[label] = 0
                    os.makedirs(os.path.join(self._output_dir, label), exist_ok=True)
                label_dict[label] += 1
                if label == self._background_label:
                    data = self._split_bg_audio(data)
                    for sample in data:
                        out_file = self._next_output_dir(label)
                        tf.io.write_file(out_file, tf.audio.encode_wav(sample.reshape((-1, 1)), self._sample_rate))
                        print(f'{os.path.abspath(out_file)} {label}', file=fw)
                else:
                    start, end = get_interval(tmp_file)
                    if end > 0:
                        end = int(end / 1000 * self._sample_rate)
                        sample = self._normalize_audio(data, end)
                        out_file = self._next_output_dir(label)
                        tf.io.write_file(out_file, tf.audio.encode_wav(sample.reshape((-1, 1)), self._sample_rate))
                        print(f'{os.path.abspath(out_file)} {label}', file=fw)
                line = f.readline()

        os.remove(tmp_file)
        fw.close()
        self._write_summary(label_dict)

def process(cfg_dir):
    p = Preprocess(cfg_dir)
    p.run()

def get_template(temp_dir):
    Preprocess.get_template(temp_dir)

