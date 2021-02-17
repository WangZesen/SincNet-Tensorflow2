import os
from functools import partial
import subprocess
import configparser
import numpy as np

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
    
    def _normalize_audio(self, data):
        n_samples = data.shape[0]
        if n_samples > self._window_size_in_sample:
            data = data[:self._window_size_in_sample]
        elif n_samples < self._window_size_in_sample:
            pad = np.zeros((self._window_size_in_sample - n_samples, ))
            data = np.concatenate((pad, data), axis=0)
        return data
    
    def _split_bg_audio(self, data):
        n_samples = data.shape[0]
        if n_samples < self._window_size_in_sample:
            pad = np.zeros((self._window_size_in_sample - n_samples, ))
            data = np.concatenate((pad, data), axis=0)
            n_samples = self._window_size_in_sample
        cnt = (n_samples - self._window_size_in_sample) // self._window_stride_in_sample
        samples = []
        for index in range(cnt + 1):
            start = index * self._window_stride_in_sample
            end = self._window_size_in_sample + index * self._window_size_in_sample
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
                    sample = self._normalize_audio(data)
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

