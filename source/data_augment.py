from .data.segment import *
from functools import partial
from scipy import signal
import tensorflow as tf
import numpy as np
import copy as cp
import configparser
import subprocess
import scipy.io
import random
import shutil
import os

class DataAugment:
    _max_gain_db = 300.0

    @staticmethod
    def get_template(temp_dir):
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.add_section('PATH')
        cfg['PATH'] = {
            '### Augmented data is created in the sample directory with different suffix'
            '#data_list: list of <data> <label> pairs': '',
            'data_list': 'test.list',
        }
        cfg.add_section('AUGMENT-SAMPLE')
        cfg['AUGMENT-SAMPLE'] = {
            '### It can have multiple section starts with "AUGMENT-<ANY>"': '',
            '#suffix: suffix to be added after _nohash_': '',
            'suffix': 'sample',
            '#recipe: detailed recipe, comes by "<effect1>-<param1>, <effect2>-<param2>, ...". Available effects are shown below': '',
            'effect-1-type': 'shift',
            'effect-2-type': 'noise',
            'effect-2-noise_list': 'noise.list',
            'effect-2-snr': '20',
            'effect-2-snr_randomness': '1',
            'effect-3-type': 'reverb',
            'effect-3-reverb_list': 'reverb.list',
            'effect-3-reverb_normalize': 'True',
        }

        with open(temp_dir, 'w') as f:
            cfg.write(f)
    
    def __init__(self, cfg_dir):
        self._parse_cfg(cfg_dir)
        self._effect_handler = {
            'noise': self._add_noise,
            'reverb': self._add_reverb,
            'shift': self._apply_shift,
        }
        with open(self._data_list, 'r') as f:
            self._data = [x.strip('\n').split(' ') for x in f.readlines()]
        self._data_rms_cache = {}

    def _parse_cfg(self, cfg_dir):
        cfg = configparser.ConfigParser()
        cfg.read(cfg_dir)
        # PATH
        self._data_list = cfg['PATH'].get('data_list')
        # CONFIG
        self._effects = []
        for section in cfg.sections():
            if section.startswith('AUGMENT-'):
                effect = {
                    'name': section,
                    'pipeline': [],
                    'suffix': cfg[section].get('suffix'),
                }
                done = False
                cnt = 1
                while not done:
                    try:
                        pipeline = {
                            'type': cfg[section].get(f'effect-{cnt}-type'),
                        }
                        if pipeline['type'] == 'reverb':
                            pipeline['reverb_list'] = cfg[section].get(f'effect-{cnt}-reverb_list')
                            pipeline['reverb_normalize'] = cfg[section].getboolean(f'effect-{cnt}-reverb_normalize', True)
                        elif pipeline['type'] == 'noise':
                            pipeline['noise_list'] = cfg[section].get(f'effect-{cnt}-noise_list')
                            pipeline['snr'] = cfg[section].getint(f'effect-{cnt}-snr')
                            pipeline['snr_randomness'] = cfg[section].getint(f'effect-{cnt}-snr_randomness', 0)
                        elif pipeline['type'] == 'shift':
                            pass
                        else:
                            raise Exception('Invalid Type Name')
                        cnt += 1
                        effect['pipeline'].append(pipeline)
                    except:
                        done = True
                if len(effect['pipeline']) > 0:
                    self._effects.append(effect)
    
    def _add_noise(self, pipeline, suffix, data_list):
        new_data_list = []
        random_suffix = '.' + str(random.randint(1e5, 1e6))
        with open(pipeline['noise_list'], 'r') as f:
            noise_list = [x.strip('\n') for x in f.readlines() if len(x) > 1]
        
        noise_audio = []
        noise_db = []
        for noise_dir in noise_list:
            audio, _ = tf.audio.decode_wav(tf.io.read_file(noise_dir))
            audio = audio.numpy().reshape((-1, ))
            noise_audio.append(audio)
            noise_db.append(get_rms_db(audio, 0, audio.shape[0]))
        
        for i in range(len(self._data)):
            audio_dir = data_list[i]
            audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(audio_dir))
            audio = audio.numpy().reshape((-1, ))
            left, right = get_boarder(audio)
            rms_db = get_rms_db(audio, left, right)
            noise_index = random.randint(0, len(noise_audio) - 1)
            random_perturb = random.uniform(-pipeline['snr_randomness'], pipeline['snr_randomness'])
            noise_gain_db = min(rms_db - noise_db[noise_index] - pipeline['snr'] + random_perturb, DataAugment._max_gain_db)
            assert noise_audio[noise_index].shape[0] >= audio.shape[0]
            random_index = random.randint(0, noise_audio[noise_index].shape[0] - audio.shape[0])
            noise_sample = noise_audio[noise_index][random_index:random_index + audio.shape[0]] * (10.0 ** (noise_gain_db / 20.0))
            audio = audio + noise_sample
            noisy_audio_dir = self._data[i][0] + random_suffix
            hash_index = self._data[i][0].find('_nohash_') + 8
            assert hash_index >= 8  # Must Contain _nohash_
            final_audio_dir = self._data[i][0][:hash_index] + suffix + ('_' if self._data[i][0][hash_index] != '.' else '') + self._data[i][0][hash_index:]
            tf.io.write_file(noisy_audio_dir, tf.audio.encode_wav(audio.reshape((-1, 1)), sample_rate))
            shutil.copyfile(noisy_audio_dir, final_audio_dir)
            os.remove(noisy_audio_dir)
            new_data_list.append(final_audio_dir)
        return new_data_list

    def _add_reverb(self, pipeline, suffix, data_list):
        new_data_list = []
        random_suffix = '.' + str(random.randint(1e3, 1e4))
        with open(pipeline['reverb_list'], 'r') as f:
            reverb_list = [x.strip('\n') for x in f.readlines()]
        reverbs = []
        for reverb_dir in reverb_list:
            audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(reverb_dir))
            audio = audio.numpy().reshape((-1, ))
            reverb_norm = (audio - np.min(audio)) / (np.max(audio) - np.min(audio))
            reverbs.append(reverb_norm)
        for i in range(len(self._data)):
            audio_dir = data_list[i]
            audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(audio_dir))
            audio = audio.numpy().reshape((-1, ))
            reverb_index = random.randint(0, len(reverbs) - 1)
            
            print(np.mean(audio))
            print(np.sum(np.multiply(audio, reverbs[reverb_index][:16000])))
            
            audio = signal.fftconvolve(audio, reverbs[reverb_index], "same")
            
            print(audio.shape)
            exit(0)
            reverb_audio_dir = self._data[i][0] + random_suffix
            hash_index = self._data[i][0].find('_nohash_') + 8
            assert hash_index >= 8  # Must Contain _nohash_
            final_audio_dir = self._data[i][0][:hash_index] + suffix + ('_' if self._data[i][0][hash_index] != '.' else '') + self._data[i][0][hash_index:]
            tf.io.write_file(reverb_audio_dir, tf.audio.encode_wav(audio.reshape((-1, 1)), sample_rate))
            shutil.copyfile(reverb_audio_dir, final_audio_dir)
            os.remove(reverb_audio_dir)
            new_data_list.append(final_audio_dir)
        return new_data_list

    def _apply_shift(self, pipeline, suffix, data_list):
        new_data_list = []
        random_suffix = '.' + str(random.randint(1e3, 1e4))
        for i in range(len(self._data)):
            audio_dir = data_list[i]
            audio, sample_rate = tf.audio.decode_wav(tf.io.read_file(audio_dir))
            audio = audio.numpy().reshape((-1, ))
            shifted = np.zeros_like(audio)
            left, right = get_boarder(audio)
            left = max(left - 100, 0)
            right = min(right + 100, audio.shape[0])
            if audio.shape[0] - (right - left + 1) > 0:
                random_index = random.randint(0, audio.shape[0] - (right - left + 1))
            else:
                random_index = 0
            shifted[random_index:random_index + right - left] = audio[left:right]
            shifted = shifted.reshape((-1, 1))
            shifted_audio_dir = self._data[i][0] + random_suffix
            hash_index = self._data[i][0].find('_nohash_') + 8
            assert hash_index >= 8  # Must Contain _nohash_
            final_audio_dir = self._data[i][0][:hash_index] + suffix + ('_' if self._data[i][0][hash_index] != '.' else '') + self._data[i][0][hash_index:]
            tf.io.write_file(shifted_audio_dir, tf.audio.encode_wav(shifted, sample_rate))
            shutil.copyfile(shifted_audio_dir, final_audio_dir)
            os.remove(shifted_audio_dir)
            new_data_list.append(final_audio_dir)
        return new_data_list

    def run(self):
        for effect in self._effects:
            inplace = False
            data_list = list(list(zip(*self._data))[0])
            for pipeline in effect['pipeline']:
                data_list = self._effect_handler[pipeline['type']](pipeline, effect['suffix'], data_list)
        
def process(cfg_dir):
    p = DataAugment(cfg_dir)
    p.run()

def get_template(temp_dir):
    DataAugment.get_template(temp_dir)

