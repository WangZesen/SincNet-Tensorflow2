import os
import re
import hashlib
import numpy as np

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1

def assign_division(filename, train_perc, valid_perc):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < train_perc:
        return 'train'
    elif percentage_hash < (train_perc + valid_perc):
        return 'valid'
    else:
        return 'test'

def get_boarder(data):
    cum_sum = np.cumsum(np.abs(data))
    left = np.searchsorted(cum_sum, 0.01 * cum_sum[-1], 'left')
    right = np.searchsorted(cum_sum, 0.99 * cum_sum[-1], 'right')
    return left, right

def get_rms_db(data, left, right):
    mean_square = np.mean(data[left:right] ** 2)
    return 10 * np.log10(mean_square)