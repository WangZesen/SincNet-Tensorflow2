import numpy as np
import math
import matplotlib.pyplot as plt

def plot_kernel(low_cutoff, bandwidth, min_freq, min_bandwidth, sample_rate, filter_dim, out_dir):
    # Frequency Domain
    _low_cutoff = low_cutoff + min_freq
    _high_cutoff = _low_cutoff + bandwidth + min_bandwidth

    for i in range(len(_low_cutoff)):
        x = [0, _low_cutoff[i][0] * sample_rate, (_low_cutoff[i][0] + _high_cutoff[i][0]) / 2 * sample_rate, _high_cutoff[i][0] * sample_rate, sample_rate]
        y = [0, 0, 1 / (bandwidth[i][0] + min_bandwidth), 0, 0]
        plt.plot(x, y)
    
    plt.show()

    # Time Domain
    col = 8
    row = len(_low_cutoff) // col + (1 if len(_low_cutoff) % col else 0)
    n = np.linspace(1, (filter_dim - 1) / 2, int((filter_dim - 1) / 2)).reshape((1, -1))  # Half of filter
    n = n / sample_rate
    
    inter_low = 2 * math.pi * np.matmul(_low_cutoff * sample_rate, n)
    low_low_pass = 2 * np.multiply(np.sin(inter_low) / inter_low, _low_cutoff)
    inter_high = 2 * math.pi * np.matmul(_high_cutoff * sample_rate, n)
    high_low_pass = 2 * np.multiply(np.sin(inter_high) / inter_high, _high_cutoff)

    band_pass = high_low_pass - low_low_pass
    
    for i in range(len(_low_cutoff)):
        plt.subplot(row, col, i + 1)
        plt.plot(n[0], band_pass[i])
    plt.show()


if __name__ == '__main__':
    low_cutoff = np.array([0.001875  , 0.00321849, 0.00508026, 0.00701582, 0.00902811,
       0.01112017, 0.01329515, 0.01555634, 0.01790716, 0.02035117,
       0.02289206, 0.02553366, 0.02827997, 0.03113514, 0.03410349,
       0.0371895 , 0.04039783, 0.04373334, 0.04720106, 0.05080624,
       0.05455432, 0.05845098, 0.06250209, 0.06671378, 0.07109242,
       0.07564462, 0.08037726, 0.0852975 , 0.09041277, 0.0957308 ,
       0.10125963, 0.10700762, 0.11298346, 0.11919616, 0.12565513,
       0.13237013, 0.1393513 , 0.14660919, 0.15415476, 0.16199945,
       0.17015508, 0.17863399, 0.187449  , 0.19661342, 0.2061411 ,
       0.21604645, 0.22634444, 0.23705062, 0.24818118, 0.25975296,
       0.2717834 , 0.28429073, 0.2972938 , 0.31081235, 0.3248667 ,
       0.3394782 , 0.35466886, 0.37046164, 0.38688043, 0.40395007,
       0.4216963 , 0.44014597, 0.45932695, 0.47926825]).reshape((-1, 1))
    bandwidth = np.array([0.00320526, 0.00379733, 0.00394786, 0.00410434, 0.00426703,
       0.00443617, 0.00461202, 0.00479483, 0.00498489, 0.00518249,
       0.00538791, 0.00560148, 0.00582352, 0.00605436, 0.00629434,
       0.00654384, 0.00680323, 0.0070729 , 0.00735326, 0.00764474,
       0.00794776, 0.0082628 , 0.00859033, 0.00893084, 0.00928484,
       0.00965288, 0.01003551, 0.0104333 , 0.01084687, 0.01127682,
       0.01172382, 0.01218854, 0.01267167, 0.01317396, 0.01369616,
       0.01423906, 0.01480348, 0.01539027, 0.01600032, 0.01663455,
       0.01729392, 0.01797942, 0.01869211, 0.01943304, 0.02020334,
       0.02100417, 0.02183675, 0.02270233, 0.02360222, 0.02453778,
       0.02551042, 0.02652162, 0.0275729 , 0.02866585, 0.02980213,
       0.03098345, 0.03221159, 0.03348842, 0.03481585, 0.0361959 ,
       0.03763066, 0.03912229, 0.04067305, 0.01448175]).reshape((-1, 1))
    plot_kernel(low_cutoff, bandwidth, 50 / 16000, 50 / 16000, 16000, 401, None)
    