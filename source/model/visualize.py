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
    
    plt.savefig(out_dir + '_freq.jpg', dpi=250)
    plt.clf()

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

    plt.savefig(out_dir + '_time.jpg', dpi=250)
    plt.clf()

if __name__ == '__main__':
    low_cutoff = np.array([0.00139425, 0.00422874, 0.00355249, 0.01436869, 0.00675843,
       0.01407022, 0.0180199 , 0.0168311 , 0.02393385, 0.03018662,
       0.01878617, 0.02628963, 0.03866723, 0.03028007, 0.03880114,
       0.03517905, 0.04528445, 0.04180424, 0.0495062 , 0.04930846,
       0.05448617, 0.06154127, 0.06894644, 0.06299099, 0.06844902,
       0.05731545, 0.08610705, 0.07940432, 0.1021184 , 0.11108598,
       0.10251453, 0.10149112, 0.10951603, 0.11364559, 0.12506545,
       0.11811613, 0.13844275, 0.13490482, 0.14236335, 0.15899585,
       0.16098686, 0.18965852, 0.1744484 , 0.16958648, 0.20083037,
       0.21073727, 0.21979141, 0.2530929 , 0.24204996, 0.27679956,
       0.2738742 , 0.28530437, 0.3145513 , 0.33402744, 0.3319927 ,
       0.34285498, 0.35666904, 0.37881398, 0.40194657, 0.40280867,
       0.4329893 , 0.45988062, 0.45698074, 0.4846047 ]).reshape((-1, 1))
    bandwidth = np.array([0.00057024, 0.00173413, 0.00290349, 0.01148339, 0.00407171,
       0.0021181 , 0.00058345, 0.01207492, 0.01173795, 0.00818146,
       0.01185847, 0.01189668, 0.00705692, 0.02254369, 0.0070699 ,
       0.00752428, 0.0119742 , 0.01197028, 0.01502354, 0.01638624,
       0.01430839, 0.0161906 , 0.01689068, 0.01270346, 0.00747843,
       0.01162465, 0.01698497, 0.00931537, 0.00662364, 0.01666979,
       0.01156814, 0.01637925, 0.0113051 , 0.01548607, 0.0175532 ,
       0.00581662, 0.00554296, 0.0266373 , 0.011229  , 0.01740404,
       0.02019789, 0.01279477, 0.01191997, 0.03738416, 0.03144418,
       0.02513716, 0.01198943, 0.0158779 , 0.01389407, 0.01135144,
       0.0124943 , 0.02153225, 0.02366693, 0.03803054, 0.04510175,
       0.02540432, 0.03429833, 0.02799218, 0.01124583, 0.0158293 ,
       0.01970603, 0.02209977, 0.02098785, 0.02148775]).reshape((-1, 1))
    plot_kernel(low_cutoff, bandwidth, 50 / 16000, 50 / 16000, 16000, 401, None)
    