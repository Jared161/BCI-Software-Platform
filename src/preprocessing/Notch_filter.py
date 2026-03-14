import numpy as np
from scipy.signal import iirnotch, filtfilt

def notch_filter(signal, fs=250, freq=50, Q=30):

    """
    Notch Filter 去除电源噪声

    signal : EEG信号
    fs     : 采样率
    freq   : 要去除的频率 (50Hz)
    Q      : 品质因数(滤波器带宽)
    """

    b, a = iirnotch(freq, Q, fs)

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal