from .power_spectral_density import PSDFeature
from .fast_fourier_transform import FFTFeature
import numpy as np

class FeatureExtractor:

    def __init__(self, fs=250):

        self.psd = PSDFeature(fs)
        self.fft = FFTFeature(fs)

    def extract(self, X):

        psd_feat = self.psd.extract(X)

        fft_feat = self.fft.extract(X)

        # 拼接特征
        features = np.concatenate([psd_feat, fft_feat], axis=1)

        return features