import numpy as np
from scipy.signal import welch


class PSDFeature:

    def __init__(self, fs=250):
        self.fs = fs

    def extract(self, X):
        """
        输入:
        X shape = (trials, channels, samples)

        输出:
        features shape = (trials, channels)
        """

        trials, channels, samples = X.shape

        features = []

        for t in range(trials):

            trial_feature = []

            for c in range(channels):

                signal = X[t, c, :]

                freqs, psd = welch(
                    signal,
                    fs=self.fs,
                    nperseg=256
                )

                # 取8-30Hz功率
                band_mask = (freqs >= 8) & (freqs <= 30)

                band_power = np.mean(psd[band_mask])

                trial_feature.append(band_power)

            features.append(trial_feature)

        return np.array(features)  #输出特征(trials , channels)