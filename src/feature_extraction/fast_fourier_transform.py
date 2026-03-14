import numpy as np


class FFTFeature:

    def __init__(self, fs=250):
        self.fs = fs

    def extract(self, X):

        trials, channels, samples = X.shape

        features = []

        for t in range(trials):

            trial_feature = []

            for c in range(channels):

                signal = X[t, c, :]

                fft_vals = np.fft.rfft(signal)

                fft_power = np.abs(fft_vals)

                trial_feature.append(np.mean(fft_power))

            features.append(trial_feature)

        return np.array(features)