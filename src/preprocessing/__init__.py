from .band_pass_filter import BandpassFilter
from .Notch_filter import notch_filter

class Preprocessing:

    def __init__(self, fs):
        self.fs = fs

    def apply(self, X):

        # notch
        notch = Notch_filter(freq=50, fs=self.fs)
        X = notch.apply(X)

        # bandpass
        bandpass = BandpassFilter(
            lowcut=8,
            highcut=30,
            fs=self.fs
        )
        X = bandpass.apply(X)

        return X