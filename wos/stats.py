from dataclasses import dataclass
import numpy as np


@dataclass
class Statistics:

    def mean(self, x):
        _mean = 0.
        _means = np.zeros(len(x))
        for i in range(len(x)):
            delta = x[i] - _mean
            _mean += delta / (i + 1)
            _means[i] = _mean
        return _means

    def m2(self, x):
        _mean = 0.
        _means = np.zeros(len(x))
        m2 = 0.
        m2s = np.zeros(len(x))
        for i in range(len(x)):
            delta = x[i] - _mean
            _mean += delta / (i + 1)
            _means[i] = _mean
            m2 += delta * (x[i] - _mean)
            m2s[i] = m2
        return m2s

    def var(self, x):
        m2s = self.m2(x)
        return m2s / (np.arange(len(x)) + 1)

    def ci(self, x):
        vars = self.var(x)
        return 1.96 * np.sqrt(vars / np.sqrt(np.arange(len(x)) + 1))
