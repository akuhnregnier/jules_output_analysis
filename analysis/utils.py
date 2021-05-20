# -*- coding: utf-8 -*-
import numpy as np


class Alpha:
    def __init__(self, alpha, signal, thres=0.001):
        """Initialise the Alpha averaging scheme.

        Parameters
        ----------
        thres : float
            The maximum fractional difference from one spinup cycle to the next has to
            be lower than this for the spinup to succeed.

        """
        self.alpha = alpha
        self.signal = signal
        self.N = len(signal)
        self.weighted = np.zeros_like(signal, dtype=np.float64)
        self.thres = thres
        self._spinup()

    def _spinup(self):
        """Spinup of the averaging."""
        prev_weighted = None
        counter = 0
        while True:
            # Perform a single spinup cycle.
            for i, sample in enumerate(self.signal):
                temp = self.weighted[(i - 1) % self.N]
                temp *= 1 - self.alpha
                temp += self.alpha * sample
                self.weighted[i] = temp

            if prev_weighted is not None:
                # Assess the convergence.
                if (
                    np.max(np.abs(self.weighted - prev_weighted) / self.weighted)
                    < self.thres
                ):
                    break

            prev_weighted = self.weighted.copy()
            counter += 1

            if counter > 1000:
                break
        return self


class Antecedent:
    def __init__(self, antec_samples, signal):
        self.antec_samples = antec_samples
        self.signal = signal
        self.N = len(signal)
        self.antec = self.signal[(np.arange(self.N) - self.antec_samples) % self.N]


class Memory:
    def __init__(self, data, period=10):
        """Initialise simple memory.

        Parameters
        ----------
        data (array-like): Data to process.
        period (int): Number of samples in the past to remember data for.

        """
        self.data = np.asarray(data)
        self.period = period

        # This will be used to record data `period` periods in the past.
        self.data = np.repeat(data[::period], period)
