from numpy.core.function_base import linspace
from scipy.stats.kde import gaussian_kde
import numpy as np
from shapelet_utils import dist_shapelet_ts, \
    information_gain
from scipy.signal._peak_finding import argrelmax, argrelmin


class ShapeletClassifier(object):
    def __init__(self, s, dim_s=(0,), p=.5, density=150):
        """
        :param s: shapelet used for classification
        :type s: np.array, shape = (len(s), len(dim(s)))
        :param dim_s: list of the shapelets dimension ids
        :type dim_s: list(int)
        :param p: percentage at which delta will be selected
        :type p: int
        :param density: sampling rate for kde in points per 1 bmd
        :type density: int
        """
        self.shapelet = s
        self.dim_s = dim_s
        self.p = p
        self.resolution = density

    def fit_precomputed(self, D_new, target):
        """
        Uses a precomputed D_new to train the classifier.
        :param D_new: best match distance of the shapelet to the time series of the dataset
        :type D_new: np.array, shape = (len(dataset),)
        :param target: list of 0 (event has NOT occurred during in this time series) or 1 if it has
        :type target: np.array, shape = (len(dataset),)
        :return: information gain, delta (classification threshold), f_c(delta) the secondary quality measure
        :rtype: tuple(float, float, float)
        """
        self.D_new = D_new
        self.information_gain, self.delta, self.f_c_delta = self.train(D_new, target)
        return self.information_gain, self.delta, self.f_c_delta

    def predict_all(self, t):
        """
        Detects and returns the time of all events occurrences in t.
        :param t: multidimensional time series
        :type t: np.array, shape = (len(t), len(dim(t)))
        :return: list of indices in the middle of the shapelet matches
        :rtype: np.array
        """
        ds = dist_shapelet_ts(self.shapelet, t, self.dim_s)
        ds[ds >= self.delta] = self.delta
        mins = argrelmin(ds, order=self.shapelet.shape[0] // 2)[0]
        return np.asarray(mins)

    def train(self, data, target):
        """
        Uses KDE to estimate the best classification threshold (delta).
        :param data: shapelet transformed dataset
        :type data: np.array, shape = (len(dataset),)
        :param target: list of 0 (event has NOT occurred during in this time series) or 1 if it has
        :type target: np.array, shape = (len(dataset),)
        :return: information gain, delta, f_c(delta)
        :rtype: tuple(float, float, float)
        """
        nin_class = data[target == 0]
        np_c = len(nin_class) / (len(target) + .0)
        in_class = data[target == 1]
        p_c = len(in_class) / (len(target) + .0)
        density = int(self.resolution * (max(data) - min(data)))
        dist_space = linspace(min(data), max(data), density)
        f_c = gaussian_kde(in_class)(dist_space)
        nf_c = gaussian_kde(nin_class)(dist_space)
        P_c = p_c * f_c / (np_c * nf_c + p_c * f_c)
        P_c[P_c > self.p] = -P_c[P_c > self.p]
        delta_candidates = argrelmax(P_c, order=3)[0]
        bsf_information_gain = 0
        bsf_delta = 0
        bsf_f_c_delta = 0
        for i in delta_candidates:
            i = [i]
            delta = dist_space[i][0]
            d1 = target[data < delta]
            d2 = target[data >= delta]
            igain = information_gain(target, d1, d2)
            f_c_delta = nf_c[i][0]
            if igain > bsf_information_gain:
                bsf_information_gain = igain
                bsf_delta = delta
                bsf_f_c_delta = f_c_delta
            elif igain == bsf_information_gain and f_c_delta < bsf_f_c_delta:
                bsf_information_gain = igain
                bsf_delta = delta
                bsf_f_c_delta = f_c_delta
        return bsf_information_gain, bsf_delta, bsf_f_c_delta
