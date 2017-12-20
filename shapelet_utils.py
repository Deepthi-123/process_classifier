import numpy as np
from numpy.lib.stride_tricks import as_strided

import scipy
from scipy.spatial.distance import cityblock, euclidean

sigma_min = .3
ord = 1
metric = "cityblock"


def distances(s, t_subs, ord=ord):
    """
    Calculates the distances between a shapelet and a list of time series subequences.
    :param s: shapelet
    :type s: np.array, shape = (len(s), len(dim(s)))
    :param t_subs: list of subsequences from a time series t
    :type t_subs: np.array, shape = (len(t_subs), len(s), len(dim(s)))
    :param ord: 1 for manhattan, 2 for euclidean distance
    :type ord: int
    :return: list of distance between the shapelet and all subsequences for all dimensions
    :rtype: np.array, shape = (len(t), len(dim(s)))
    """
    return (np.linalg.norm((s - t_subs), ord=ord, axis=1)) / s.shape[0]


def distance_matrix(subs1, subs2=None, metric=metric):
    """
    :param subs1: list of l1 many subsequence of length len(s)
    :type s: np.array, shape = (l1, len(s))
    :param subs2: list of l2 many subsequence of length len(s)
    :type s: np.array, shape = (l2, len(s))
    :param metric: name of the distance metric
    :type metric: string, "cityblock" or "euclidean"
    :return: distance matrix for the subsequences on X an Y
    :rtype: np.array, shape = (l1, l2)
    """
    if subs2 is None:
        subs2 = subs1
    return scipy.spatial.distance.cdist(subs1, subs2, metric=metric) / subs1[0].shape[0]


def distance_matrix3D(subs1, subs2, metric=metric):
    """
    Calls 'distance_matrix' on every dimension and returns the average.
    :param subs1: list of l1 many multidimensional subsequences of length len(s)
    :type subs1: np.array, shape = (l1, len(s), len(dim(s)))
    :param subs2: list of l2 many multidimensional subsequences of length len(s)
    :type subs2: np.array, shape = (l2, len(s), len(dim(s)))
    :param metric: name of the distance metric
    :type metric: string, "cityblock" or "euclidean"
    :return: distance matrix between every pair of subsequences
    :rtype: np.array, shape = (l1, l2)
    """
    d_m = np.zeros((subs1.shape[0], subs2.shape[0]))
    for axis in range(subs1.shape[-1]):
        d_m += distance_matrix(subs1[:, :, axis], subs2[:, :, axis], metric)
    return d_m / subs1.shape[-1]


def dist_shapelet_ts(s, t, dim_s):
    """
    :param s: a shapelet
    :type s: np.array, shape = (len(s), len(dim(s)))
    :param t: time series with length len(t) and at least d many dimensions
    :type t: np.array, shape = (len(t), len(dim(t))) with len(dim(s)) <= len(dim(t))
    :param dim_s: dim(s), ids of the shapelets dimensions
    :type dim_s: np.array, shape = (len(dim(s)),) with dim_s \in dim(t)
    :return: distances between the shapelet and all subsequences in t
    :rtype: np.array, shape = (len(t),)
    """
    subs = subsequences(t, s.shape[0])  # (len(x), len(shapelet), axis)
    subs = z_normalize(subs)  # (len(x), len(shapelet), axis)
    return distances(s, subs[:, :, dim_s]).mean(axis=1)  # (len(x),)


def z_normalize(t):
    """
    :param t: list of time series subsequences
    :type t: np.array, shape = (len(t), len(s), len(dim(s)))
    :return: list of z-normalized time series subsequences
    :rtype: np.array, shape = (len(t), len(s), len(dim(s)))
    """
    std = np.std(t, axis=1)
    if isinstance(std, float):
        if std < sigma_min:
            std = 1
    else:
        std[std < sigma_min] = 1.
    tmp_ts = ((t.swapaxes(0, 1) - np.mean(t, axis=1)) / std).swapaxes(0, 1)
    return tmp_ts


def subsequences(t, len_s):
    """
    :param t: multidimensional time series
    :type t: np.array, shape = (len(t), len(dim(t)))
    :param len_s: len(s), desired subsequence length
    :type len_s : int
    :return: list of all len_s long subsequences from t
    :rtype: np.array, shape = (len(t), len(s), len(dim(t)))
    """
    if t.ndim == 1:
        m = 1 + t.size - len_s
        s = t.itemsize
        shapelets = as_strided(np.copy(t), shape=(m, len_s), strides=(s, s))
        return shapelets
    else:
        shapelets = None
        for i in range(t.shape[1]):
            next_dim = subsequences(t[:, i], len_s)[..., None]
            if shapelets is None:
                shapelets = next_dim
            else:
                shapelets = np.concatenate((shapelets, next_dim), axis=2)
        return shapelets


def information_gain(before_split, after_split1, after_split2):
    """
    :param before_split: dataset before split
    :type before_split: np.array, len(before_split) = len(after_split1) + len(after_split2)
    :param after_split1: dataset1 after split
    :type after_split1: np.array
    :param after_split2: dataset2 after split
    :type after_split2: np.array
    :return: information gain
    :rtype: float
    """
    f_d1 = after_split1.shape[0] / float(before_split.shape[0])
    f_d2 = after_split2.shape[0] / float(before_split.shape[0])
    gain = entropy(before_split) - (f_d1 * entropy(after_split1) + f_d2 * entropy(after_split2))
    return gain


def entropy(target):
    """
    :param target: list of class labels
    :type target: np.array
    :return: entropy of 'target'
    :rtype: float
    """
    try:
        counts = np.bincount(target)
        probs = counts[np.nonzero(counts)] / float(len(target))
        return - np.sum(probs * np.log2(probs))
    except:
        return 0
