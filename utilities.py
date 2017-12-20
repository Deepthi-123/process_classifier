#-*- coding: UTF-8 -*-
import sys
from collections import defaultdict
import math
import numpy as np
from itertools import chain, combinations


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx

def powerset(iterable):
    """
    powerset([1,2,3]) --> [() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)]
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class Counter(object):
    def __init__(self, max_i, steps=100, prints=50, prefix=None):
        self.max_i = float(max_i)
        self.steps = steps
        self.a = [int(i* (self.max_i / prints)) for i in range(prints)]
        self.a.append(max_i)
        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = str(prefix)+" "*(20-len(str(prefix)))
        self.printProgress(0)

    def printProgress(self, iteration, suffix='', decimals=1):
        """
        Call in a loop to create terminal progress bar
        :param iteration: current iteration
        :type iteration: int
        :param suffix: suffix string
        :type suffix: string
        :param decimals: positive number of decimals in percent complete
        :type decimals: int
        """
        if iteration not in self.a:
            return
        formatStr = "{0:." + str(decimals) + "f}"
        percents = formatStr.format(100 * (iteration / float(self.max_i)))
        filledLength = int(round(self.steps * iteration / float(self.max_i)))
        bar = '█' * filledLength + '-' * (self.steps - filledLength)
        sys.stdout.write('\r%s |%s| %s%s %s' % (self.prefix, bar, percents, '%', suffix)),
        if iteration == self.max_i:
            sys.stdout.write('\n')
        sys.stdout.flush()