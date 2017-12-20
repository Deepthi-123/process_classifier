from shapelet_utils import distance_matrix3D
import numpy as np

class Clustering(object):
    def __init__(self, d_max):
        """
        :param d_max: cluster radius
        :type d_max: int
        """
        self.d = d_max
        self.distance_matrix = distance_matrix3D

    def fit(self, S):
        """
        :param S: list of shapelet candidates of the same length
        :type S: np.array, shape = (len(S), len(s), len(dim(s)))
        :return: list containing the cluster id for every s in S
        :rtype: np.array, shape = (len(S),)
        """
        self.S = S
        self.centers = None
        outs = S
        while outs.shape[0] > 0:
            centroid = outs[0]
            d_m = self.distance_matrix(outs, centroid[None,...])
            ins = outs[(d_m < self.d).T[0]]
            centroid = ins.mean(axis=0)
            try:
                self.centers = np.concatenate((self.centers, centroid[None, ...]))
            except ValueError:
                self.centers = np.array([centroid])

            outs = self.not_in_cluster(outs)

        self.labels_ = self.predict_all(S)
        return self.labels_

    def not_in_cluster(self, old_nic):
        """
        :param old_nic: shapelets that did not belong to a cluster in the previous step
        :type old_nic: np.array
        :return: all shapelets that do not belong to a cluster
        :rtype: np.array
        """
        return old_nic[self.distance_matrix(old_nic, self.centers).min(axis=1) >= self.d]

    def nn_centers(self):
        """
        :return: the 1-nearest neighbor for every cluster center
        :rtype: np.array, shape = (len(self.clusters),)
        """
        medians = []
        d_m = self.distance_matrix(self.S, self.centers)
        for l in range(self.centers.shape[0]):
            medians.append(self.S[d_m[:, l].argmin()])
        return np.array(medians)

    def predict_all(self, S):
        """
        :param S: list of shapelets
        :type S: np.array, shape = (len(S), len(s), len(dim(s)))
        :return: id of the closest cluster for each s in S
        :rtype: np.array, shape = (len(S))
        """
        return self.distance_matrix(S, self.centers).argmin(axis=1)