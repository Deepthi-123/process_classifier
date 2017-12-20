import csv
import numpy as np


def import_db():
    """
    :return: the training examples (data) and list of event labels (target) from db.csv
    :rtype: np.array, shape = (len(data), len(target))
    """
    data = []
    target = []
    with open('../dataset/db.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            data.append(np.array([eval(row["x"]), eval(row["y"]), eval(row["z"])]).T)
            d = dict()
            k = eval(row["label"])
            v = eval(row["time_stamps"])
            for i in range(len(k)):
                d[k[i]] = v[i]
            target.append(d)
    return np.array(data), np.array(target)