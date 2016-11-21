import pandas as pd
import numpy as np


def csv_reader(filepath, label_index=1):
    # read data into numpy array
    data = pd.read_csv(filepath)
    data.replace('?', -99999, inplace=True)
    data = np.array(data.as_matrix())
    data = np.delete(data, 2, axis=1)
    data = np.delete(data, 0, axis=1)
    data = data.astype(float)

    labels = np.empty(len(data))
    for i, row in enumerate(data):
        labels[i] = row[label_index]

    data = np.delete(data, label_index, axis=1)

    return labels, data
