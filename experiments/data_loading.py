import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel


def numpy_load_and_symmetrize(path, **kwargs):
    data = np.loadtxt(path, **kwargs)
    if (data - data.T).mean() > 1e-6:
        raise ValueError("Matrix is too far from symmetry.")
    return (data + data.T) / 2


def read_table_to_array(path):
    return pd.read_table(path, index_col=0).values


def load_regulatory_network_features(path):
    return rbf_kernel(np.loadtxt(path, delimiter=','))

