import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import sys
import os

sys.path.append(os.path.abspath('.'))
from hrp_engine.hrp import get_quasi_diag

# Generate random data
np.random.seed(42)
data = np.random.rand(10, 5)
corr = np.corrcoef(data)
dist = np.sqrt(2.0 * np.clip(1.0 - corr, 0.0, 2.0))

# Distance between columns
dist_cols = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        dist_cols[i, j] = np.sqrt(np.sum((dist[i] - dist[j])**2))
        
condensed = squareform(dist_cols)
Z = sch.linkage(condensed, method='single')

custom_order = get_quasi_diag(Z)
scipy_order = list(sch.leaves_list(Z))

print("Custom order:", custom_order)
print("SciPy leaves_list order:", scipy_order)
print("Matches?", custom_order == scipy_order)
