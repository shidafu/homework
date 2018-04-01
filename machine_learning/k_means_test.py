# #############################################################################
# Generate sample data
from sklearn.datasets import make_blobs
import numpy as np

n = 1000
C_ = [[1, 1], [-1, -1], [1, -1],[2, -2]]
k = len(C_)
X, L = make_blobs(n_samples = n, centers = C_,
                  cluster_std=0.5, random_state=0)

# #############################################################################
# Compute kmeans
import cluster
import random
import gui
import matplotlib.pyplot as plt

A_id = random.sample(range(n), k)
A = X[A_id]
k_means = cluster.k_means(A)

for i in range(100):
    C = k_means.predict(X)
    gui.plot_cluster(X, A, C, title="Run times:" + str(i))
    C, A, isend, up_dist = k_means.update(X, c_list = C)
    if isend:
        print("Cluster convergence")
        break
    else:
        print("Cluster unconvergence:"+ str(up_dist))