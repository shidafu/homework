# #############################################################################
# Generate sample data
from sklearn.datasets import make_blobs
import numpy as np

n1 = 500
n2 = 500
n =n1 + n2
k = 3
np.random.seed(0)
sd = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n1, 2), sd),
         .7 * np.random.randn(n2, 2) + np.array([-6, 3]),
         .3 * np.random.randn(n2, 2) + np.array([0, -3])]

# #############################################################################
# Compute kmeans
import cluster
import random
import gui
import matplotlib.pyplot as plt

A_id = random.sample(range(n), k)
A = X[A_id]
GMM = cluster.GMM(X,A)

for i in range(100):

    C, A, isend, up_dist = GMM.update(X)
    gui.plot_cluster(X, A, C, title="Run times:" + str(i))
    if isend:
        print("Cluster convergence")
        break
    else:
        print("Cluster unconvergence:"+ str(up_dist))