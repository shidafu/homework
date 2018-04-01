# #############################################################################
# Generate sample data
from sklearn.datasets import make_blobs
import numpy as np

# n = 100
# C_ = [[1, 1], [-1, -1], [1, -1],[2, -2]]
# k = len(C_)
# X, L = make_blobs(n_samples = n, centers = C_,
#                   cluster_std=0.5, random_state=0)

n1 = 500
n2 = 500
n =n1 + n2
k = 3
np.random.seed(0)
sd = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n1, 2), sd),
         .7 * np.random.randn(n2, 2) + np.array([-6, 3]),
         .3 * np.random.randn(n2, 2) + np.array([0, -3])]

permut = np.random.permutation(len(X))
A = X[permut]

# #############################################################################
# Compute kmeans
import VQ
import random
import gui
import matplotlib.pyplot as plt

sofm = VQ.sofm(2,
             VQ.grid_2d,[16,16],
             VQ.dist_euclid,
             VQ.radius_mexhat,[1.0,2.0,0.1,4.0], act_para=[1.0, 0.0, 0.5])
plt.imshow(sofm.radius,cmap='gray')
# print(sofm.radius)
plt.show()
for j in range(10):
    for i in range(len(X)):
        y_list,maxid_list =sofm.predict(np.array([A[i]]))
        gui.plot_feature_map(y_list,sofm.xy,maxid_list, title="Run times:" + str(i))
        # print(np.array([X[i]]))
        sofm.update(np.array([A[i]]),1.0)
        #gui.plot_feature_cluster(A,sofm.w,y_list,title="Run times:" + str(i))
# for i in range(1000):
#     y_list,maxid_list =sofm.predict(X)
#     #gui.plot_feature_map(y_list,sofm.xy,maxid_list, title="Run times:" + str(i))
#     # print(np.array([X[i]]))
#     sofm.update(X,0.5)
#     gui.plot_feature_cluster(X,sofm.w,y_list,title="Run times:" + str(i))