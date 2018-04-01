import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
def sample_2d(num=1,xlim=[-1, 1],ylim=[-1, 1]):
    fig = plt.figure()
    ax = fig.gca()
    #设置坐标轴范围
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #设置坐标轴名称
    plt.xlabel('x1')
    plt.ylabel('x2')
    #设置坐标轴刻度
    # x_ticks = np.arange(-1, 1.2, 0.2)
    # y_ticks = np.arange(-1, 1.2, 0.2)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)
    #隐藏坐标轴
    #ax.axis('off')
    #设置坐标轴缩放
    ax.set_aspect('equal')
    ax.grid(color='b', linestyle=':', linewidth=1)
    #plt.grid()
    #取坐标点
    sample=np.array(plt.ginput(num))
    fig.clf()
    plt.close()
    return sample

def plot_2d(sample,xlim=None,ylim=None,fig=None,mark='ro'):
    sample=np.array(sample)
    if xlim is None:
        xlim = [np.amin(sample[:,0]),np.amax(sample[:,0])]
        xrang = xlim[1]-xlim[0]
        xlim = [xlim[0]-xrang/8,xlim[1]+xrang/8]
    if ylim is None:
        ylim = [np.amin(sample[:,1]),np.amax(sample[:,1])]
        yrang = ylim[1]-ylim[0]
        ylim = [ylim[0]-yrang/8,xlim[1]+yrang/8]
    needShow=False
    if fig is None:
        fig = plt.figure()
        needShow=True
    ax = fig.gca()
    #设置坐标轴范围
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #设置坐标轴名称
    plt.xlabel('x1')
    plt.ylabel('x2')
    #设置坐标轴刻度
    # x_ticks = np.arange(-1, 1.2, 0.2)
    # y_ticks = np.arange(-1, 1.2, 0.2)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)
    #隐藏坐标轴
    #ax.axis('off')
    #设置坐标轴缩放
    ax.set_aspect('equal')
    ax.grid(color='b', linestyle=':', linewidth=1)
    ax.plot(sample[:,0], sample[:,1],mark,markerfacecolor = 'white')
    if needShow:
        plt.show()
    return fig

def plot_cluster(X,W,C,title="Cluster"):
    class_num=len(W)
    plt.clf()
    for k, col in zip(range(class_num), colors): # 并行遍历
        C_k = C == [k]
        plt.plot(X[C_k, 0], X[C_k, 1], col + '.')
        plt.plot(W[k,0], W[k,1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title(title)
    #plt.show()
    plt.pause(0.0000001)

def plot_feature_map(f,xy,maxid_list,title="Feature map"):
    plt.clf()
    # fig = plt.figure()
    # ax = fig.gca()
    # plt.axes('equal')
    print(title)
    plt.title(title)
    for i in range(1):
        # plt.subplot(3,3,int(i)+1)
        plt.scatter(xy[:,0], xy[:,1],c=f[int(i),:])
        plt.scatter(xy[maxid_list[i],0], xy[maxid_list[i],1],marker='+')
    plt.pause(0.0000001)

def plot_feature_cluster(x,w,f,title="Feature cluster"):
    plt.clf()
    # fig = plt.figure()
    # ax = fig.gca()
    # plt.axes('equal')
    plt.title(title)
    plt.scatter(w[:,0], w[:,1],c=f[0,:],marker='+')
    # for i, ww in enumerate(w):
    #     plt.text(w[i,0], w[i,1], 'aaaa')
    plt.scatter(x[:,0], x[:,1],c='b',marker='.',s=1)
    # plt.show()
    plt.pause(0.0000001)

def plot_feature_cluster2(x,w,f,w1,f1,title="Feature cluster"):
    plt.clf()
    # fig = plt.figure()
    # ax = fig.gca()
    # plt.axes('equal')
    plt.title(title)
    plt.scatter(w[:,0], w[:,1],c=f[0,:],marker='+')
    # for i, ww in enumerate(w):
    #     plt.text(w[i,0], w[i,1], 'aaaa')
    plt.scatter(x[:,0], x[:,1],c='b',marker='x')
    plt.scatter(w1[:,0], w1[:,1],c=f1[0,:],marker='v')
    plt.show()
    # plt.pause(0.0000001)