import numpy as np
import numpy.random as rnd
from numpy import genfromtxt
from sklearn.cluster import KMeans
from SOM import som
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.cluster.vq as vq


def plot_SOM_on_data(data, traj, node_color='red', edge_color='blue', data_color=None):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(*data, color=data_color, s=1.0)
    ax.scatter(*np.array([traj[:, 0]]).T, color=node_color, s=15.0)

    for u in range(1, traj.shape[1]):
        ax.plot(*np.array([traj[:, u - 1], traj[:, u]]).T, color=node_color)

    fig.savefig('som_fittingdata.png')

    plt.show()


def est_trajectories(x_0, P_lilk, SOM_graph, actions, T=1000, eta=0.95):
    traj = np.zeros((len(x_0), T))
    traj[:, 0] = x_0
    for t in range(1, T):
        s = np.argmin([np.sum((SOM_graph.nodes[v]['w'] - traj[:, t - 1]) ** 2) for v in SOM_graph])
        a = rnd.choice(list(range(actions.shape[1])), p=P_lilk[s])
        v_1 = actions[:, a]
        v_2 = SOM_graph.nodes[s]['w'] - traj[:, t - 1]
        traj[:, t] = traj[:, t - 1] + eta * v_1 + (1 - eta) * v_2
    return traj


def calc_P(matX, SOM_graph, matA, matV):
    P = np.zeros((len(SOM_graph.nodes), matA.shape[1]))
    m, n = matX.shape
    for i in range(n - 1):
        vecX = matX[:, i]
        vecV = matV[:, i]
        s = np.argmin([np.sum((SOM_graph.nodes[v]['w'] - vecX) ** 2) for v in SOM_graph])
        a = np.argmin([np.sum((v - vecV) ** 2) for v in matA.T])
        P[s, a] += 1
    P /= P.sum()
    return P


def calc_liklihood(P):
    states = (np.sum(P, axis=1)[:, np.newaxis])
    states[states == 0] = 1
    return P / states


def create_prototypes(matV, k=9, random_state=None):
    # kmeans = KMeans(n_clusters=k,tol=1e-5,n_init=1).fit(matV.T)
    matM, inds = vq.kmeans2(matV.T, k=k, iter=1000, minit='points')

    # return kmeans.cluster_centers_.T

    return matM.T


def calc_velocities(matX):
    m, n = matX.shape
    matV = np.zeros((m, n - 1))
    for i in range(1, n):
        matV[:, i - 1] = matX[:, i] - matX[:, i - 1]
    return matV


def expectation(X):
    return np.mean(X, axis=1)


def load_data(data_path):
    return genfromtxt(data_path, delimiter=',')


if __name__ == '__main__':
    data_path = 'q3dm1-path2.csv'
    data = load_data(data_path).T
    velocities = calc_velocities(data)
    actions = create_prototypes(velocities, k=9, random_state=0)

    SOM_graph = som.glasses_graph(20)
    SOM_graph = som.initSOM(data, SOM_graph)
    SOM_graph = som.trainSOMV2(data, SOM_graph)
    states = np.vstack([SOM_graph.nodes[v]['w'] for v in SOM_graph.nodes]).T
    P = calc_P(data, SOM_graph, actions, velocities)
    P_lilk = calc_liklihood(P)

    traj = est_trajectories(data[:, 0], P_lilk, SOM_graph, actions)
    plot_SOM_on_data(data, traj)
