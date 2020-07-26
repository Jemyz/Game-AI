import numpy as np
import networkx as nx
import numpy.random as rnd
from numpy import genfromtxt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def initSOM(data, G):
    G = nx.convert_node_labels_to_integers(G)
    m, n = data.shape
    smpl = rnd.choice(n, len(G.nodes()), replace=False)
    for i, v in enumerate(G):
        G.nodes[v]['w'] = data[:, smpl[i]]
    return G


def trainSOM(matX, G, tmax=1000, sigma0=1., eta0=1.):
    m, n = matX.shape
    # compute matrix of squared path length distances between neurons
    # NOTE: networkx returns a numpy matrix, but we want a numpy array
    # because this allows for easy squaring of its entries
    matD = np.asarray(nx.floyd_warshall_numpy(G)) ** 2
    # a list of tmax random indices into the columns of matrix X
    smpl = rnd.randint(0, n, size=tmax)
    for t in range(tmax):
        # sample a point x, i.e. a column of matrix X
        vecX = matX[:, smpl[t]]
        # determine the best matching unit
        b = np.argmin([np.sum((G.nodes[v]['w'] - vecX) ** 2) for v in G])
        # update the learning rate
        eta = eta0 * (1. - t / tmax)
        # update the topological adaption rate
        sigma = sigma0 * np.exp(-t / tmax)
        # update all weights
        for i, v in enumerate(G):
            # evaluate neighborhood function
            h = np.exp(-0.5 * matD[b, i] / sigma ** 2)
            G.nodes[v]['w'] += eta * h * (vecX - G.nodes[v]['w'])
    return G


def trainSOMV2(matX, G, tmax=1000, sigma0=1., eta0=1.):
    """
    a numpythonic version of online SOM training
    """
    m, n = matX.shape
    matW = np.vstack([G.nodes[v]['w'] for v in G.nodes()]).T
    m, k = matW.shape
    matD = np.asarray(nx.floyd_warshall_numpy(G)) ** 2
    smpl = rnd.randint(0, n, size=tmax)
    for t in range(tmax):
        # NOTE: for all of the below to work, we must reshape the sampled column of X
        vecX = matX[:, smpl[t]].reshape(m, 1)
        b = np.argmin(np.sum((matW - vecX) ** 2, axis=0))
        eta = eta0 * (1. - t / tmax)
        sigma = sigma0 * np.exp(-t / tmax)
        vecH = np.exp(-0.5 * matD[b, :] / sigma ** 2)
        matW += eta * vecH * (vecX - matW)
        for i, v in enumerate(G):
            G.nodes[v]['w'] = np.ravel(matW[:, i])
    return G


def trainSOMV3(matX, G, tmax=100, sigma0=1., eta0=1.):
    """
    a version of batch SOM training
    """
    m, n = matX.shape
    # compute matrix of squared path length distances between neurons
    # NOTE: networkx returns a numpy matrix, but we want a numpy array
    # because this allows for easy squaring of its entries
    matD = np.asarray(nx.floyd_warshall_numpy(G)) ** 2
    # a list of tmax random indices into the columns of matrix X
    bs = np.zeros(n, dtype=np.int32)
    old_bs = np.zeros(n, dtype=np.int32)

    for t in range(tmax):
        # sample a point x, i.e. a column of matrix X
        for i in range(n):
            vecX = matX[:, i]
            # determine the best matching unit
            bs[i] = np.argmin([np.sum((G.nodes[v]['w'] - vecX) ** 2) for v in G])
        if all(bs == old_bs):
            return G
        old_bs = bs.copy()
        sigma = sigma0 * np.exp(-t / tmax)

        # update all weights
        for i, v in enumerate(G):
            # evaluate neighborhood function
            num = np.zeros(m)
            den = 0
            for j in range(n):
                h = np.exp(-0.5 * matD[bs[j], i] / sigma ** 2)
                num += h * matX[:, j]
                den += h

            G.nodes[v]['w'] = num / (den + 0.01)
    return G


def plot_SOM_on_data(data, SOM_graph, node_color='black', edge_color='black', data_color='blue', seed=3):
    my_pos = nx.spring_layout(SOM_graph, seed=seed)
    nx.draw(SOM_graph, pos=my_pos, with_labels=True, node_color=node_color, node_size=400, edge_color=edge_color,
            linewidths=1,
            font_size=15)
    plt.savefig('som_mapspace.png')
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(*data, color=data_color, alpha=0.5, s=2.0)

    for u, v in SOM_graph.edges():
        ax.scatter(*np.array([SOM_graph.nodes[u]['w']]).T, color=node_color, alpha=1.0, s=15.0)
        ax.scatter(*np.array([SOM_graph.nodes[v]['w']]).T, color=node_color, alpha=1.0, s=15.0)

        ax.plot(*np.array([SOM_graph.nodes[u]['w'], SOM_graph.nodes[v]['w']]).T, color=edge_color)

    fig.savefig('som_fittingdata.png')

    plt.show(block=False)


def glasses_graph(k=24):
    ring1 = nx.generators.cycle_graph(k // 2)
    ring2 = nx.generators.cycle_graph(k // 2)
    G = nx.disjoint_union(ring1, ring2)
    G.add_edge(0, k - 1)
    return G


def two_rings_graph(k=24):
    ring1 = nx.generators.classic.cycle_graph(k // 2)
    ring2 = nx.generators.cycle_graph(k // 2)

    G = nx.disjoint_union(ring1, ring2)
    for i in range(k // 2):
        G.add_edge(i, i + k // 2)

    return G


def wheel_graph(k=24):
    return nx.generators.circular_ladder_graph(k)


def mesh_graph(m, n):
    return nx.generators.grid_2d_graph(m, n)


def mse(matX, G):
    result = 0
    m, n = matX.shape
    for i in range(n):
        vecX = matX[:, i]
        result += np.min([np.sum((G.nodes[v]['w'] - vecX) ** 2) for v in G])
    return result / n


def load_data(data_path):
    return genfromtxt(data_path, delimiter=',')


if __name__ == '__main__':
    SOM_graph = nx.generators.classic.cycle_graph(12)
    SOM_graph = glasses_graph()
    # SOM_graph = two_rings_graph()
    # SOM_graph = wheel_graph()
    # SOM_graph = mesh_graph(5, 5)

    data_path = 'q3dm1-path2.csv'
    data = load_data(data_path).T
    SOM_graph = initSOM(data, SOM_graph)
    SOM_graph = trainSOMV2(data, SOM_graph)
    print(mse(data, SOM_graph))
    plot_SOM_on_data(data, SOM_graph)
    plt.show()
