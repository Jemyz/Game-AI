import numpy as np
import numpy.random as rnd
import numpy.linalg as la
from collections import Counter
import scipy.cluster.vq as vq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_data_centers(matX, matM):
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(*matX, alpha=0.05)
    ax.scatter(*matM, marker='^', alpha=1)

    plt.show()


def average_log_likelihood(P, n, s, s0, state2index, index2state):
    lle = np.zeros(n).astype(object)

    for i in range(n):
        sseq = ''.join(generateStateSequence(s0, P, s, state2index, index2state))
        iseq = get_iseq(state2index, sseq)
        lle[i] = calc_log_likelihood(P, iseq)

    return np.mean(lle), np.var(lle)


def pr_sseq(sseq, n):
    i = 0
    for _ in range(int(np.floor(len(sseq) // n))):
        print(sseq[i:i + n])
        i += n


def get_pi_iter(P, i=0, tmax=1000):
    m = P.shape[0]
    vecPI = np.zeros(m)
    vecPI[i] = 1
    for t in range(tmax):
        vecPI = np.dot(P, vecPI)
    return vecPI


def get_pi_eign(P):
    evals, evecs = la.eig(P)
    vecPI = evecs[:, 0]
    vecPI = vecPI / np.sum(vecPI)
    return vecPI


def get_pi(P):
    m = P.shape[0]
    vecB = np.hstack((np.zeros(m), 1))
    matI = np.eye(m)
    matA = np.vstack((matI - P, np.ones(m)))
    vecPI = la.lstsq(matA, vecB)[0]
    return vecPI


def calc_log_likelihood(P, iseq):
    return np.sum([np.log(P[iseq[t], iseq[t - 1]]) for t in range(1, len(iseq))])


def calc_prob(P, iseq):
    return np.prod([P[iseq[t], iseq[t - 1]] for t in range(1, len(iseq))])


def generate_episoids(X0, P, state2index, index2state, tau=10, times=10000):
    episoids = np.zeros((times, tau)).astype(object)
    for i in range(times):
        sequence = generateStateSequence(X0, P, tau, state2index, index2state)
        episoids[i] = np.array(sequence)
    return episoids


def generateStateSequence(s0, P, n, state2index, index2state):
    sseq = [s0]
    for t in range(1, n):
        i = state2index[sseq[t - 1]]
        j = rnd.choice(range(len(P)), p=P[:, i])
        sseq.append(index2state[j])
    return sseq


def estimateStateTransitions(iseq):
    m = len(list(set(iseq)))
    P = np.zeros((m, m))
    for t in range(1, len(iseq)):
        P[iseq[t], iseq[t - 1]] += 1
    return P / np.sum(P, axis=0)


def get_iseq(state2index, sseq):
    return [state2index[s] for s in sseq]


def set_dict(states, indices):
    state2index = dict(zip(states, indices))
    index2state = dict(zip(indices, states))
    return state2index, index2state


def get_states_indices(sseq):
    states = sorted(list(set(sseq)))
    indices = list(range(len(states)))
    return states, indices


if __name__ == '__main__':

    # practical 14.1
    P1 = np.array(
        [[0.25, 0.10, 0.25],
         [0.50, 0.80, 0.50],
         [0.25, 0.10, 0.25]])

    P2 = np.array(
        [[0.30, 0.20, 0.50],
         [0.50, 0.30, 0.20],
         [0.20, 0.50, 0.30]])
    m = len(P1)
    ts = [1, 2, 4, 6, 16]

    for i in range(m):
        for t in ts:
            print(get_pi_iter(P1, i, t))
        print()

    print(get_pi_eign(P1))
    print(get_pi_eign(P2))

    # practical 14.2

    states = ['A', 'B', 'C']
    indices = range(len(states))
    state2index, index2state = set_dict(states, indices)

    num = 10000
    s = 10
    episoids1 = generate_episoids('A', P1, state2index, index2state)
    occurences = Counter(episoids1[:, -1])
    for state, occ in occurences.items():
        print(state, occ / num)
    print()
    episoids2 = generate_episoids('A', P2, state2index, index2state)
    occurences = Counter(episoids2[:, -1])
    for state, occ in occurences.items():
        print(state, occ / num)

    # practical 15.3

    matX = np.loadtxt('q3dm1-path2.csv', delimiter=',').T

    matM, inds = vq.kmeans2(matX.T, k=10, iter=100, minit='++')
    matM = matM.T

    print(np.round(estimateStateTransitions(inds), 2))

    # practical 16.1
    taus = [5, 10, 100]

    for tau in taus:
        print(tau, tau * np.log(P1[1, 1]) + np.log(1 - P1[1, 1]))

    # practical 16.2

    for tau in taus:
        print(tau, tau * np.log(P1[1, 1]) + np.log(P1[2, 1]))
