import networkx as nx
import numpy as np

def Lapacian(G):
    """
    计算网络的拉普拉斯矩阵
    :param G: networkx
    :return:
    """
    A = nx.adj_matrix(G).todense()
    return np.diag(np.array(sum(A)).flatten()) - A

def Spectral_Entropy(L,tau):
    """
    :param L:
    :param tau:
    :return: 计算谱
    """
    eigValue, eigVector = np.linalg.eig(L)
    Z_ = np.sum(np.exp(eigValue * -tau))
    t = -tau * eigValue
    S = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_)
    return S