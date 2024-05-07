import networkx as nx
import numpy as np
from tqdm import tqdm


def entropy_diff_time_small(G):
    Ls = np.sort(nx.laplacian_spectrum(G))
    diff_time = Ls[np.where(Ls > 10 ** -12)[0]][0]
    k_ = len(G.edges()) / len(G.nodes())
    # max_S = np.log(len(G.nodes))/100
    # N = np.log(len(G.nodes()))
    # n_ = np.log(len(G.nodes))
    beta = -np.log(.9) / diff_time
    # beta = -np.log(10/(len(G.edges())/len(G.nodes())))/diff_time
    # beta = (diff_time + k_)/2
    print('beta = ', beta)
    sp = np.exp(-beta * Ls)
    # sp = np.delete(sp,np.where(sp<10**-10))
    Z = np.sum(sp)
    p = np.exp(-beta * Ls) / Z
    p = np.delete(p, np.where(p < 10 ** -20))
    S = np.sum(-p * np.log2(p))
    return S, beta


def entropy_diff_time_mid(G):
    Ls = np.sort(nx.laplacian_spectrum(G))
    diff_time = Ls[np.where(Ls > 10 ** -12)[0]][0]
    k_ = len(G.edges()) / len(G.nodes())
    # max_S = np.log(len(G.nodes))/100
    # N = np.log(len(G.nodes()))
    # n_ = np.log(len(G.nodes))
    beta = -np.log(.33) / diff_time
    # beta = -np.log(10/(len(G.edges())/len(G.nodes())))/diff_time
    # beta = (diff_time + k_)/2
    print('beta = ', beta)
    sp = np.exp(-beta * Ls)
    # sp = np.delete(sp,np.where(sp<10**-10))
    Z = np.sum(sp)
    p = np.exp(-beta * Ls) / Z
    p = np.delete(p, np.where(p < 10 ** -20))
    S = np.sum(-p * np.log2(p))
    return S, beta


def entropy_diff_time_large(G):
    Ls = np.sort(nx.laplacian_spectrum(G))
    diff_time = Ls[np.where(Ls > 10 ** -12)[0]][0]
    k_ = len(G.edges()) / len(G.nodes())
    beta = -np.log(.01) / diff_time
    print('beta = ', beta)
    sp = np.exp(-beta * Ls)
    Z = np.sum(sp)
    p = np.exp(-beta * Ls) / Z
    p = np.delete(p, np.where(p < 10 ** -20))
    S = np.sum(-p * np.log2(p))
    return S, beta


def entropy(G, beta):
    Ls = np.sort(nx.laplacian_spectrum(G))
    Z = np.sum(np.exp(-beta * Ls))
    p = np.exp(-beta * Ls) / Z
    p = np.delete(p, np.where(p < 10 ** -20))
    # p=np.delete(p,np.where(p<10**-8))
    S = np.sum(-p * np.log2(p))
    return S


def entanglement_small(G):
    S_1, beta = entropy_diff_time_small(G)
    ent = {}
    nodes = list(G.nodes())
    # for i in pbar(range(len(nodes))):

    for i in tqdm(range(len(nodes)), position=0, leave=True):
        # for i in range(len(nodes)):
        G_i = G.copy()
        k = G_i.degree[nodes[i]]
        G_i.remove_node(nodes[i])
        S_2 = entropy(G_i, beta)
        G_star = nx.star_graph(k + 1)
        S_star = entropy(G_star, beta)
        # S_star=0
        # Z_star = 1 + (k+1-2)*np.exp(-beta)
        # S_star = beta*(k+1-2)*np.exp(-beta)/Z_star + np.log2(Z_star)

        S_2 = S_2 + S_star
        ent[nodes[i]] = S_2 - S_1
    return ent


def entanglement_mid(G):
    # pbar = ProgressBar()
    S_1, beta = entropy_diff_time_mid(G)
    ent = {}
    nodes = list(G.nodes())
    for i in tqdm(range(len(nodes))):
        # for i in range(len(nodes)):
        G_i = G.copy()
        k = G_i.degree[nodes[i]]
        G_i.remove_node(nodes[i])
        S_2 = entropy(G_i, beta)
        G_star = nx.star_graph(k + 1)
        S_star = entropy(G_star, beta)
        # S_star=0
        # Z_star = 1 + (k+1-2)*np.exp(-beta)
        # S_star = beta*(k+1-2)*np.exp(-beta)/Z_star + np.log2(Z_star)

        S_2 = S_2 + S_star
        ent[nodes[i]] = S_2 - S_1
    return ent


def entanglement_large(G):
    S_1, beta = entropy_diff_time_large(G)
    ent = {}
    nodes = list(G.nodes())
    for i in tqdm(range(len(nodes))):
        # for i in range(len(nodes)):
        G_i = G.copy()
        k = G_i.degree[nodes[i]]
        G_i.remove_node(nodes[i])
        S_2 = entropy(G_i, beta)
        G_star = nx.star_graph(k + 1)
        S_star = entropy(G_star, beta)
        # S_star=0
        # Z_star = 1 + (k+1-2)*np.exp(-beta)
        # S_star = beta*(k+1-2)*np.exp(-beta)/Z_star + np.log2(Z_star)

        S_2 = S_2 + S_star
        ent[nodes[i]] = S_2 - S_1
    return ent
