import networkx as nx
import numpy as np
import os
import pandas as pd
import yaml

def read_edgeList(file_name, split_tag=' '):
    G = nx.Graph()
    # file = open(file_name)
    # line = file.readline().strip("\n")
    # while line:
    #     sub = line.split(split_tag)
    #     G.add_edge(int(sub[0]) - 1, int(sub[1]) - 1)  # 我们的数据节点从1开始编号，所以要-1
    #     line = file.readline().strip("\n")
    # file.close()
    edges = np.loadtxt(file_name, delimiter=split_tag, dtype=int)
    if np.min(edges) > 0:
        edges = edges - 1
    G.add_edges_from(edges)
    return G

def write_edgelist(G, file_name):
    with open(file_name, 'w') as f:
        for e in G.edges:
            f.write(str(e[0]+1)+' '+str(e[1]+1)+'\n') #写入文件的节点编号从1开始，但是python中读入nx的节点编号从0开始


def get_base_path(name, Order):
    return os.path.join('..', 'data', name, '{}-simplex'.format(Order))

# simplex_dict的存储路径
# 读文件时候要注意加上header=None，否则会把第一行看作数header
# pd.read_csv(get_simplex_dict_path(name, Order=2), header=None).values
def get_simplex_list_path(name, Order):
    return os.path.join('..', 'data', name, '{}-simplex'.format(Order), 'simplex-index.csv')

# 里面记录了各阶单纯形的数量
def get_statistic_path(name):
    return os.path.join('..', 'data', name, name + '_statistics.yaml')

def get_simplex_list(name, Order):
    if Order <= 0:
        with open(get_statistic_path(name), "r") as f:
            n0 = yaml.safe_load(f)['n_simplex'][0]
        return list(range(n0))
    else:
        return pd.read_csv(get_simplex_list_path(name, Order), header=None).values

# 生成SIR标签的路径
def get_SIR_path(name, Order, beta1, beta2):
    return os.path.join(get_base_path(name, Order), 'SIR_data', '{}_{}_{}.csv'.format(name, beta1, beta2))

# SIR_path中每一行代表一万次的均值，再对这些均值取平均
def get_SIR(name, Order, beta1, beta2):
    sir = pd.read_csv(get_SIR_path(name, Order, beta1, beta2), header=None).values
    if len(sir.shape)>1 and len(sir[1]) > 1:
        # 若是原始数据（未取均值，每行是一条记录）
        return np.mean(sir, axis=0)
    else:
        return sir