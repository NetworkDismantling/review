import argparse
import copy
import os

from tqdm.auto import tqdm

from network_dismantling.vertex_entanglement.reinsertion.reinsertion import reinsertion, get_gcc
from network_dismantling.vertex_entanglement.utils import fileUtils
from network_dismantling.vertex_entanglement.utils.graphUtils import *


def generate_Belta(eigValue):
    # eigValue 从小到大排序后的特征值
    gap = 100
    a = list(np.linspace(eigValue[0], 5 / eigValue[1], gap))
    b = list(np.linspace(a[-1], 10 / eigValue[1], gap))
    return a[0:-1] + b


def new_generate_beta(lambda2):
    # you could set a small gap to aacclerate computing!
    gap = 100
    a = list(np.linspace(0, 5 / lambda2, gap))
    # a = list(np.linspace(0, 1 / lambda2, gap*50))
    b = list(np.linspace(a[-1], 10 / lambda2, gap))
    return a[0:-1] + b


def VertexEnt(G, belta=None, perturb_strategy='default', printLog=False):
    """
    近似计算的方法计算节点纠缠度
    :param G:
    :return:
    """
    # 邻接矩阵
    # nodelist: list, optional:
    # The rows and columns are ordered according to the nodes in nodelist.
    # If nodelist is None, then the ordering is produced by G.nodes().
    A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
    assert 0 in G, "Node 0 should be in the input graph!"
    assert np.allclose(A, A.T), "adjacency matrix should be symmetric"
    # 拉普拉斯矩阵
    L = np.diag(np.array(sum(A)).flatten()) - A
    N = G.number_of_nodes()

    eigValue, eigVector = np.linalg.eigh(L)
    print("Finish calucating eigen values!")
    eigValue = eigValue.real
    eigVector = eigVector.real  # 每个特征向量一列

    # sort_idx = np.argsort(eigValue)
    # eigValue=eigValue[sort_idx]
    # eigVector=eigVector[:,sort_idx]

    if printLog:
        print(eigValue)

    if belta is None:
        # belta = generate_Belta(eigValue)
        num_components = nx.number_connected_components(G)
        print(f"Tere are {num_components} components.")
        belta = new_generate_beta(eigValue[num_components])

    # %%
    S = np.zeros(len(belta))
    for i in range(0, len(belta)):
        b = belta[i]
        Z_ = np.sum(np.exp(eigValue * -b))
        t = -b * eigValue
        S[i] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_)

    print("Finish calucating spectral entropy!")
    if printLog:
        print(S)

    # %%
    lambda_ral = np.zeros((N, N))
    if perturb_strategy == 'default':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues", unit="node"):
            # dA=np.zeros((N,N))
            neibour = list(G.neighbors(v_id))
            kx = G.degree(v_id)
            A_loc = A[neibour][:, neibour]
            N_loc = kx + np.sum(A_loc) / 2
            weight = 2 * N_loc / kx / (kx + 1)
            if weight == 1:
                lambda_ral[v_id] = eigValue
            else:
                neibour.append(v_id)
                neibour = sorted(neibour)
                dA = weight - A[neibour][:, neibour]
                dA = dA - np.diag([weight] * (kx + 1))
                dL = np.diag(np.array(sum(dA)).flatten()) - dA
                for j in range(0, N):
                    t__ = eigVector[neibour, j].T @ dL @ eigVector[neibour, j]
                    if isinstance(t__, float):
                        lambda_ral[v_id, j] = eigValue[j] + t__
                    else:
                        lambda_ral[v_id, j] = eigValue[j] + t__[0, 0]
    elif perturb_strategy == 'remove':
        for v_id in tqdm(range(0, N), desc="Computing eigenValues for removed networks", unit="node", position=0,
                         leave=True):
            neibour = list(G.neighbors(v_id))
            pt_A = copy.deepcopy(A)
            pt_A[v_id, :] = 0
            pt_A[:, v_id] = 0
            dA = pt_A - A
            dA = dA[neibour][:, neibour]
            dL = np.diag(np.array(sum(dA)).flatten()) - dA
            for j in range(0, N):
                lambda_ral[v_id, j] = eigValue[j] + (eigVector[neibour, j].T @ dL @ eigVector[neibour, j])[0, 0]

    # %%
    E = np.zeros((len(belta), N))
    for x in tqdm(range(0, N), desc="Searching minium entanglement", unit="node"):
        xl_ = lambda_ral[x, :]
        for i in range(0, len(belta)):
            b = belta[i]
            Z_ = np.sum(np.exp(-b * xl_))
            t = -b * xl_
            E[i, x] = -sum(np.exp(t) * (t / np.log(2) - np.log2(Z_)) / Z_) - S[i]

    VE = np.min(E, axis=0)
    mean_tau = np.mean(np.array(belta)[np.argmin(E, axis=0)])
    print(f"VE mean_tau={mean_tau}")
    return VE


def get_ve_nodeList(graph: nx, VE: np, dismantling_threshold=0.01):
    """
    This function is more appropriate when there are many identical VE values in the network, otherwise it is recommended to use the get_ve_nodeList_quick function
    当网络中有较多相同的VE值时用该函数更合适，否则建议使用 get_ve_nodeList_quick 函数
    :param graph: Note that nodes should be numbered starting from 0 #注意graph中要求节点从0开始编号
    :param VE:
    :param dismantling_threshold:
    :return:
    """
    assert graph.has_node(0), "Nodes should be numbered starting from 0!"
    G = copy.deepcopy(graph)
    target_size = int(dismantling_threshold * G.number_of_nodes())
    if target_size <= 2:
        target_size = 2

    # Sort VE from small to large and obtain the sorted index list
    delque = np.argsort(VE)  # 对VE从小到大进行排序，并获取排序后的索引列表

    # 用于存储相同元素的索引
    # Index used to store identical elements
    index_lists = []
    # 遍历排序后的索引列表
    # Traverse the sorted index list
    for i, index in enumerate(delque):
        if i == 0 or VE[index] != VE[delque[i - 1]]:
            # 若当前元素与前一个元素不相同，则创建新的索引列表
            # If the current element is different from the previous element, create a new index list
            index_lists.append([index])
        else:
            # 若当前元素与前一个元素相同，则将索引添加到当前列表中
            # If the current element is the same as the previous element, add the index to the current list
            index_lists[-1].append(index)

    # 最终的移除顺序
    # Final removal order
    remove_list = []
    gcc_list = []
    for same_VE_list in index_lists:
        while len(same_VE_list) > 0:
            max_index = np.argmax([G.degree[v] for v in same_VE_list])
            remove_node = same_VE_list[max_index]

            G.remove_node(remove_node)

            temp_gcc = get_gcc(G)
            if temp_gcc <= target_size:
                break

            # +1是因为 G中节点从0开始，而输出结果节点从1开始
            # +1 is because the nodes in G start from 0, and the output result nodes start from 1
            remove_list.append(remove_node + 1)
            gcc_list.append(temp_gcc)

            del same_VE_list[max_index]

    return remove_list, gcc_list


def get_ve_nodeList_quick(graph: nx, VE: np, dismantling_threshold=0.01):
    """ quick version """
    assert graph.has_node(0), "Nodes should be numbered starting from 0!"
    G = copy.deepcopy(graph)
    target_size = int(dismantling_threshold * G.number_of_nodes())
    if target_size <= 2:
        target_size = 2

    ds = np.array([G.degree(v) for v in range(G.number_of_nodes())])
    # 对VE从小到大进行排序，并获取排序后的索引列表
    # Sort VE from small to large and obtain the sorted index list
    delque = np.argsort(VE - ds / 100000)

    # 最终的移除顺序
    # Final removal order
    remove_list = []
    gcc_list = []

    for v in tqdm(delque, desc="Computing gcc", unit="node"):
        G.remove_node(v)
        temp_gcc = get_gcc(G)
        if temp_gcc <= target_size:
            break
        remove_list.append(v + 1)
        gcc_list.append(temp_gcc)

    return remove_list, gcc_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='crime', help='graph name')
    parser.add_argument('--dth', default=0.01, help='dismantling_threshold')
    parser.add_argument('--sort_strategy', default='default', choices=['default', 'quick'])  # 从VE值获取节点序列的策略
    parser.add_argument('--perturb_strategy', default='default', choices=['default', 'remove'])  # 网络扰动策略
    parser.add_argument('--belta', default=None)
    args = parser.parse_args()

    assert args.dth < 1.0 and args.dth > 0.0, 'Dismantling threshold be be less than 1.0!'
    return args


if __name__ == '__main__':

    args = parse_args()
    # args.perturb_strategy = 'remove'
    # netname = args.net

    for netname in [
        'Crime']:  # 'collaboration','grid','GrQC','facebook','figeys','ham','hep'，'hepth','RU','ES','lastfm','FR','DE','ENGB',
        path = os.path.join('..', 'results', netname)
        edge_path = os.path.join('..', 'data', netname, 'edges.txt')
        os.makedirs(path, exist_ok=True)

        G = fileUtils.read_edgeList(edge_path)
        N = G.number_of_nodes()
        print("===================================================")
        print(f"There are {N} nodes and {G.number_of_edges()} edges in {netname}")

        if args.perturb_strategy == 'default':
            ve_value_path = os.path.join(path, 'VE_value.txt')
        elif args.perturb_strategy == 'remove':
            ve_value_path = os.path.join(path, 'VE_remove_value.txt')

        if not os.path.exists(ve_value_path):
            VE = VertexEnt(G, belta=args.belta, perturb_strategy=args.perturb_strategy)
            # np.savetxt(ve_value_path, VE, fmt='%.16f')
        else:
            VE = np.loadtxt(ve_value_path).tolist()

        if args.sort_strategy == 'default':
            remove_list, gcc_list = get_ve_nodeList(G, VE, args.dth)
        elif args.sort_strategy == 'quick':
            remove_list, gcc_list = get_ve_nodeList_quick(G, VE, args.dth)

        if args.perturb_strategy == 'default':
            np.savetxt(os.path.join(path, 'VE_nodelist.txt'), remove_list, fmt='%d')
            np.savetxt(os.path.join(path, 'VE_gcc.txt'), gcc_list, fmt='%d')
            print(f"Number of attacked nodes:{len(gcc_list)}, R:{np.sum(gcc_list) / N / N}")

            edges = np.loadtxt(edge_path, delimiter=' ', dtype=int)  # 边列表
            graph = nx.Graph()
            graph.add_edges_from(edges)
            reinsertion(graph, remove_list, args.dth, metric='VER', rel_path=path)
        elif args.perturb_strategy == 'remove':
            np.savetxt(os.path.join(path, 'VE_remove_nodelist.txt'), remove_list, fmt='%d')
            np.savetxt(os.path.join(path, 'VE_remove_gcc.txt'), gcc_list, fmt='%d')
            print(f"(remove) Number of attacked nodes:{len(gcc_list)}, R:{np.sum(gcc_list) / N / N}")

            edges = np.loadtxt(edge_path, delimiter=' ', dtype=int)  # 边列表
            graph = nx.Graph()
            graph.add_edges_from(edges)
            reinsertion(graph, remove_list, args.dth, metric='VER_remove', rel_path=path)
