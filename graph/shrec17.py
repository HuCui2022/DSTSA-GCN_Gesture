import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 22 
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (3, 1), (4, 3), (5, 4), (6, 5), (7, 2), (8, 7), (9, 8), (10, 9), (11, 2), (12, 11),
                     (13, 12), (14, 13), (15, 2), (16, 15), (17, 16), (18, 17), (19, 2), (20, 19), (21, 20), (22, 21),
                     (2, 2)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index] 
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.hops, self.dis_adjs = self.get_distance_graph()

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A

    def get_distance_graph(self):
        A = self.A.sum(0)  # shape: 22, 22, adjmatrix
        A[A != 0] = 1
        dis_adj = [None for _ in range(num_node)]
        dis_adj[0] = np.eye(num_node)
        dis_adj[1] = A
        hops = 0 * dis_adj[0]  # values ==0 , shape : 25 25
        for i in range(2, num_node):
            dis_adj[i] = dis_adj[i - 1] @ A.transpose(0, 1)
            dis_adj[i][dis_adj[i] != 0] = 1  # 得到第 i 阶的矩阵

        for i in range(num_node - 1, 0, -1):
            if np.any(dis_adj[i] - dis_adj[i - 1]):
                dis_adj[i] = dis_adj[i] - dis_adj[i - 1]  # 去掉重复的部分，也就是只在 i 阶出现，但是不在 i-1阶出现的值。
                hops += i * dis_adj[i]
            else:
                dis_adj[i] *= 0.0
                # 这里不该是continue
                # self.hops : 表示不同整数的矩阵。表示点和点之间的相对距离。

        # self.hops = torch.tensor(self.hops).long()  # shape : 20 20 将不同相对距离矩阵进行合并。
        # self.hops shape : 20 20 values is differnte distance

        dis_adjs = np.concatenate([np.expand_dims(A, 0) for A in dis_adj], 0)
        return hops, dis_adjs

