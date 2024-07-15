# An implementation of ``Balanced K-Means for Clustering.'' (https://rdcu.be/cESzk)
import logging
import copy

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from munkres import Munkres
from lib_graph_partition.hungarian import Hungarian
from lib_graph_partition.hungarian_1 import KMMatcher


class ConstrainedKmeansBase:
    def __init__(self, data_feat, num_clusters, node_threshold, terminate_delta, max_iteration=20):
        self.logger = logging.getLogger('constrained_kmeans_base')

        self.data_feat = data_feat ## 嵌入
        self.num_clusters = num_clusters
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta
        self.max_iteration = max_iteration ## 最大迭代值 20

    def initialization(self):
        centroids = np.random.choice(np.arange(self.data_feat.shape[0]), self.num_clusters, replace=False)  ## 随机划分中心节点、相当于初始化
        self.centroid = dict(zip(range(self.num_clusters), self.data_feat[centroids])) ## 打包好

    def clustering(self):
        centroid = copy.deepcopy(self.centroid)
        centroid_delta = {} 
        km_base_delta = []

        for i in range(self.max_iteration):
            self.logger.info('iteration %s' % (i))
            self._node_reassignment() ## 重新分配节点
            self._centroid_updating() ## 中心节点嵌入更新

            # record the average change of centroids, if the change is smaller than a very small value, then terminate
            delta = self._centroid_delta(centroid, self.centroid) ## delta在更新，但是在不断变大
            centroid_delta[i] = delta
            km_base_delta.append(delta)
            centroid = copy.deepcopy(self.centroid)

            if delta <= self.terminate_delta: ## 中心节点进步的差距太小，就没必要再迭代了
                break
            self.logger.info("delta: %s" % delta)

        return self.clusters, km_base_delta

    def _node_reassignment(self):
        self.logger.info('Node reassignment begins')
        self.clusters = dict(
            zip(np.arange(self.num_clusters), [np.zeros(0, dtype=np.uint64) for _ in range(self.num_clusters)]))

        distance = np.zeros([self.num_clusters, self.data_feat.shape[0]])
        # cost_matrix = np.zeros([self.data_feat.shape[0], self.data_feat.shape[0]])
        for i in range(self.num_clusters):
            distance[i] = np.sum((self.data_feat - self.centroid[i]) ** 2, axis=1) ## 对嵌入与中心节点之间的距离
        cost_matrix = np.tile(distance, (self.data_feat.shape[0], 1)) ## 按需要重新塑造距离
        cost_matrix = cost_matrix[:self.data_feat.shape[0], :]

        # too slow
        # matrix = np.array(cost_matrix)
        # m = Munkres()
        # assignment = m.compute(matrix)
        # assignment = np.array(assignment)
        # assignment = assignment[:, 1]

        # hungarian = Hungarian(cost_matrix)
        # hungarian.calculate()
        # assignment = hungarian.get_results()
        # assignment = np.array(assignment)
        # assignment = assignment[np.argsort(assignment[:, 0])]
        # assignment = assignment[:, 1]

        matcher = KMMatcher(cost_matrix)
        assignment, _ = matcher.solve() ## 计算出相应的相似权重值  assignment是挑选出的编号 ，根据树结构挑选出的最大权重路径

        partition = np.zeros(self.data_feat.shape[0])
        for i in range(self.data_feat.shape[0]):
            partition[assignment[i]] = i % self.num_clusters ## 将节点根据路径划分编号

        for i in range(self.num_clusters):
            self.clusters[i] = np.where(partition == i)[0] ## 碎片聚类成功

    def _centroid_updating(self):
        self.logger.info('Updating centroid begins')
        for i in range(self.num_clusters):
            self.centroid[i] = np.mean(self.data_feat[self.clusters[i]], axis=0) ## 重新计算中心节点嵌入

    def _centroid_delta(self, centroid_pre, centroid_cur):
        delta = 0.0
        for i in range(len(centroid_cur)):
            delta += np.sum(np.abs(centroid_cur[i] - centroid_pre[i]))

        return delta


if __name__ == '__main__':
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    data_feat = np.array([[1, 2],
                          [1, 3],
                          [1, 4],
                          [1, 5],
                          [10, 2],
                          [10, 3]])
    num_clusters = 2
    node_threshold = 3
    terminate_delta = 0.001

    cluster = ConstrainedKmeansBase(data_feat, num_clusters, node_threshold, terminate_delta)
    cluster.initialization()
    cluster.clustering()
