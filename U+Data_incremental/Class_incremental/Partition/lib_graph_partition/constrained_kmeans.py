import logging
import copy

from tqdm import tqdm

import numpy as np
import cupy as cp


class ConstrainedKmeans:
    def __init__(self, data_feat, num_clusters, node_threshold, terminate_delta, max_iteration=20):
        self.logger = logging.getLogger('constrained_kmeans')

        self.data_feat = data_feat
        self.num_clusters = num_clusters
        self.node_threshold = node_threshold
        self.terminate_delta = terminate_delta
        self.max_iteration = max_iteration

    def initialization(self):
        centroids = np.random.choice(np.arange(self.data_feat.shape[0]), self.num_clusters, replace=False)
        #print(centroids)
        self.centroid = {}
        for i in range(self.num_clusters):
            self.centroid[i] = self.data_feat[centroids[i]]
        

    def clustering(self):
        centroid = copy.deepcopy(self.centroid)
        km_delta = []

        pbar = tqdm(total=self.max_iteration)
        pbar.set_description('Clustering')

        for i in tqdm(range(self.max_iteration)):
            self.logger.info('iteration %s' % (i,))

            self._node_reassignment()
            self._centroid_updating()

            # record the average change of centroids, if the change is smaller than a very small value, then terminate
            delta = self._centroid_delta(centroid, self.centroid)
            km_delta.append(delta)
            centroid = copy.deepcopy(self.centroid)

            if delta <= self.terminate_delta:
                break
            self.logger.info("delta: %s" % delta)
        pbar.close()
        return self.clusters, km_delta

    def _node_reassignment(self): ## 其他部分都一样，这一部分是和base不同的地方
        self.clusters = {}
        for i in range(self.num_clusters):
            self.clusters[i] = np.zeros(0, dtype=np.uint64)

        distance = np.zeros([self.num_clusters, self.data_feat.shape[0]])

        for i in range(self.num_clusters):
            distance[i] = np.sum(np.power((self.data_feat.get() - self.centroid[i].get()), 2), axis=1)

        sort_indices = np.unravel_index(np.argsort(distance, axis=None), distance.shape)
        ## 这里开始不同，不是匹配，而是对距离开始排序
        clusters = sort_indices[0] ## 节点所属的碎片数目
        #print(clusters)
        users = sort_indices[1] ## 节点原来的序号
        #print(users)
        selected_nodes = np.zeros(0, dtype=np.int64)
        counter = 0

        while len(selected_nodes) < self.data_feat.shape[0]:
            cluster = int(clusters[counter]) ## 按节点一个一个分，从第一个节点开始
            user = users[counter] ## 第一个用户所对应的原节点编号
            if self.clusters[cluster].size < self.node_threshold:
                self.clusters[cluster] = np.append(self.clusters[cluster], cp.asnumpy(int(user)))
                #print(self.clusters[cluster])
                selected_nodes = np.append(selected_nodes, cp.asnumpy(int(user)))
                ## 上述将节点分门别类的挨个加入对应碎片

                # delete all the following pairs for the selected user
                user_indices = np.where(users == user)[0] ## 节点分过，就不能再用了，排除之后的所有进入其他碎片的可能
                a = np.arange(users.size)
                b = user_indices[user_indices > counter] ## 当用户的索引大于计数器时
                remain_indices = a[np.where(np.logical_not(np.isin(a, b)))[0]] ## 是的变False，也就是说是的不留
                clusters = clusters[remain_indices]
                users = users[remain_indices]

            counter += 1

    def _centroid_updating(self):
        for i in range(self.num_clusters):
            self.centroid[i] = np.mean(self.data_feat[self.clusters[i].astype(int)], axis=0)

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

    cluster = ConstrainedKmeans(data_feat, num_clusters, node_threshold, terminate_delta)
    cluster.initialization()
    cluster.clustering()