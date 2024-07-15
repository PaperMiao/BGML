import math
import pickle

import cupy as cp
import numpy as np
import logging

from sklearn.cluster import KMeans

import config
from lib_graph_partition.constrained_kmeans_base import ConstrainedKmeansBase
from lib_graph_partition.partition import Partition
from lib_graph_partition.constrained_kmeans import ConstrainedKmeans
from lib_node_embedding.node_embedding import NodeEmbedding


class PartitionKMeans(Partition):
    def __init__(self, args, graph, dataset):
        super(PartitionKMeans, self).__init__(args, graph, dataset)

        self.logger = logging.getLogger('partition_kmeans') ## 聚类划分
        cp.cuda.Device(self.args['cuda']).use()  ## 使用cupy，理清楚cuda
        self.load_embeddings() ## 定义加载嵌入

    def load_embeddings(self):
        node_embedding = NodeEmbedding(self.args, self.graph, self.dataset) ## 输入参数、训练图、以及数据集 到nodeembedding生成器

        if self.partition_method in ["sage_km", "sage_km_base"]: ## 划分是不是这两种，不是的话，这个函数页不支持
            self.node_to_embedding = node_embedding.sage_encoder() ## 就加载SAGE编码器就行
            # print(len(self.node_to_embedding))
        else:
            raise Exception('unsupported embedding method')

    def partition(self):
        self.logger.info("partitioning")  ## 正在划分

        embedding = [] ## 建一个嵌入的空集
        for node in self.node_to_embedding.keys(): 
            embedding.append(self.node_to_embedding[node]) ## 针对性的存储嵌入

        if not self.args['is_constrained']:
            cluster = KMeans(n_clusters=self.num_shards, random_state=10)  ## 根据碎片数量聚类，random_state=10，确定质心初始化的随机数生成。使用一个int，使随机性具有确定性。
            cluster_labels = cluster.fit_predict(embedding) ## 拟合嵌入和标签

            node_to_community = {} ## 节点标注社区
            for com, node in zip(cluster_labels, self.node_to_embedding.keys()):
                ## print(node) ## 某碎片社区内节点的数量
                ## print(com)  ## com是社区编号
                node_to_community[node] = com ## 把所有的节点打上社区的编号
                ## print(node_to_community) ## 编好的社区

            community_to_node = {} ## 分化好碎片，然后把属于该分区的节点都划进去
            for com in range(len(set(node_to_community.values()))):
                community_to_node[com] = np.where(np.array(list(node_to_community.values())) == com)[0]
            community_to_node = dict(sorted(community_to_node.items()))
            ## print(community_to_node)

        else:
            # node_threshold = math.ceil(self.graph.number_of_nodes() / self.num_shards)
            # node_threshold = math.ceil(self.graph.number_of_nodes() / self.num_shards + 0.05*self.graph.number_of_nodes())
            node_threshold = math.ceil(
                self.graph.number_of_nodes() / self.args['num_shards'] + self.args['shard_size_delta'] * (
                            self.graph.number_of_nodes() - self.graph.number_of_nodes() / self.args['num_shards']))
            ## 节点阈值、math.ceil函数返回一个大于或等于一个给定数字的最小整数 
            ## 含义应该是不能让碎片太小 Balance！！！！ 使用碎片的delta也是为了不让与之框死在平均数之下
            self.logger.info("#.nodes: %s. Shard threshold: %s." % (self.graph.number_of_nodes(), node_threshold))

            if self.partition_method == 'sage_km_base': ## 两种方案的区别在于使用的聚类方式不一样 轻易不用base，尤其是大数据集，慢
                cluster = ConstrainedKmeansBase(np.array(embedding), num_clusters=self.num_shards,
                                                node_threshold=node_threshold,
                                                terminate_delta=self.args['terminate_delta'])
                ## self.args['terminate_delta'] 停止率
                cluster.initialization() ## 初始化
                community, km_deltas = cluster.clustering() ## 开始聚类
                pickle.dump(km_deltas, open(config.ANALYSIS_PATH + "partition/base_bkm_" + self.args['dataset_name'], 'wb'))
                ## pickle.dump()函数将Python对象序列化为字节流，并将字节流写入文件中。

                community_to_node = {}
                for i in range(self.num_shards):
                    community_to_node[i] = np.array(community[i]) ## 对节点进行分类标注，类似if之前的

            if self.partition_method == 'sage_km': ## 和base的区别在于，用的不是树结构，采用的排序的方法
                cluster = ConstrainedKmeans(cp.array(embedding), num_clusters=self.num_shards,
                                               node_threshold=node_threshold,
                                               terminate_delta=self.args['terminate_delta'])
                cluster.initialization()
                community, km_deltas = cluster.clustering()
                pickle.dump(km_deltas, open(config.ANALYSIS_PATH + "partition/bkm_" + self.args['dataset_name'], 'wb'))
                ## pickle.dump()函数将Python对象序列化为字节流，并将字节流写入文件中。

                community_to_node = {}
                for i in range(self.num_shards):
                    community_to_node[i] = np.array(cp.asarray(community[i]).get().astype(int))
                    # print(max(community_to_node[i]))

        ## 将一级划分的各社区的节点嵌入提取出来，用于二级划分
        community_embedding = {}
        for i in range(self.num_shards):
            community_indices = community_to_node[i]
            community_embedding[i] = []
            for j in range(len(community_indices)): 
                community_embedding[i].append(np.array(embedding)[community_indices[j]]) 
            
        ## 对各大一级社区挨个划分二级社区
        second_node_threshold = {}
        community_to_node_all = {}
        for i in range(self.num_shards):
            second_node_threshold[i] = math.ceil(
                    len(community_to_node[i]) / self.args['second_num_shards'] + self.args['shard_size_delta'] * (
                                len(community_to_node[i]) - len(community_to_node[i]) / self.args['second_num_shards']))
            self.logger.info("Secondary partition #.nodes: %s. Second shard threshold: %s." % (len(community_to_node[i]), second_node_threshold[i]))
            cluster_2 = ConstrainedKmeans(cp.array(community_embedding[i]), num_clusters=self.args['second_num_shards'],
                                               node_threshold=second_node_threshold[i],
                                               terminate_delta=self.args['terminate_delta'])
            cluster_2.initialization() ## 初始化
            community_2, km_deltas_2 = cluster_2.clustering() ## 开始聚类
            pickle.dump(km_deltas_2, open(config.ANALYSIS_PATH + "partition/bkm_2_" + self.args['dataset_name'] + str(i), 'wb'))
                ## pickle.dump()函数将Python对象序列化为字节流，并将字节流写入文件中。

            community_to_node_all[i] = {}
            for j in range(self.args['second_num_shards']):
                community_to_node_all[i][j] = np.array(cp.asarray(community_2[j]).get().astype(int))

        return community_to_node, community_to_node_all

