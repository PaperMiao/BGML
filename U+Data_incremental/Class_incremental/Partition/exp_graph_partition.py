import logging
import time

import torch
from sklearn.model_selection import train_test_split 
import numpy as np
from torch_geometric.data import Data
import torch_geometric as tg
import networkx as nx

from exp import Exp
from lib_utils.utils import connected_component_subgraphs
from lib_graph_partition.graph_partition import GraphPartition
from lib_utils import utils


class ExpGraphPartition(Exp):
    def __init__(self, args):
        super(ExpGraphPartition, self).__init__(args)

        self.logger = logging.getLogger('exp_graph_partition') ## 形成图划分日志

        self.load_data() # 定义加载数据
        self.train_test_split() # 定义训练、测试划分
        self.gen_train_graph() ## 定义生成训练图？？
        self.graph_partition() # 定义图划分
        self.generate_shard_data() ## 定义生成碎片数据？
        self.generate_second_shard_data()

        ## 二次划分
        ## self.second_graph_partition()
        ## self.second_shards_data()

    def load_data(self):
        self.data = self.data_store.load_raw_data() ## 加载初始数据raw

    def train_test_split(self): # 训练、测试划分
        if self.args['is_split']:
            self.logger.info('splitting train/test data')
            self.train_indices, self.test_indices = train_test_split(np.arange((self.data.num_nodes)), test_size=self.args['test_ratio'], random_state=100)
            ## 这个划分函数是包里自带的
            ## 控制在应用拆分之前应用于数据的混洗 random_state , 根据随机参数打乱
            ## 测试集比例是自己定的，Coauthor类数据集效果会不会和这个有关？好像不是。。。。之前的训练、测试、验证的比例为0.1 0.8 0.1
            self.data_store.save_train_test_split(self.train_indices, self.test_indices) ## 将划分的索引存下来

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices)) ## 生成掩码
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))
        else:
            self.train_indices, self.test_indices = self.data_store.load_train_test_split() ## 因为数据集划分是随机的，所以可以按照之前划分的数据集来

            self.data.train_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.train_indices))
            self.data.test_mask = torch.from_numpy(np.isin(np.arange(self.data.num_nodes), self.test_indices))

    def gen_train_graph(self):
        # delete ratio of edges and update the train graph 删除边、更新训练图
        if self.args['ratio_deleted_edges'] != 0:
            self.logger.debug("Before edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
                self.data.num_nodes, self.data.num_edges)) # 处理之前，原图的节点数量以及边数量

            # self._ratio_delete_edges()
            self.data.edge_index = self._ratio_delete_edges(self.data.edge_index) ## 按比例提取出新的图所对应的训练集

        # decouple train test edges.
        edge_index = self.data.edge_index.numpy()
        test_edge_indices = np.logical_or(np.isin(edge_index[0], self.test_indices),
                                          np.isin(edge_index[1], self.test_indices)) ## 取出切割出的图里面相应的测试集、训练集
        train_edge_indices = np.logical_not(test_edge_indices)
        edge_index_train = edge_index[:, train_edge_indices] ## 刷出了测试集的边

        self.train_graph = nx.Graph() ## 创建一个空图  ## Graph with 0 nodes and 0 edges
        self.train_graph.add_nodes_from(self.train_indices) ## 先按照索引把点放进去 Graph with 16499 nodes and 0 edges


        # use largest connected graph as train graph
        if self.args['is_prune']:
            self._prune_train_set()

        # reconstruct a networkx train graph
        for u, v in np.transpose(edge_index_train):
            self.train_graph.add_edge(u, v)  ## 分解出上述的邻接矩阵 将边放进新的图里面去

        self.logger.debug("After edge deletion. train graph  #.Nodes: %f, #.Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        self.logger.debug("After edge deletion. train data  #.Nodes: %f, #.Edges: %f" % (
            self.data.num_nodes, self.data.num_edges))
        self.data_store.save_train_data(self.data)
        self.data_store.save_train_graph(self.train_graph) ## 保存训练的图和训练的数据
        # print(self.train_graph)

    def graph_partition(self):
        if self.args['is_partition']:
            self.logger.info('graph partitioning')  ## 划分开始的时候，日志

            start_time = time.time() ## 记录图划分所需要时间
            partition = GraphPartition(self.args, self.train_graph, self.data)
            self.community_to_node, self.community_to_node_all = partition.graph_partition()
            partition_time = time.time() - start_time
            self.logger.info("Partition cost %s seconds." % partition_time)
            self.data_store.save_community_data(self.community_to_node) ## 储存划分好的碎片
            self.data_store.save_second_community_data(self.community_to_node_all) ## 二级碎片
            
        else:
            self.community_to_node = self.data_store.load_community_data() ## 下次直接加载碎片就好
            self.community_to_node_all = self.data_store.load_second_community_data() ## 加载二级碎片

    def generate_shard_data(self): ## 根据划分好的碎片，生成碎片的训练，测试数据
        self.logger.info('generating shard data')

        self.shard_data = {}
        for shard in range(self.args['num_shards']):
            train_shard_indices = list(self.community_to_node[shard]) ## 划分出来的训练碎片是不超过255的
            shard_indices = np.union1d(train_shard_indices, self.test_indices) ## 取碎片训练索引和全部测试集的并集
            print(shard_indices)

            x = self.data.x[shard_indices]
            y = self.data.y[shard_indices]
            edge_index = utils.filter_edge_index_1(self.data, shard_indices)

            data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
            data.train_mask = torch.from_numpy(np.isin(shard_indices, train_shard_indices))
            data.test_mask = torch.from_numpy(np.isin(shard_indices, self.test_indices))
            print(data)

            self.shard_data[shard] = data

        self.data_store.save_shard_data(self.shard_data)

    def generate_second_shard_data(self): ## 解析二级碎片，并且存储下来
        self.logger.info('generating second shard data')

        self.second_shard_data = {}
        for first_shard in range(self.args['num_shards']):
            self.second_shard_data[first_shard] = {}
            for second_shard in range(self.args['second_num_shards']):
                second_train_shard_indices = list(self.community_to_node_all[first_shard][second_shard])
                second_shard_indices = np.union1d(second_train_shard_indices, self.test_indices) ## 取碎片训练索引和全部测试集的并集

                x_2 = self.data.x[second_shard_indices]
                y_2 = self.data.y[second_shard_indices]
                edge_index_2 = utils.filter_edge_index_1(self.data, second_shard_indices)

                data_2 = Data(x=x_2, edge_index=torch.from_numpy(edge_index_2), y=y_2)
                data_2.train_mask = torch.from_numpy(np.isin(second_shard_indices, second_train_shard_indices))
                data_2.test_mask = torch.from_numpy(np.isin(second_shard_indices, self.test_indices))
                # print(data_2)
                
                self.second_shard_data[first_shard][second_shard] = data_2

            self.data_store.save_second_shard_data(self.second_shard_data[first_shard], '_second_'+str(first_shard))

    def _prune_train_set(self):
        # extract the the maximum connected component
        self.logger.debug("Before Prune...  #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))

        self.train_graph = max(connected_component_subgraphs(self.train_graph), key=len)
        ## 找到子图上的所有连接，并提取最大的，作用应该是微调，完善子图？

        self.logger.debug("After Prune... #. of Nodes: %f, #. of Edges: %f" % (
            self.train_graph.number_of_nodes(), self.train_graph.number_of_edges()))
        # self.train_indices = np.array(self.train_graph.nodes)

    def _ratio_delete_edges(self, edge_index):
        edge_index = edge_index.numpy() ## 转numpy

        unique_indices = np.where(edge_index[0] < edge_index[1])[0] ## 统一索引 行<列 邻接矩阵的右上角 提取的行
        unique_indices_not = np.where(edge_index[0] > edge_index[1])[0] ## 邻接矩阵左下角 行
        remain_indices = np.random.choice(unique_indices,
                                           int(unique_indices.shape[0] * (1.0 - self.args['ratio_deleted_edges'])),
                                           replace=False)  ## 随机选择保留的边
        
        # print(remain_indices)
        # print(edge_index)
        # print(edge_index[0, remain_indices]) 行索引
        # print(edge_index[1, remain_indices]) 对与行相对应的列也提取。对称的
        # print(edge_index.shape[1])
        # print(edge_index[0, remain_indices] * edge_index.shape[1] * 2)

        remain_encode = edge_index[0, remain_indices] * edge_index.shape[1] * 2 + edge_index[1, remain_indices]
        unique_encode_not = edge_index[1, unique_indices_not] * edge_index.shape[1] * 2 + edge_index[0, unique_indices_not]
        sort_indices = np.argsort(unique_encode_not)
        remain_indices_not = unique_indices_not[sort_indices[np.searchsorted(unique_encode_not, remain_encode, sorter=sort_indices)]]
        remain_indices = np.union1d(remain_indices, remain_indices_not) ## 两个集合的并集，就是随便取其中一个的序列

        # self.data.edge_index = torch.from_numpy(edge_index[:, remain_indices])
        return torch.from_numpy(edge_index[:, remain_indices]) ## 得到最后的列索引，也就是边的集合，2*保留边的数量，可以转变为新的邻接矩阵
