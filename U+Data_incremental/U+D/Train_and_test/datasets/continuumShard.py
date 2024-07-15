import os
import tqdm
import dgl
import torch
import pickle
import logging
import random
import os.path
import numpy as np
from dgl import DGLGraph
from dgl.data import citegrh
import scipy.sparse as sp
from itertools  import compress
from torchvision.datasets import VisionDataset
from ogb.nodeproppred import NodePropPredDataset
import torch_geometric.utils

class ContinuumShard(VisionDataset):
    def __init__(self, root='/home/mjx/课题1代码/GML-master/U+Data_incremental/Unlearning/Partition/temp_data/processed_data/cora', name='shard_data', data_type='train', k_hop=1, download=True, task_type=0, thres_nodes = 50, first_shard = None, second_shard = None, Unlearning = True):
        super(ContinuumShard, self).__init__(root)
        self.name = name
        self.thres_nodes = thres_nodes
        self.k_hop = k_hop
        if name == 'shard_data':
            self.again = 0
        else:
            self.again = 1
        self.download(first_shard, second_shard)
        print(self.data)

        if data_type == 'train': # data incremental; use test and train as train
            if Unlearning == True:
                Unlearning_node_indices, self.data.train_mask = self.delete_random_true_shorten_tensor(self.data.train_mask)
                self.features = torch.FloatTensor(self.data.x.cpu())
                for index in sorted(Unlearning_node_indices, reverse=True):
                    self.features = torch.cat([self.features[:index], self.features[index+1:]])
                self.data.x = self.features 
                self.ids = torch.LongTensor(list(range(self.features.size(0))))
                self.data.edge_index = self.shorten_edge_index(self.data.edge_index.cpu(), Unlearning_node_indices.cpu())
                self.edge_index, _ = torch_geometric.utils.add_self_loops(self.data.edge_index)
                self.src, self.dst = self.edge_index  ## 图结构解析完毕
                for index in sorted(Unlearning_node_indices, reverse=True):
                    self.data.y = torch.cat([torch.LongTensor(self.data.y.cpu())[:index], torch.LongTensor(self.data.y.cpu())[index+1:]])
                    self.labels = self.data.y## 标签解析完毕    11.21.17：20
                for index in sorted(Unlearning_node_indices, reverse=True):
                    self.data.test_mask = torch.cat([self.data.test_mask[:index], self.data.test_mask[index+1:]])   
                print(self.data) 
                self.new_data_save(first_shard, second_shard)       
                # pickle.dump(self.data, open('/home/mjx/GML-master/U+Data_incremental/Unlearning/Train_shards/data_U/After_Unlearning', 'wb'))
                self.mask = self.data.train_mask

            else:
                self.features = torch.FloatTensor(self.data.x.cpu())
                self.ids = torch.LongTensor(list(range(self.features.size(0))))
                self.edge_index, _ = torch_geometric.utils.add_self_loops(self.data.edge_index)
                self.src, self.dst = self.edge_index  ## 图结构解析完毕
                self.labels = torch.LongTensor(self.data.y.cpu()) ## 标签解析完毕    11.21.17：20
                self.mask = self.data.train_mask
        elif data_type == 'test':
            if Unlearning == True:
                self.download(first_shard, second_shard)
                # self.data = pickle.load(open('/home/mjx/GML-master/U+Data_incremental/Unlearning/Train_shards/data_U/After_Unlearning', 'rb'))
                self.features = torch.FloatTensor(self.data.x.cpu())
                self.ids = torch.LongTensor(list(range(self.features.size(0))))
                self.edge_index, _ = torch_geometric.utils.add_self_loops(self.data.edge_index)
                self.src, self.dst = self.edge_index  ## 图结构解析完毕
                self.labels = torch.LongTensor(self.data.y.cpu()) ## 标签解析完毕    11.21.17：20
                print(self.data)
                self.mask = torch.BoolTensor(self.data.test_mask.cpu()) # use val as test, since val is larger than test
            else:
                self.features = torch.FloatTensor(self.data.x.cpu())
                self.ids = torch.LongTensor(list(range(self.features.size(0))))
                self.edge_index, _ = torch_geometric.utils.add_self_loops(self.data.edge_index)
                self.src, self.dst = self.edge_index  ## 图结构解析完毕
                self.labels = torch.LongTensor(self.data.y.cpu()) ## 标签解析完毕    11.21.17：20
                self.mask = torch.BoolTensor(self.data.test_mask.cpu())
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

        print('{} Dataset for {} Loaded.'.format(self.name, data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index):
        '''
        Return:
            if k > 1
            k_neighbor: (K, n, 1, f), K dimenstion is list, n is neighbor
            if k = 1
            neighbot: (n,1,f) 1 here for channels
            feature: (1,f)
            label: (1,)
        '''
        if self.k_hop == None:
            k_hop = 1
        else:
            k_hop = self.k_hop
        
        neighbors_khop = list()
        ids_khop = [self.ids[self.mask][index]]
        ## TODO: simplify this process
        for k in range(k_hop):
            ids = torch.LongTensor()
            neighbor = torch.FloatTensor()
            for i in ids_khop:
                ids = torch.cat((ids.cpu(), self.dst[self.src==i].cpu()),0)
                neighbor = torch.cat((neighbor, self.get_neighbor(i)),0)
            ## TODO random selection in pytorch is tricky
            if ids.shape[0]>self.thres_nodes:
                indices = torch.randperm(ids.shape[0])[:self.thres_nodes]
                ids = ids[indices]
                neighbor = neighbor[indices]
            ids_khop = ids ## temp ids for next level
            neighbors_khop.append(neighbor) #cat different level neighbor
        ## reserve for some simple baseline.
        if self.k_hop == None:
            neighbors_khop = neighbors_khop[0]
        return self.features[self.mask][index].unsqueeze(-2), self.labels[self.mask][index], neighbors_khop

    def get_neighbor(self, ids):
        return self.features[self.dst[self.src==ids]].unsqueeze(-2)
    
    def download(self, first_shard, second_shard):
        """Download data if it doesn't exist in processed_folder already."""
        # print('Loading {} Dataset...'.format(self.name))
        logger = logging.getLogger('Loading {} Dataset...'.format(self.name))
        logger.info('loading first shard data')
        first_shard_data = pickle.load(open(self.root+'/shard_data_sage_km_5_0.005_0', 'rb'))
        # print(first_shard_data)
        logger.info('loading second shard data')
        second_shard_data = {}
        for i in range(len(first_shard_data)):
            second_shard_data[i] = pickle.load(open(self.root+'/shard_data_sage_km_5_0.005_0' + 'second' + '_second_' + str(i), 'rb'))
            # print(second_shard_data[i])
        if self.again == 0:
            self.data = first_shard_data[first_shard]
        else:
            self.data = second_shard_data[first_shard][second_shard]
        
        self.feat_len, self.num_class = self.data.x.shape[1], len(list(set(self.data.y.cpu().numpy())))


    def delete_random_true_shorten_tensor(self, tensor):
        true_indices = torch.nonzero(tensor).squeeze()
        random_indices = torch.randperm(len(true_indices))[:1]
        deleted_indices = true_indices[random_indices]

        # 删除True的位置
        deleted_indices = sorted(deleted_indices, reverse=True)
        for index in deleted_indices:
            tensor = torch.cat([tensor[:index], tensor[index+1:]])

        return torch.tensor(deleted_indices), tensor
    
    def shorten_edge_index(self, edge_index, deleted_indices):
        # 检查 edge_index 是否是合法的大小
        if edge_index.size(0) != 2:
            raise ValueError("edge_index 应该是一个大小为 [2, m] 的张量。")
        
        # 将 deleted_indices 转换为长整型
        deleted_indices = deleted_indices.long()

        # 利用 torch.isin 创建掩码，以便删除包含在 deleted_indices 中的列
        mask = ~torch.isin(edge_index, deleted_indices).any(dim=0)
        shortened_edge_index = edge_index[:, mask]

        for idx in deleted_indices:
            shortened_edge_index[shortened_edge_index > idx] -= 1

        return shortened_edge_index
    
    def new_data_save(self, first_shard, second_shard):
        if second_shard == None:
            print('Updating first shard data')
            first_shard_data = pickle.load(open(self.root+'/shard_data_sage_km_5_0.005_0', 'rb'))
            first_shard_data[first_shard] = self.data
            pickle.dump(first_shard_data, open(self.root+'/shard_data_sage_km_5_0.005_0', 'wb'))
            print(first_shard_data)
        else:
            print('Updating second shard data')
            second_shard_data = pickle.load(open(self.root+'/shard_data_sage_km_5_0.005_0' + 'second' + '_second_' + str(first_shard), 'rb'))
            second_shard_data[second_shard] = self.data
            pickle.dump(second_shard_data, open(self.root+'/shard_data_sage_km_5_0.005_0' + 'second' + '_second_' + str(first_shard), 'wb'))
            print(second_shard_data)