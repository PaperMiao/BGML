import os
import tqdm
import dgl
import torch
import pickle
import logging
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
    def __init__(self, root='/home/mjx/课题1代码/GML-master/Partition/temp_data/processed_data/cora', name='shard_data', data_type='train', k_hop=1, download=True, task_type=0, thres_nodes = 50, first_shard = None, second_shard = None):
        super(ContinuumShard, self).__init__(root)
        self.name = name
        self.thres_nodes = thres_nodes
        self.k_hop = k_hop
        if name == 'shard_data':
            self.again = 0
        else:
            self.again = 1
        self.download(first_shard, second_shard)
        self.features = torch.FloatTensor(self.data.x.cpu())
        self.ids = torch.LongTensor(list(range(self.features.size(0))))
        self.edge_index, _ = torch_geometric.utils.add_self_loops(self.data.edge_index)
        self.src, self.dst = self.edge_index  ## 图结构解析完毕
        self.labels = torch.LongTensor(self.data.y.cpu()) ## 标签解析完毕    11.21.17：20

        if data_type == 'train': # data incremental; use test and train as train
            self.mask = self.data.train_mask
        elif data_type == 'incremental': # class incremental; use test and train as train
            mask = self.data.train_mask
            if type(task_type)==list:
                self.mask = torch.BoolTensor()
                for i in task_type:
                    self.mask =torch.cat([self.mask,np.logical_and((self.labels==i),mask).type(torch.bool)],0)
            else:
                self.mask = (np.logical_and((self.labels==task_type),mask)).type(torch.bool)
        elif data_type == 'test':
            self.mask = torch.BoolTensor(self.data.test_mask.cpu()) # use val as test, since val is larger than test
        elif data_type == 'valid':
            self.mask = torch.BoolTensor(self.data.val_mask)
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