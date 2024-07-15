import os
import dgl
import tqdm
import torch
import os.path
import numpy as np
import scipy.sparse as sp
from dgl import DGLGraph
from dgl.data import citegrh
from itertools  import compress
from torchvision.datasets import VisionDataset
from .continuumLS import ContinuumLS
from .continuumOGB import ContinuumOGB
from .continuumShard import ContinuumShard


def graph_collate(batch):
    feature = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.stack([item[1] for item in batch], dim=0)
    neighbor = [item[2] for item in batch]
    return [feature, labels, neighbor]

def continuum(root='/datasets/', name='reddit', data_type='train', task_type = 0, k_hop = 1, download=True, thres_nodes = 20, first_shard = None, second_shard = None):
    name = name.lower()
    if name in ['reddit', 'flickr']:
        return ContinuumLS(root=root, name=name, data_type=data_type, task_type = task_type, download=download, k_hop = k_hop, thres_nodes = thres_nodes)
    elif name in ['cora', 'citeseer', 'pubmed']:
        return Continuum(root=root, name=name, data_type=data_type, task_type = task_type, k_hop = k_hop, download=download, thres_nodes = thres_nodes)
    elif name in ["ogbn-products", "ogbn-arxiv", "ogbn-proteins"]:
        return ContinuumOGB(root=root, name=name, data_type=data_type, task_type = task_type, download=download,k_hop = k_hop, thres_nodes = thres_nodes)
    elif name in ['shard_data', 'second_shard_data']:
        return ContinuumShard(root=root, name=name, data_type=data_type, task_type = task_type, download=download,k_hop = k_hop, thres_nodes = thres_nodes, first_shard = first_shard, second_shard = second_shard)
    else:
        raise RuntimeError('name type {} wrong'.format(name))

class Continuum(VisionDataset):
    def __init__(self, root='~/.dgl', name='cora', data_type='train', k_hop=1, download=True, task_type=0, thres_nodes = 50):
        super(Continuum, self).__init__(root)
        self.name = name
        self.thres_nodes = thres_nodes ## 干啥的？
        self.k_hop = k_hop # 这个得留着
        self.download() ## 下载数据，这个和其他联网下载的数据不同，需要接入本地存下的碎片数据
        self.features = torch.FloatTensor(self.data.features) # 特征，可接
        self.ids = torch.LongTensor(list(range(self.features.size(0)))) ## 索引 特征的第一维，节点数 可接
        graph = self.data[0] ## graph？
        print(graph)
        graph = dgl.add_self_loop(graph)
        self.src, self.dst = graph.edges() ## 有索引，需要变成tensor
        self.labels = torch.LongTensor(self.data.labels) ## 标签也有

        if data_type == 'train': # data incremental; use test and train as train
            self.mask = np.logical_or(self.data.test_mask, self.data.train_mask) ## 这个也可以做到，训练集和测试集掩码都有
        elif data_type == 'incremental': # class incremental; use test and train as train
            mask = np.logical_or(self.data.test_mask, self.data.train_mask) ## 可行
            if type(task_type)==list:
                self.mask = torch.BoolTensor()
                for i in task_type:
                    self.mask =torch.cat([self.mask,np.logical_and((self.labels==i),mask).type(torch.bool)],0) ## 逻辑判断，应该也可行吧
            else:
                self.mask = (np.logical_and((self.labels==task_type),mask)).type(torch.bool)
        elif data_type == 'test':
            self.mask = torch.BoolTensor(self.data.val_mask) # use val as test, since val is larger than test ## 没有验证集？这段如果用不到。直接删除
        elif data_type == 'valid':
            self.mask = torch.BoolTensor(self.data.val_mask)
        else:
            raise RuntimeError('data type {} wrong'.format(data_type))

        print('{} Dataset for {} Loaded.'.format(self.name, data_type))

    def __len__(self):
        return len(self.labels[self.mask])

    def __getitem__(self, index): ## index似乎不用管，会按照索引自行运转每个节点
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
        
        neighbors_khop = list() ## 一个空的list
        ids_khop = [self.ids[self.mask][index]] ## ？这是干嘛的 倒是可行
        ## TODO: simplify this process
        for k in range(k_hop): # 通常只用一阶的
            ids = torch.LongTensor() ## 节点序列数
            neighbor = torch.FloatTensor()
            for i in ids_khop:
                ids = torch.cat((ids, self.dst[self.src==i]),0)
                neighbor = torch.cat((neighbor, self.get_neighbor(i)),0)
            ## TODO random selection in pytorch is tricky
            if ids.shape[0]>self.thres_nodes:
                indices = torch.randperm(ids.shape[0])[:self.thres_nodes] ## 不要太多，可以切片
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
    
    def download(self):
        """Download data if it doesn't exist in processed_folder already."""
        print('Loading {} Dataset...'.format(self.name))
        processed_folder = os.path.join(self.root, self.name)
        os.makedirs(processed_folder, exist_ok=True)
        os.environ["DGL_DOWNLOAD_DIR"] = processed_folder
        data_file = os.path.join(processed_folder, 'data.pt')
        if os.path.exists(data_file):
            self.data = torch.load(data_file)
        else:
            if self.name.lower() == 'cora':
                self.data = citegrh.load_cora()
            elif self.name.lower() == 'citeseer':
                self.data = citegrh.load_citeseer()
            elif self.name.lower() == 'pubmed':
                self.data = citegrh.load_pubmed()
            else:
                raise RuntimeError('Citation dataset name {} wrong'.format(self.name))
            with open(data_file, 'wb') as f:
                torch.save(self.data, data_file)
        #  print(self.data) 对于碎片数据来说，直接重写一下加载数据集的函数，没啥参考意义
        self.feat_len, self.num_class = self.data.features.shape[1], self.data.num_labels
