import os
import logging

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler

from lib_gnn_model.graphsage.graphsage_net import SageNet
from lib_gnn_model.gnn_base import GNNBase
import config


class SAGE(GNNBase):
    def __init__(self, args, num_feats, num_classes, data=None):
        super(SAGE, self).__init__()
        self.logger = logging.getLogger('graphsage')
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ## GPU
        # self.device = torch.device('cpu')
        self.hidden = self.args['SAGE_hidden']
        self.model = SageNet(num_feats, self.hidden, num_classes).to(self.device) ## 定义模型 输入、隐藏层神经元个数、类别数 ## 这个网络定义的没有问题，就是普通的两层SAGE
        self.data = data

    def train_model(self):  ## 默认训练100轮，更改至中控！
        self.model.train()  ## 训练
        self.model.reset_parameters()  ## 重置参数
        self.model, self.data = self.model.to(self.device), self.data.to(self.device) 
        self.data.y = self.data.y.squeeze().to(self.device) ## 标签
        self._gen_train_loader()  ## 定义训练数据、每层采样多少什么的
        self.num_epochs = self.args['num_epochs']

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['train_lr'], weight_decay=self.args['train_weight_decay']) ## 优化器，参数设置

        for epoch in range(self.num_epochs):
            self.logger.info('epoch %s' % (epoch,)) ## 日志生成
            for batch_size, n_id, adjs in self.train_loader: ## 从train_loader中提取的内容
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples. 因为是两层，所以会有一个列表储存
                # adjs 与批量的划分也有关，就是SAGE流程图中的东西
                adjs = [adj.to(self.device) for adj in adjs]

                optimizer.zero_grad()
                out = self.model(self.data.x[n_id], adjs)
                loss = F.nll_loss(out, self.data.y[n_id[:batch_size]])
                loss.backward()
                optimizer.step() ## 完整的正常训练过程

            train_acc, test_acc = self.evaluate_model() ## 测试, 评估模型
            self.logger.info(f'Train: {train_acc:.4f}, Test: {test_acc:.4f}') ## 得到训练结果和测试结果 

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_subgraph_loader() ## 生成用于测试的子图序列集

        out = self.model.inference(self.data.x, self.subgraph_loader, self.device) ## 计算输出，最终表示
        y_true = self.data.y.cpu().unsqueeze(-1) ## 标签
        y_pred = out.argmax(dim=-1, keepdim=True) ## 模型预测值

        results = []
        for mask in [self.data.train_mask, self.data.test_mask]:
            results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

        return results ## 结果中包含训练和测试的结果

    def posterior(self): ## 后验信息，不知道啥时候用到，看到再说~~~
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_subgraph_loader()

        posteriors = self.model.inference(self.data.x, self.subgraph_loader, self.device)
        
        for _, mask in self.data('test_mask'):
            posteriors = F.log_softmax(posteriors[mask], dim=-1) ## log_softmax在softmax的结果上再做多一次log运算

        return posteriors.detach()

    def generate_embeddings(self):
        self.model.eval()
        self.model, self.data = self.model.to(self.device), self.data.to(self.device)
        self._gen_subgraph_loader()

        logits = self.model.inference(self.data.x, self.subgraph_loader, self.device)
        return logits ## 生成所需的节点嵌入

    def _gen_train_loader(self):
        if self.data.edge_index.shape[1] == 0:  ## 也就是说没有边的存在
            self.data.edge_index = torch.tensor([[1, 2], [2, 1]]) ## 生成一个空的边索引对
        self.train_loader = NeighborSampler(self.data.edge_index, node_idx=self.data.train_mask,
                                            # sizes=[25, 10], batch_size=128, shuffle=True,
                                            # sizes=[25, 10], num_nodes=self.data.num_nodes,
                                            sizes=[10, 10], num_nodes=self.data.num_nodes,
                                            # sizes=[5, 5], num_nodes=self.data.num_nodes,
                                            # batch_size=128, shuffle=True,
                                            batch_size=64, shuffle=True,
                                            num_workers=0)  ## 训练图生成器 根据邻居采样

    def _gen_subgraph_loader(self):
        self.subgraph_loader = NeighborSampler(self.data.edge_index, node_idx=None,
                                               # sizes=[-1], num_nodes=self.data.num_nodes,
                                               sizes=[10], num_nodes=self.data.num_nodes,
                                               # batch_size=128, shuffle=False,
                                               batch_size=64, shuffle=False,
                                               num_workers=0) ## 子图生成器


if __name__ == '__main__':
    os.chdir('../../')

    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    dataset_name = 'cora'
    dataset = Planetoid(config.RAW_DATA_PATH, dataset_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    graphsage = SAGE(dataset.num_features, dataset.num_classes, data)
    graphsage.train_model()
