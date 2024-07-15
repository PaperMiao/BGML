import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from lib_aggregator.opt_dataset import OptDataset
from lib_dataset.data_store import DataStore
from lib_utils import utils
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class OptimalAggregator:
    def __init__(self, run, target_model, data, args):
        self.logger = logging.getLogger('optimal_aggregator') ## 自适应聚合
        self.args = args
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.run = run
        self.target_model = target_model
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_shards = args['num_shards']

    def generate_train_data(self, posteriors):
        data_store = DataStore(self.args) ## 存储数据
        train_indices, _ = data_store.load_train_test_split() ## 加载训练测试集划分

        # sample a set of nodes from train_indices 从训练索引中采样
        if self.args["num_opt_samples"] == 100:
            train_indices = np.random.choice(list(range(300)), size=100, replace=False)
        elif self.args["num_opt_samples"] == 10000:
            train_indices = np.random.choice(train_indices, size=int(train_indices.shape[0] * 0.1), replace=False)
        elif self.args["num_opt_samples"] == 1:
            train_indices = np.random.choice(train_indices, size=int(train_indices.shape[0]), replace=False)

        train_indices = np.sort(train_indices) ## 对采样的索引进行排序
        self.logger.info("Using %s samples for optimization" % (int(train_indices.shape[0]))) ## 日志
        x = self.data.x[train_indices] ## 训练特征
        y = self.data.y[train_indices]  ## 标签
        edge_index = utils.filter_edge_index(self.data.edge_index, train_indices) ## 根据采样出来的节点，对应地裁剪边索引出来

        train_data = Data(x=x, edge_index=torch.from_numpy(edge_index), y=y)
        train_data.train_mask = torch.zeros(train_indices.shape[0], dtype=torch.bool) ## 训练掩码
        train_data.test_mask = torch.ones(train_indices.shape[0], dtype=torch.bool)
        self.true_labels = y

        self.posteriors = posteriors

    def optimization(self):
        weight_para = nn.Parameter(torch.full((self.args['second_num_shards'],), fill_value=1.0 / self.args['second_num_shards']), requires_grad=True) ## 权重初始化
        optimizer = optim.Adam([weight_para], lr=self.args['opt_lr']) ## 优化器
        scheduler = MultiStepLR(optimizer, milestones=[500, 1000], gamma=self.args['opt_lr']) ## 学习率变化器

        train_dset = OptDataset(self.posteriors, self.true_labels) ## 将后验和真实标签进行训练
        train_loader = DataLoader(train_dset, batch_size=32, shuffle=True, num_workers=0) ## 加载训练数据
        lambda_B = 0.8
        min_loss = 1000.0
        for epoch in range(self.args['opt_num_epochs']):
            loss_all = 0.0

            for posteriors, labels in train_loader:
                labels = labels.to(self.device)

                optimizer.zero_grad()
                loss = self._loss_fn(posteriors, labels, weight_para)
                loss.backward()
                loss_all += loss

                optimizer.step()
                with torch.no_grad():
                    weight_para[:] = torch.clamp(weight_para, min=0.0) ## 得到权重

            scheduler.step()

            if loss_all < min_loss:
                ret_weight_para = copy.deepcopy(weight_para) ## 复制无cuda权重
                min_loss = loss_all

        weight_raw = ret_weight_para / torch.sum(ret_weight_para) ## 归一化权重

        weight = weight_raw
        weight[weight_raw.argmax()] = lambda_B
        weight[weight_raw.argmin()] = 1-lambda_B

        return weight

    def _loss_fn(self, posteriors, labels, weight_para):  ## 自适应参数模型的损失设计
        aggregate_posteriors = torch.zeros_like(posteriors[0])
        for shard in range(self.args['second_num_shards']):
            aggregate_posteriors += weight_para[shard] * posteriors[shard]

        aggregate_posteriors = F.softmax(aggregate_posteriors, dim=1).to(self.device)
        loss_1 = F.cross_entropy(aggregate_posteriors, labels) ## 交叉熵损失
        loss_2 = torch.sqrt(torch.sum(weight_para ** 2)) ## L2范数

        return loss_1 + loss_2
