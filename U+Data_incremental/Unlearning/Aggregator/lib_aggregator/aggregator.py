import logging
import torch

torch.cuda.empty_cache()

from sklearn.metrics import f1_score
import numpy as np

from lib_aggregator.optimal_aggregator import OptimalAggregator
from lib_dataset.data_store import DataStore


class Aggregator:
    def __init__(self, target_model, data, shard_data, args):
        self.logger = logging.getLogger('Aggregator') ## 日志聚合
        self.args = args

        self.data_store = DataStore(self.args) ## 数据存储

        self.target_model = target_model ## 目标模型
        self.data = data ## 原始数据
        self.shard_data = shard_data ## 碎片数据

        self.num_shards = args['num_shards']

    def generate_posterior(self, suffix=""): ## 生成后验，后验信息是为了后续计算聚合之后的总结果建的
        self.true_label = self.shard_data[0].y[self.shard_data[0]['test_mask']].detach().cpu().numpy() ## 真实标签
        self.posteriors = {}

        for shard in range(self.args['num_shards']):
            self.target_model.data = self.shard_data[shard] ## 碎片数据输入
            self.data_store.load_target_model(self.run, self.target_model, shard, suffix) ## 加载原来训练好储存的目标模型
            self.posteriors[shard] = self.target_model.posterior() ## 碎片的后验
        self.logger.info("Saving posteriors.")
        self.data_store.save_posteriors(self.posteriors, self.run, suffix) ## 保存后验信息

    def aggregate(self):
        if self.args['aggregator'] == 'mean':
            aggregate_f1_score = self._mean_aggregator() ## 平均聚合
        elif self.args['aggregator'] == 'optimal':
            aggregate_f1_score = self._optimal_aggregator() ## 自适应聚合
        elif self.args['aggregator'] == 'majority':
            aggregate_f1_score = self._majority_aggregator() ## 最大聚合
        else:
            raise Exception("unsupported aggregator.")

        return aggregate_f1_score

    def _mean_aggregator(self): ## 平均聚合
        posterior = self.posteriors[0]
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard]

        posterior = posterior / self.num_shards 
        return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro")

    def _majority_aggregator(self): ## 最大聚合，直接找最好的碎片
        pred_labels = []
        for shard in range(self.num_shards):
            pred_labels.append(self.posteriors[shard].argmax(axis=1).cpu().numpy())

        pred_labels = np.stack(pred_labels)
        pred_label = np.argmax(
            np.apply_along_axis(np.bincount, axis=0, arr=pred_labels, minlength=self.posteriors[0].shape[1]), axis=0)

        return f1_score(self.true_label, pred_label, average="micro")

    def _optimal_aggregator(self):
        optimal = OptimalAggregator(self.run, self.target_model, self.data, self.args) ## 定义自适应聚合优化器
        optimal.generate_train_data() ## 加载训练数据
        weight_para = optimal.optimization() ## 优化训练
        self.data_store.save_optimal_weight(weight_para, run=self.run) ## 存储权重

        posterior = self.posteriors[0] * weight_para[0] 
        for shard in range(1, self.num_shards):
            posterior += self.posteriors[shard] * weight_para[shard] ## 根据得到的权重进行聚合

        return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro")
