import logging
import time
import torch
import tqdm
import glob
import os
import re
import torch.nn as nn
import numpy as np
from exp import Exp
from lib_aggregator.aggregator import Aggregator
from models.models import AttnPlainNet
from torch_geometric.data import NeighborSampler
from lib_utils import utils
import torch.nn.functional as nf
from datasets import continuum, graph_collate
import torch.utils.data as Data
import pickle
from sklearn.metrics import f1_score
from lib_aggregator.optimal_aggregator import OptimalAggregator

class F1_score_aggregator(Exp):
    def __init__(self, args):
        super(F1_score_aggregator, self).__init__(args)

        self.logger = logging.getLogger('F1_score_aggregator')
        par_shard = 4
        self.load_data(par_shard) ## 加载数据
        self.time = pickle.load(open('/home/mjx/GML-master/U+Data_incremental/U+D/Train_and_test/time.pkl', 'rb'))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model_name = self.args['target_model'] ## 模型
        self.num_opt_samples = self.args['num_opt_samples'] ## 
        self.save = self.args['load']
        self.num_shards = self.args['num_shards']
        
        num_feats = self.data.num_features
        num_classes = len(self.data.y.unique())
        hidden = self.args['hidden']
        dropout = self.args['drop']
        self.target_model = AttnPlainNet(num_feats, num_classes, hidden, dropout)
       
        self.true_label_0 = self.shard_data[0].y[self.shard_data[0]['test_mask']].detach().cpu().numpy() ## 真实标签 
        self.first_posteriors = {}
        start_time = time.time()
        for first_shard in range(self.args['num_shards']):
            self.target_model.data = self.shard_data[first_shard] ## 碎片数据输入
            self.test_data = continuum(root=self.args['data_root'], name='shard_data', data_type='test', download=True, k_hop=self.args['k'], first_shard = first_shard, second_shard = None)
            self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=self.args['batch_size'], shuffle=False, collate_fn=graph_collate, drop_last=True)
            self.load_target_model(self.target_model, first_shard) ## 加载原来训练好储存的目标模型
            self.first_posteriors[first_shard] = self.posterior(self.target_model, self.target_model.data) ## 碎片的后验
        self.logger.info("Saving first_shard posteriors.")
        self.data_store.save_posteriors(self.first_posteriors, suffix = '_first') ## 保存后验信息

        self.second_posteriors = {}
        for second_sh in range(self.args['second_num_shards']):
            self.target_model.shard_data = self.second_shard_data[second_sh] ## 碎片数据输入
            self.test_data = continuum(root=self.args['data_root'], name='second_shard_data', data_type='test', download=True, k_hop=self.args['k'], first_shard = par_shard, second_shard = second_sh)
            self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=self.args['batch_size'], shuffle=False, collate_fn=graph_collate, drop_last=True)
            self.load_target_model(self.target_model, par_shard, second_sh) ## 加载原来训练好储存的目标模型
            self.second_posteriors[second_sh] = self.posterior(self.target_model, self.target_model.shard_data) ## 碎片的后验

        self.logger.info("Saving second_shard posteriors.")
        self.data_store.save_posteriors(self.second_posteriors, suffix = '_second') ## 保存后验信息
        self.true_label = self.true_label_0[0:len(self.first_posteriors[0])]

        if self.args['is_Unlearning'] == True:
            files = glob.glob(r'/home/mjx/GML-master/U+Data_incremental/Unlearning/Train_shards/shards_models/Unlearning_first*')
            filename = [os.path.basename(file) for file in files]
            U_shard = self.extract_numbers_and_none(filename[0])
            if U_shard[1] == None:
                self.target_model.data = self.shard_data[U_shard[0]] ## 碎片数据输入
                self.test_data = continuum(root=self.args['data_root'], name='shard_data', data_type='test', download=True, k_hop=self.args['k'], first_shard = U_shard[0], second_shard = None)
                self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=self.args['batch_size'], shuffle=False, collate_fn=graph_collate, drop_last=True)
                self.load_target_model(self.target_model, first_shard) ## 加载原来训练好储存的目标模型
                self.first_posteriors[U_shard[0]] = self.posterior(self.target_model, self.target_model.data) ## 碎片的后验
                self.logger.info("Saving U_first_shard posteriors.")
                self.data_store.save_posteriors(self.first_posteriors, suffix = '_first_U') ## 保存后验信息

            else:
                self.target_model.shard_data = self.second_shard_data[U_shard[1]] ## 碎片数据输入
                self.test_data = continuum(root=self.args['data_root'], name='second_shard_data', data_type='test', download=True, k_hop=self.args['k'], first_shard = U_shard[0], second_shard = U_shard[1])
                self.test_loader = Data.DataLoader(dataset=self.test_data, batch_size=self.args['batch_size'], shuffle=False, collate_fn=graph_collate, drop_last=True)
                self.load_target_model(self.target_model, U_shard[0], U_shard[1]) ## 加载原来训练好储存的目标模型
                self.second_posteriors[U_shard[1]] = self.posterior(self.target_model, self.target_model.shard_data) ## 碎片的后验
                self.logger.info("Saving U_second_shard posteriors.")
                self.data_store.save_posteriors(self.first_posteriors, suffix = '_second_U') ## 保存后验信息

        score, self.first_posteriors[par_shard] = self._optimal_aggregator(self.second_posteriors)
        aggregate_f1_score = self.aggregate(self.first_posteriors) ## 聚合
        aggregate_time = time.time() - start_time  ## 记录总体的聚合时间  

        self.logger.info("Aggregate cost %s seconds." % aggregate_time) ## 输出聚合花费的时间
        self.logger.info("Final Test F1: %s" % (aggregate_f1_score)) ## 日志输出最后精度
        node_unlearning_time = self.unlearning_time_statistic() ## 总的时间汇总成列表，每个节点、每个碎片组的遗忘

        unlearning_time = np.append(1, node_unlearning_time) ## 记录每次的遗忘时间表
        self.unlearning_time_avg = np.average(unlearning_time) ## 取平均
        self.unlearning_time_std = np.std(unlearning_time) ## 误差
        self.logger.info("%s %s %s" % (aggregate_f1_score, self.unlearning_time_avg, self.unlearning_time_std))

    def load_data(self, par_shard):
        self.shard_data = self.data_store.load_first_shard_data() ## 加载划分好的碎片数据
        self.second_shard_data = self.data_store.load_second_shard_data('_second_'+str(par_shard)) ## 加载最差碎片的二级碎片粒
        self.data = self.data_store.load_raw_data() ## 加载初始数据

    def load_target_model(self, model, first_shard, second_shard = None):
        model.load_state_dict(torch.load(self.save + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard), map_location='cpu'))

    def posterior(self, model, data):
        self.logger.debug("generating posteriors")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.data = model.to(self.device), data.to(self.device)
        self.model.eval()

        posteriors = self.performance()  ## inference已经放进去了，需要重新写
        posteriors = nf.log_softmax(posteriors, dim=-1)
        return posteriors.detach()

    def performance(self):
        with torch.no_grad():
            x = []
            for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(self.test_loader)):
                if torch.cuda.is_available():
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    if not self.args['k']:
                        neighbor = [element.to(self.device) for element in neighbor]
                    else:
                        neighbor = [[element.to(self.device) for element in item]for item in neighbor]

                outputs = self.model(inputs, neighbor)
                x.append(outputs.cpu())
            output = torch.cat(x, dim=0)
        return output

    def unlearning_time_statistic(self): ## 遗忘时间统计
        if self.args['is_train_target_model'] and self.args['num_shards'] != 1: ## 满足条件
            self.community_to_node = self.data_store.load_community_data() # 加载划分好的社区数据
            node_list = []
            for key, value in self.community_to_node.items():
                node_list.extend(value)   ## 拉直成为整个节点表

            # random sample 5% nodes, find their belonging communities
            sample_nodes = np.random.choice(node_list, int(0.05 * len(node_list)))
            belong_community = []
            for sample_node in range(len(sample_nodes)):
                for community, node in self.community_to_node.items():
                    if np.in1d(sample_nodes[sample_node], node).any():
                        belong_community.append(community)

            # calculate the total unlearning time and group unlearning time
            group_unlearning_time = [] ## 组遗忘时间
            node_unlearning_time = [] ## 节点遗忘时间
            for shard in range(self.args['num_shards']):
                if belong_community.count(shard) != 0:
                    group_unlearning_time.append(self.time[shard]) ## 按组添加
                    node_unlearning_time.extend([float(self.time[shard]) for j in range(belong_community.count(shard))]) ## 直接将所有时间列表扩张
 
            return node_unlearning_time

        elif self.args['is_train_target_model'] and self.args['num_shards'] == 1:
            return self.time[0]

        else:
            return 0

    def aggregate(self, posteriors):
        if self.args['aggregator'] == 'mean':
            aggregate_f1_score = self._mean_aggregator(posteriors) ## 平均聚合
        elif self.args['aggregator'] == 'optimal':
            aggregate_f1_score = self._optimal_aggregator(posteriors) ## 自适应聚合
        elif self.args['aggregator'] == 'majority':
            aggregate_f1_score = self._majority_aggregator(posteriors) ## 最大聚合
        else:
            raise Exception("unsupported aggregator.")

        return aggregate_f1_score

    def _mean_aggregator(self, posteriors): ## 平均聚合
        posterior = posteriors[0]
        for shard in range(1, self.num_shards):
            posterior += posteriors[shard]

        posterior = posterior / self.num_shards 
        return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro")

    def _majority_aggregator(self, posteriors): ## 最大聚合，直接找最好的碎片
        pred_labels = []
        for shard in range(self.args['second_num_shards']):
            pred_labels.append(posteriors[shard].argmax(axis=1).cpu().numpy())

        pred_labels = np.stack(pred_labels)
        pred_label = np.argmax(
            np.apply_along_axis(np.bincount, axis=0, arr=pred_labels, minlength=posteriors[0].shape[1]), axis=0)

        return f1_score(self.true_label, pred_label, average="micro"), 0.8*posteriors[0] + 0.2*posteriors[1]

    def _optimal_aggregator(self, posteriors):
        optimal = OptimalAggregator(1, self.target_model, self.data, self.args) ## 定义自适应聚合优化器
        optimal.generate_train_data(posteriors) ## 加载训练数据
        weight_para = optimal.optimization() ## 优化训练
        self.logger.info("second shards weight_para %s %s" % (weight_para[0], weight_para[1]))
        self.data_store.save_optimal_weight(weight_para, run=1) ## 存储权重

        posterior = posteriors[0] * weight_para[0] 
        for shard in range(1, self.args['second_num_shards']):
            posterior += posteriors[shard] * weight_para[shard] ## 根据得到的权重进行聚合

        return f1_score(self.true_label, posterior.argmax(axis=1).cpu().numpy(), average="micro"), posterior
    

    def extract_numbers_and_none(self, input_string):
        # 使用正则表达式匹配数字和None
        matches = re.findall(r'\d+|None', input_string)
        
        # 将匹配到的字符串转换为相应的类型
        result = [int(match) if match.isdigit() else None for match in matches]
        
        return result
