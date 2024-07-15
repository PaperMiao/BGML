# Copyright <2019> <Chen Wang <https://chenwang.site>, Carnegie Mellon University>

# Redistribution and use in source and binary forms, with or without modification, are 
# permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of 
# conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list 
# of conditions and the following disclaimer in the documentation and/or other materials 
# provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be 
# used to endorse or promote products derived from this software without specific prior 
# written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
# DAMAGE.

import os
import tqdm
import copy
import logging
import pickle
import torch
import os.path
import configargparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch_geometric.utils
from models import LifelongRehearsal
from models import SAGE, GCN, APPNP, MLP, GAT, APP
from models import LGL, AFGN, PlainNet, AttnPlainNet
from models import KTransCAT, AttnKTransCAT
from datasets import continuum, graph_collate
from torch_util import count_parameters, EarlyStopScheduler, performance
import time 

## AFGN is LGL with attention; AttnPlainNet is the PlainNet with attention

## LGL和PlainNet分别是什么网络？ KTransCAT又是什么网络？
 
nets = {'sage':SAGE, 'lgl': LGL, 'ktranscat':KTransCAT, 'attnktranscat':AttnKTransCAT, 'gcn':GCN, 'appnp':APPNP, 'app':APP, 'mlp':MLP, 'gat':GAT, 'afgn':AFGN, 'plain':PlainNet, 'attnplain':AttnPlainNet}

def train(loader, net, criterion, optimizer, device):
    net.train()
    train_loss, correct, total = 0, 0, 0  ##训练损失，正确、总的
    for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(loader)):
        inputs, targets = inputs.to(device), targets.to(device)   ##输入和目标
        if not args.k:  ##不同层数的邻居
            neighbor = [element.to(device) for element in neighbor]
        else:
            neighbor = [[element.to(device) for element in item]for item in neighbor]

        optimizer.zero_grad()
        outputs = net(inputs, neighbor)  ##带梯度的结果

        loss = criterion(outputs, targets) ##根据是否预测准确测损失
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  
        ##outputs.data没有带梯度、输出数据值中的最大值和位置索引，不是和1比较
        ## predicted 预测的结果 0-6

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

    return (train_loss/(batch_idx+1), correct/total)

def shards_train(args, first_shard, second_shard = None, second = False):
    if second == True:
        name = 'second_shard_data'
    else:
        name = 'shard_data'
    # Datasets
    train_data = continuum(root=args.data_root, name=name, data_type='train', download=True, k_hop=args.k, first_shard = first_shard, second_shard = second_shard)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate, drop_last=True)
    ## train_loader加载的数据是一个封装好的东西“<torch.utils.data.dataloader.DataLoader object at 0x7f17aade6070>”
    test_data = continuum(root=args.data_root, name=name, data_type='test', download=True, k_hop=args.k, first_shard = first_shard, second_shard = second_shard)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
    # valid_data = continuum(root=args.data_root, name=args.dataset, data_type='valid', download=True, k_hop=args.k)
    # valid_loader = Data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)

    Net = nets[args.model.lower()]
    if args.model.lower() in ['ktranscat', 'ktranscat']:
        net = LifelongRehearsal(args, Net, feat_len=test_data.feat_len, num_class=test_data.num_class, k = args.k, hidden = args.hidden, drop = args.drop)
    else:
        net = LifelongRehearsal(args, Net, feat_len=test_data.feat_len, num_class=test_data.num_class, hidden = args.hidden, drop = args.drop)
    evaluation_metrics = []
    num_parameters = count_parameters(net)
    print('number of parameters:', num_parameters)

    if args.load is not None:
        net.backbone.load_state_dict(torch.load(args.load, map_location=args.device))
        train_acc, test_acc, valid_acc = performance(train_loader, net, args.device, k=args.k),  performance(test_loader, net, args.device, k=args.k)
        print("Train Acc: %.3f, Test Acc: %.3f, Valid Acc: %.3f"%(train_acc, test_acc, valid_acc))
        exit()

    start_time = time.time()
    task_ids = [i for i in range(test_data.num_class)]
    for i in range(0, test_data.num_class, args.merge):
        ## merge the class if needed
        if (i+args.merge > test_data.num_class):
            tasks_list = task_ids[i:test_data.num_class]
        else:
            tasks_list = task_ids[i:i+args.merge]

        incremental_data = continuum(root=args.data_root, name=args.dataset, data_type='incremental', download=True, task_type = tasks_list, k_hop = args.k)
        incremental_loader = Data.DataLoader(dataset=incremental_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate, drop_last=True)

        for batch_idx, (inputs, targets, neighbor) in enumerate(tqdm.tqdm(incremental_loader)):
            net.observe(inputs, targets, neighbor, batch_idx%args.jump==0)

        train_acc, test_acc = performance(incremental_loader, net, args.device, k=args.k), performance(test_loader, net, args.device, k=args.k)
        evaluation_metrics.append([i, len(incremental_data), train_acc, test_acc])
        print("Train Acc: %.3f, Test Acc: %.3f"%(train_acc, test_acc))

        if args.save is not None:
            torch.save(net.backbone.state_dict(), args.save + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard))

        if args.eval: 
            with open(args.eval + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard) +'-acc.txt','a') as file:
                file.write((str([i, train_acc, test_acc])+'\n').replace('[','').replace(']',''))
    train_time = time.time() - start_time ## 记录时间

    evaluation_metrics = torch.Tensor(evaluation_metrics)
    print('        | task | sample | train_acc | test_acc |')
    print(evaluation_metrics)

    return train_acc, test_acc, train_time

if __name__ == '__main__':
    # Arguements
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--device", type=str, default='cuda:1', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/home/mjx/GML-master/U+Data_incremental/Class_incremental/Partition/temp_data/processed_data/cora', help="learning rate")
    parser.add_argument("--model", type=str, default='attnplain', help="LGL or SAGE")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, pubmed")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default='/home/mjx/GML-master/U+Data_incremental/Class_incremental/Train_shards/shards_models', help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.02, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")  ## 触发条件后的学习率乘子
    parser.add_argument("--min-lr", type=float, default=0.001, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=3, help="patience for Early Stop") ## 提前停止的耐性值
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--memory-size", type=int, default=512, help="number of samples")
    parser.add_argument("--milestones", type=int, default=15, help="milestones for applying multiplier") ## 应用乘数的地方
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=5, help="number of epochs for early stop training") ## 提前停止训练的 epoch 数
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier") ## 学习率乘数
    parser.add_argument("--seed", type=int, default=1, help='Random seed.')
    parser.add_argument("--eval", type=str, default='/home/mjx/GML-master/U+Data_incremental/Class_incremental/Train_shards/shards_results', help="the path to eval the acc") ## 评估 acc 的路径
    parser.add_argument("--merge", type=int, default=1, help='Merge some class if needed.')
    parser.add_argument("--k", type=int, default=None, help='khop.')
    parser.add_argument("--hidden", type=int, nargs="+", default=[92,92])
    parser.add_argument("--drop", type=float, nargs="+", default=[0.3,0.1])
    parser.add_argument("--first_shards_num", type=int, default=5)
    parser.add_argument("--second_shards_num", type=int, default=2)
    parser.add_argument("--jump", type=int, default=1, help="reply samples")
    parser.add_argument("--iteration", type=int, default=10, help="number of training iteration")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    first_shards_results = {}
    T_time = {}
    rank = []
    for first_sh in range(args.first_shards_num):
        print('training target models, first shard %s' % (first_sh))
        train_acc, test_acc, train_time = shards_train(args, first_sh)
        print("Model training time: %s" % (train_time)) ## 日志显示训练时间
        first_shards_results[first_sh] = train_acc, test_acc
        T_time[first_sh] = train_time
        rank.append(test_acc)
    par_shard = rank.index(min(rank))
    
    second_shard_results = {}
    for second_sh in range(args.second_shards_num):
        print('training target models, first shard %s , second shard %s' % (par_shard, second_sh))
        train_acc, test_acc, train_time = shards_train(args, par_shard, second_sh, second = True)
        print("Model training time: %s" % (train_time)) ## 日志显示训练时间
        second_shard_results[second_sh] = train_acc, test_acc
        print(second_shard_results)

    pickle.dump(T_time, open('/home/mjx/GML-master/U+Data_incremental/Class_incremental/Train_shards/time.pkl', 'wb'))