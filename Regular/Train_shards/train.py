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

    Net = nets[args.model.lower()] ## .lower() 将模型中的大写换为小写

    if args.model.lower() in ['ktranscat', 'attnktranscat']:  ## 是否有K阶邻居的要求
        net = Net(feat_len=train_data.feat_len, k=args.k, num_class=train_data.num_class).to(args.device)
    else:
        assert(args.k == None)
        net = Net(feat_len=train_data.feat_len, num_class=train_data.num_class, hidden = args.hidden, dropout = args.drop).to(args.device)
    # print(net)

    if args.load is not None: ## 是否有预训练模型需要加载
        net.load_state_dict(torch.load(args.load, map_location=args.device))  ## 加载模型
        train_acc, test_acc = performance(train_loader, net, args.device, args.k),  performance(test_loader, net, args.device, args.k)
        ## 直接执行预训练模型，并未产生梯度回传，训练网络
        print("Train Acc: %.5f, Test Acc: %.5f"%(train_acc, test_acc))
        exit()

    print(net)
    criterion = nn.CrossEntropyLoss() ## 交叉熵损失
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)  ## 定义优化算法
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)
    ## 提前停止器

    # Training
    paramsnumber = count_parameters(net) ## 计算网络参数并输出
    print('number of parameters:', paramsnumber)
    if args.eval: ## 如果是True，可以写入一个文件
        with open(args.eval + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard) +'-acc.txt','w') as file:
            file.write('shard-first_' + str(first_shard) +'-second_' + str(second_shard) + " number of prarams " + str(paramsnumber) + "\n")
            file.write("epoch | train_acc | test_acc |\n")

    best_acc = 0 ## 初始化
    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, net, criterion, optimizer, args.device) # 根据训练函数将所有数据按批量训练一次
        test_acc = performance(test_loader, net, args.device, args.k) # validate 测试性能
        print("epoch: %d, train_loss: %.4f, train_acc: %.4f, test_acc: %.4f"
                % (epoch, train_loss, train_acc, test_acc))
        if args.eval: ## 记录实验结果
            with open(args.eval + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard) +'-acc.txt','a') as file:
                file.write((str([epoch, train_acc, test_acc])+'\n').replace('[','').replace(']',''))  

        if test_acc > best_acc:
            print("New best Model, copying...")
            best_acc, best_net = test_acc, copy.deepcopy(net) ## 显示最好的结果和复制最好的模型

        if scheduler.step(error=1-test_acc): ## 达到误差，或3轮不上升
            print('Early Stopping!')
            break
    train_time = time.time() - start_time ## 记录时间

    torch.save(best_net.state_dict(), args.save + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard))
    print('save shards models, first shard %s , second shard %s' % (first_shard, second_shard))
    train_acc, test_acc = performance(train_loader, best_net, args.device, args.k), performance(test_loader, best_net, args.device, args.k)
    print('train_acc: %.4f, test_acc: %.4f'%(train_acc, test_acc))
    ## 输出最好的模型对应的实验结果

    if args.eval: ## 可写入txt文件记录
        with open(args.eval + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard) +'-acc.txt','a') as file:
            file.write((str([epoch, train_acc, test_acc])+'\n').replace('[','').replace(']',''))

    return train_acc, test_acc, train_time

if __name__ == '__main__':
    # Arguements
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/home/mjx/GML-master/Regular/Partition/temp_data/processed_data/cora', help="learning rate")
    parser.add_argument("--model", type=str, default='attnplain', help="LGL or SAGE")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, pubmed")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default='/home/mjx/GML-master/Regular/Train_shards/shards_models', help="model file to save")
    parser.add_argument("--optm", type=str, default='SGD', help="SGD or Adam")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")  ## 触发条件后的学习率乘子
    parser.add_argument("--min-lr", type=float, default=0.001, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=3, help="patience for Early Stop") ## 提前停止的耐性值
    parser.add_argument("--batch-size", type=int, default=10, help="number of minibatch size")
    parser.add_argument("--milestones", type=int, default=15, help="milestones for applying multiplier") ## 应用乘数的地方
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--early-stop", type=int, default=5, help="number of epochs for early stop training") ## 提前停止训练的 epoch 数
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum of the optimizer")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier") ## 学习率乘数
    parser.add_argument("--seed", type=int, default=1, help='Random seed.')
    parser.add_argument("--eval", type=str, default='/home/mjx/GML-master/Regular/Train_shards/shards_results', help="the path to eval the acc") ## 评估 acc 的路径
    parser.add_argument("--k", type=int, default=None, help='khop.')
    parser.add_argument("--hidden", type=int, nargs="+", default=[80,80])
    parser.add_argument("--drop", type=float, nargs="+", default=[0.3,0.1])
    parser.add_argument("--first_shards_num", type=int, default=5)
    parser.add_argument("--second_shards_num", type=int, default=2)
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

    pickle.dump(T_time, open('/home/mjx/GML-master/Regular/Train_shards/time.pkl', 'wb'))