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
import random
import logging
import pickle
import torch
import os.path
import configargparse
import numpy as np
import torch.nn as nn
import torch.utils.data as Data_1
from torch_geometric.data import Data
from torch.nn.functional import pairwise_distance
from torch.autograd import Variable
import torch_geometric.utils
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from models import SAGE, GCN, APPNP, MLP, GAT, APP
from models import LGL, AFGN, PlainNet, AttnPlainNet
from models import KTransCAT, AttnKTransCAT
from datasets import continuum, graph_collate
from torch_util import count_parameters, EarlyStopScheduler, performance
import time 
import cupy as cp

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

def load_raw_data(args):
    if args.dataset in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(args.raw_root, args.dataset, transform=T.NormalizeFeatures())
        # labels = np.unique(dataset.data.y.numpy())
    else:
        raise Exception('unsupported dataset')
    data = dataset[0]
    return data

def matrix_cosine_similarity(matrix1, matrix2):
    # 计算L2范数（欧氏距离），广播使得两个矩阵的计算变得一致
    distances = pairwise_distance(matrix1[:, None, :], matrix2, p=2, keepdim=False)

    # 取倒数并进行适当的归一化
    similarities = 1 / (1 + distances)

    return similarities

def add_nodes_with_similar_edges(existing_graph, n, new_node_features, incremental_data_y, k=3):
    x = existing_graph.x.to(args.device)
    edge_index = existing_graph.edge_index.to(args.device)
    new_node_indices = torch.arange(x.size(0), x.size(0) + n, dtype=torch.long).to(args.device)   
    # 将新节点特征添加到原有节点特征中
    x = torch.cat([x, new_node_features.to(args.device)], dim=0)
    # 计算新节点与所有节点（包括其他新节点）之间的相似度
    sim_matrix = matrix_cosine_similarity(x[new_node_indices], x)
    # 选择相似度最高的 k 个节点与新节点建立边关系，排除自身
    _, topk_indices = sim_matrix.topk(k + 1, dim=1, largest=True, sorted=True)  # 多选一个，排除自身
    topk_indices = topk_indices[:, 1:].to(args.device)     # 排除自身后的前 k 个节点
    new_edges_1 = torch.cat([
            topk_indices.reshape(1,n*k)[0],
            new_node_indices.view(-1, 1).repeat(1, k).view(-1),]).view(2,-1).to(args.device)      
    new_edges_2 = torch.cat([
            new_node_indices.view(-1, 1).repeat(1, k).view(-1),
            topk_indices.reshape(1,n*k)[0],]).view(2,-1).to(args.device)
    # 构建新的边索引
    new_edge_index = torch.cat([edge_index, new_edges_1, new_edges_2], dim=1)
    # 标签、train_mask、test_mask
    new_labels = torch.cat((existing_graph.y.to(args.device) , incremental_data_y.to(args.device) ))
    new_train_mask = torch.cat((existing_graph.train_mask.to(args.device) , torch.ones(n, dtype=torch.bool).to(args.device) ))
    new_test_mask = torch.cat((existing_graph.test_mask.to(args.device) , torch.zeros(n, dtype=torch.bool).to(args.device) ))
    # 创建新的 Data 对象
    new_graph = Data(x=x, edge_index=new_edge_index, y=new_labels, train_mask=new_train_mask, test_mask=new_test_mask)
    # 输出每个新节点的邻居节点
    # new_neighbors = {new_node.item(): new_graph.edge_index[1, new_graph.edge_index[0] == new_node].tolist() for i, new_node in enumerate(new_node_indices)}
    new_neighbors = {new_node.item(): new_edges_2[1, new_edges_2[0] == new_node].tolist() for new_node in new_node_indices}

    return new_graph, new_neighbors

def shards_train(args, first_shard, second_shard = None, second = False, Unlearning = False):
    if second == True:
        name = 'second_shard_data'
    else:
        name = 'shard_data'
    # Datasets
    train_data = continuum(root=args.data_root, name=name, data_type='train', download=True, k_hop=args.k, first_shard = first_shard, second_shard = second_shard, Unlearning = Unlearning)
    train_loader = Data_1.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate, drop_last=True)
    ## train_loader加载的数据是一个封装好的东西“<torch.utils.data.dataloader.DataLoader object at 0x7f17aade6070>”
    test_data = continuum(root=args.data_root, name=name, data_type='test', download=True, k_hop=args.k, first_shard = first_shard, second_shard = second_shard, Unlearning = Unlearning)
    test_loader = Data_1.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
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

def tune_train(args, inputs, targets, neighbor, first_shard, second_shard = None, second = False):
    Net = nets[args.model.lower()]
    net = Net(feat_len=inputs.shape[-1], num_class=args.raw_data_num_classes, hidden = args.hidden, dropout = args.drop).to(args.device)
    criterion = nn.CrossEntropyLoss() ## 交叉熵损失
    optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, weight_decay=1e-5)  ## 定义优化算法
    net.load_state_dict(torch.load(args.save + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard), map_location=args.device))  ## 加载模型, args.load这里需要修改，根据碎片的不同，加载不同的模型
    for iter in tqdm.tqdm(range(args.iteration)):
        inputs, targets = inputs.to(args.device), targets.to(args.device)   ##输入和目标
        neighbor = [element.to(args.device) for element in neighbor]
        outputs = net(inputs, neighbor)
        loss = criterion(outputs, targets) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(), args.save + '/' + 'first_' + str(first_shard) +'-second_' + str(second_shard))
    print('Updating shards models, first shard %s , second shard %s' % (first_shard, second_shard))
    if second == True:
        name = 'second_shard_data'
    else:
        name = 'shard_data'
    # Datasets
    train_data = continuum(root=args.data_root, name=name, data_type='train', download=True, k_hop=args.k, first_shard = first_shard, second_shard = second_shard)
    train_loader = Data_1.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=graph_collate, drop_last=True)
    ## train_loader加载的数据是一个封装好的东西“<torch.utils.data.dataloader.DataLoader object at 0x7f17aade6070>”
    test_data = continuum(root=args.data_root, name=name, data_type='test', download=True, k_hop=args.k, first_shard = first_shard, second_shard = second_shard)
    test_loader = Data_1.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=graph_collate, drop_last=True)
    train_acc, test_acc = performance(train_loader, net, args.device, args.k), performance(test_loader, net, args.device, args.k)
    print('train_acc: %.4f, test_acc: %.4f'%(train_acc, test_acc))

if __name__ == '__main__':
    # Arguements
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument("--device", type=str, default='cuda:1', help="cuda or cpu")
    parser.add_argument("--data-root", type=str, default='/home/mjx/GML-master/U+Data_incremental/U+D/Partition/temp_data/processed_data/cora', help="shard data")
    parser.add_argument("--raw-root", type=str, default='/home/mjx/GML-master/U+Data_incremental/U+D/Partition/temp_data/raw_data/', help="raw data")
    parser.add_argument("--centroid_root", type=str, default='/home/mjx/GML-master/U+Data_incremental/U+D/Partition/temp_data/analysis_data/partition/', help="centroids")
    parser.add_argument("--incremental_embedding_root", type=str, default='/home/mjx/GML-master/Regular/Partition/temp_data/processed_data/cora', help="incremental_embedding_root")
    parser.add_argument("--model", type=str, default='attnplain', help="LGL or SAGE")
    parser.add_argument("--dataset", type=str, default='cora', help="cora, citeseer, pubmed")
    parser.add_argument("--load", type=str, default=None, help="load pretrained model file")
    parser.add_argument("--save", type=str, default='/home/mjx/GML-master/U+Data_incremental/U+D/Train_and_test/shards_models', help="model file to save")
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
    parser.add_argument("--eval", type=str, default='/home/mjx/GML-master/U+Data_incremental/U+D/Train_and_test/shards_results', help="the path to eval the acc") ## 评估 acc 的路径
    parser.add_argument("--k", type=int, default=None, help='khop.')
    parser.add_argument("--hidden", type=int, nargs="+", default=[80,80])
    parser.add_argument("--drop", type=float, nargs="+", default=[0.3,0.1])
    parser.add_argument("--first_shards_num", type=int, default=5)
    parser.add_argument("--second_shards_num", type=int, default=2)
    parser.add_argument("--once_incremental_num", type=int, default=20)
    parser.add_argument("--raw_data_num_classes", type=int, default=7)
    parser.add_argument("--iteration", type=int, default=10, help="number of training iteration")
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    ## 训练T0时刻的一级碎片
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
    ## 训练一级碎片中最差的碎片的二级碎片
    second_shard_results = {}
    for second_sh in range(args.second_shards_num):
        print('training target models, first shard %s , second shard %s' % (par_shard, second_sh))
        train_acc, test_acc, train_time = shards_train(args, par_shard, second_sh, second = True)
        print("Model training time: %s" % (train_time)) ## 日志显示训练时间
        second_shard_results[second_sh] = train_acc, test_acc
        print(second_shard_results)
    pickle.dump(T_time, open('/home/mjx/GML-master/U+Data_incremental/U+D/Train_and_test/time.pkl', 'wb'))

    ## 读取各碎片质心
    centroids = pickle.load(open(args.centroid_root + 'first_centroid_bkm_' + args.dataset, 'rb'))
    sec_cen = pickle.load(open(args.centroid_root + 'second_centroid_bkm2_' + args.dataset + str(par_shard), 'rb'))
    centroids[par_shard] = sec_cen.get(0)
    centroids[par_shard + 0.1] = sec_cen.get(1)
    ## 读取增量数据的嵌入和特征
    _, _, data_incremental_indicies = pickle.load(open(args.data_root + '/train_test_split0.1', 'rb'))
    raw_data = load_raw_data(args) 
    incremental_node_to_embedding = pickle.load(open(args.incremental_embedding_root+'/embedding_sage_0', 'rb'))
    embedding = [] ## 建一个嵌入的空集
    feature = []
    labels = []
    for i in range(len(data_incremental_indicies)):
        embedding.append(incremental_node_to_embedding.get(data_incremental_indicies[i]))
        feature.append(raw_data.x[data_incremental_indicies[i], :])
        labels.append(raw_data.y[data_incremental_indicies[i]])

    for incremental_order in range(5):
        ## 遗忘环节
        Unlearning_belong_to_first_shard =  random.randint(0, args.first_shards_num-1) # 4
        print('Unlearning_node_belong_to, first shard %s' % Unlearning_belong_to_first_shard)
        if Unlearning_belong_to_first_shard == par_shard:
            Unlearning_belong_to_second_shard = random.randint(0, args.second_shards_num-1)
            print('training target models, first shard %s , second shard %s' % (par_shard, Unlearning_belong_to_second_shard))
            train_acc, test_acc, train_time = shards_train(args, par_shard, Unlearning_belong_to_second_shard, second = True, Unlearning = True)
            print("Model training time: %s" % (train_time)) ## 日志显示训练时间

        else:
            Unlearning_belong_to_second_shard = None
            print('training target models, first shard %s' % (Unlearning_belong_to_first_shard))
            train_acc, test_acc, train_time = shards_train(args, Unlearning_belong_to_first_shard, Unlearning = True)
            print("Model training time: %s" % (train_time)) ## 日志显示训练时间

        ## 根据增量信息更新碎片图环节
        incremental_data_embedding = embedding[incremental_order*args.once_incremental_num : (incremental_order+1)*args.once_incremental_num]
        incremental_data_feature = feature[incremental_order*args.once_incremental_num : (incremental_order+1)*args.once_incremental_num]
        incremental_data_labels = labels[incremental_order*args.once_incremental_num : (incremental_order+1)*args.once_incremental_num]
        print(incremental_data_labels)
        node_location = [] ## 存成字典？
        location_dict = {}
        for node in incremental_data_embedding:
            node = cp.array(node, dtype=np.float32)
            # 计算节点与各个质心的距离
            distances = {key: cp.linalg.norm(node - value) for key, value in centroids.items()}
            min_distance_key = min(distances, key=distances.get)
            node_location.append(min_distance_key) ## 当值等于par_shard时，该节点所属于的位置就是par_shard下的第一个二级碎片，而par_shard+0.1时属于第二个二级碎片
            min_distance_value = distances[min_distance_key]
            print(f'The organization with the minimum distance is {min_distance_key} with a distance of {min_distance_value}')
        for index, value in enumerate(node_location):
            if value in location_dict.keys():
                location_dict[value].append(index)
            else:
                location_dict[value] = [index]    ## 位置字典，每个碎片中要添加的节点索引已经归类，下一步就是针对性的更新图了
        ## 碎片图更新环节
        first_shard_data_Update = pickle.load(open(args.data_root+'/shard_data_sage_km_5_0.005_0', 'rb'))
        second_shard_data_Update = pickle.load(open(args.data_root+'/shard_data_sage_km_5_0.005_0' + 'second' + '_second_' + str(par_shard), 'rb'))
        for node_location, incremental_index in location_dict.items():
            if not incremental_index:  # 如果值为空，跳过
                continue
            elif node_location == par_shard:
                existing_shard_graph = second_shard_data_Update[0]
                incremental_data_x = torch.stack([incremental_data_feature[i] for i in incremental_index])
                incremental_data_y = torch.stack([incremental_data_labels[i] for i in incremental_index])
                shard_incremental_num = len(incremental_index)
                new_graph, new_neighbors = add_nodes_with_similar_edges(existing_shard_graph, shard_incremental_num, incremental_data_x, incremental_data_y)
                new_neighbors_feature = [new_graph.x[value].unsqueeze(1) for value in new_neighbors.values()]
                tune_train(args, incremental_data_x.unsqueeze(1), incremental_data_y, new_neighbors_feature, par_shard, second_shard = 0, second = True)
                second_shard_data_Update[0] = new_graph
            elif node_location == par_shard + 0.1:
                existing_shard_graph = second_shard_data_Update[1]
                incremental_data_x = torch.stack([incremental_data_feature[i] for i in incremental_index])
                incremental_data_y = torch.stack([incremental_data_labels[i] for i in incremental_index])
                shard_incremental_num = len(incremental_index)
                new_graph, new_neighbors = add_nodes_with_similar_edges(existing_shard_graph, shard_incremental_num, incremental_data_x, incremental_data_y)
                new_neighbors_feature = [new_graph.x[value].unsqueeze(1) for value in new_neighbors.values()]
                tune_train(args, incremental_data_x.unsqueeze(1), incremental_data_y, new_neighbors_feature, par_shard, second_shard = 1, second = True)
                second_shard_data_Update[1] = new_graph
            else:
                existing_shard_graph = first_shard_data_Update[node_location]
                incremental_data_x = torch.stack([incremental_data_feature[i] for i in incremental_index])
                incremental_data_y = torch.stack([incremental_data_labels[i] for i in incremental_index])
                shard_incremental_num = len(incremental_index)
                new_graph, new_neighbors = add_nodes_with_similar_edges(existing_shard_graph, shard_incremental_num, incremental_data_x, incremental_data_y)
                new_neighbors_feature = [new_graph.x[value].unsqueeze(1) for value in new_neighbors.values()]
                tune_train(args, incremental_data_x.unsqueeze(1), incremental_data_y, new_neighbors_feature, node_location, second_shard = None)
                first_shard_data_Update[node_location] = new_graph
        pickle.dump(first_shard_data_Update, open(args.data_root+'/shard_data_sage_km_5_0.005_0', 'wb'))
        pickle.dump(second_shard_data_Update, open(args.data_root+'/shard_data_sage_km_5_0.005_0' + 'second' + '_second_' + str(par_shard), 'wb'))
        ## 训练增量