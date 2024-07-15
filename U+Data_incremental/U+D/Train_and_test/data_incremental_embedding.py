import os
import pickle
import logging
import shutil

import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Coauthor
import torch_geometric.transforms as T

def load_raw_data(dataset_name):
    if dataset_name in ["cora", "pubmed", "citeseer"]:
        dataset = Planetoid('/home/mjx/GML-master/U+Data_incremental/U+D/Partition/temp_data/raw_data/', dataset_name, transform=T.NormalizeFeatures())
        # labels = np.unique(dataset.data.y.numpy())
    else:
        raise Exception('unsupported dataset')
    data = dataset[0]
    return data

root = '/home/mjx/GML-master/U+Data_incremental/U+D/Partition/temp_data/processed_data/cora'
train_indices, test_indices, data_incremental_indicies = pickle.load(open(root+'/train_test_split0.1', 'rb'))
print(data_incremental_indicies)
data = load_raw_data("cora") 
print(data)
data.incremental_mask = torch.from_numpy(np.isin(np.arange(data.num_nodes), data_incremental_indicies))
print(data)
print(data.x[data.incremental_mask].size())
incremental_embedding_root = '/home/mjx/GML-master/Regular/Partition/temp_data/processed_data/cora'
node_to_embedding = pickle.load(open(incremental_embedding_root+'/embedding_sage_0', 'rb')) ## 2166 是训练图里面的节点的嵌入
# print(node_to_embedding)
print(data.y[data_incremental_indicies])
embedding = [] ## 建一个嵌入的空集
feature = []
incremental_labels = []
for i in range(len(data_incremental_indicies)):
    embedding.append(node_to_embedding.get(data_incremental_indicies[i]))
    feature.append(data.x[data_incremental_indicies[i], :])
print(len(embedding))
print(len(feature))
print(feature)
print(embedding)

for incremental_order in range((len(data_incremental_indicies)//20)):
    incremental_data_embedding = embedding[incremental_order*20 : (incremental_order+1)*20]
    #print(len(incremental_data_embedding))
    #print(incremental_order)


centroids = pickle.load(open('/home/mjx/GML-master/U+Data_incremental/U+D/Partition/temp_data/analysis_data/partition/first_centroid_bkm_cora', 'rb'))
print(centroids)
sec_cen = pickle.load(open('/home/mjx/GML-master/U+Data_incremental/U+D/Partition/temp_data/analysis_data/partition/second_centroid_bkm2_cora1', 'rb'))
print(sec_cen)
centroids[1.0] = sec_cen.get(0)
centroids[1.1] = sec_cen.get(1)
print(centroids)

import cupy as cp

# 定义节点
node = cp.array(embedding[0], dtype=np.float32)  # 请将 "your_values_here" 替换为实际的节点值

# 计算节点与各个质心的距离
distances = {key: cp.linalg.norm(node - value) for key, value in centroids.items()}

# 找到距离最小的组织
min_distance_key = min(distances, key=distances.get)
min_distance_value = distances[min_distance_key]

# 打印结果
print(f'The organization with the minimum distance is {min_distance_key} with a distance of {min_distance_value}')


my_list = [1, 2, 3, 2, 4, 1, 5, 6, 4]

# 创建一个空字典
my_dict = {}

# 遍历列表
for index, value in enumerate(my_list):
    # 如果值在字典中已经存在，将当前索引添加到对应值的列表中
    if value in my_dict:
        my_dict[value].append(index)
    else:
        # 如果值在字典中不存在，创建一个新的键值对
        my_dict[value] = [index]

# 打印结果
print(my_dict)

import torch
from torch.nn.functional import pairwise_distance

def matrix_cosine_similarity(matrix1, matrix2):
    # 计算L2范数（欧氏距离），广播使得两个矩阵的计算变得一致
    distances = pairwise_distance(matrix1[:, None, :], matrix2, p=2, keepdim=False)

    # 取倒数并进行适当的归一化
    similarities = 1 / (1 + distances)

    return similarities


import torch
from torch_geometric.data import Data
from torch.nn.functional import cosine_similarity

def add_nodes_with_similar_edges(existing_graph, n, new_node_features, k=3):
    x = existing_graph.x
    edge_index = existing_graph.edge_index

    new_node_indices = torch.arange(x.size(0), x.size(0) + n, dtype=torch.long)

    # 将新节点特征添加到原有节点特征中
    x = torch.cat([x, new_node_features], dim=0)

    # 计算新节点与所有节点（包括其他新节点）之间的相似度
    sim_matrix = matrix_cosine_similarity(x[new_node_indices], x)

    # 选择相似度最高的 k 个节点与新节点建立边关系，排除自身
    print(sim_matrix)
    _, topk_indices = sim_matrix.topk(k + 1, dim=1, largest=True, sorted=True)  # 多选一个，排除自身
    print(topk_indices)
    topk_indices = topk_indices[:, 1:]  # 排除自身后的前 k 个节点
    print(topk_indices)
    print(new_node_indices.view(-1, 1).repeat(1, k).view(-1))
    print(topk_indices.reshape(1,n*k)[0])

    new_edges_1 = torch.cat([
            topk_indices.reshape(1,n*k)[0],
            new_node_indices.view(-1, 1).repeat(1, k).view(-1),]).view(2,-1)
        
    new_edges_2 = torch.cat([
            new_node_indices.view(-1, 1).repeat(1, k).view(-1),
            topk_indices.reshape(1,n*k)[0],]).view(2,-1)
    # 构建新的边索引
    new_edge_index = torch.cat([edge_index, new_edges_1, new_edges_2], dim=1)
    print(new_edge_index)

    # 创建新的 Data 对象
    new_graph = Data(x=x, edge_index=new_edge_index)

    # 输出每个新节点的邻居节点
    # new_neighbors = {new_node.item(): new_graph.edge_index[1, new_graph.edge_index[0] == new_node].tolist() for i, new_node in enumerate(new_node_indices)}
    new_neighbors = {new_node.item(): new_edges_2[1, new_edges_2[0] == new_node].tolist() for new_node in new_node_indices}

    return new_graph, new_neighbors

# 例子
existing_graph = Data(x=torch.rand(5, 10), edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]]))
n = 3
new_node_features = torch.rand(n, 10)

new_graph, new_neighbors = add_nodes_with_similar_edges(existing_graph, n, new_node_features)
print(new_graph)
print("新节点邻居：")
print(new_neighbors)

my_dict = {'a': [1, 2, 3], 'b': [], 'c': [4, 5]}

for key, value in my_dict.items():
    if not value:  # 如果值为空，跳过
        continue
    
    # 对于不为空的值，执行操作 A
    # 例如，打印键和对应的值
    print(f"Key: {key}, Value: {value}")
    # 在这里添加你的操作 A 的代码


import torch
from torch_geometric.data import Data

# 假设 data 是你的图数据，data.x 是一个 (2000, 12) 的张量
# 这里使用随机数据作为示例
data = Data(x=torch.rand(2000, 12))

# 切片前 20 个节点
num_nodes_to_keep = 20
sliced_data = Data(x=data.x[:num_nodes_to_keep])

# 将每个节点的特征存储在列表中
node_feature_list = [sliced_data.x[i] for i in range(sliced_data.x.size(0))]

# 随机抽取三个节点的特征
num_nodes_to_sample = 3
random_indices = torch.randperm(len(node_feature_list))[:num_nodes_to_sample]
print(random_indices)
sampled_features = [node_feature_list[i] for i in random_indices]
labels = [torch.tensor([1]), torch.tensor([1]), torch.tensor([5]), torch.tensor([2])]
index_la = [1, 2, 3]
labels_a = [labels[i] for i in index_la]
print(torch.cat(labels_a))
# 输出抽取的三个节点的特征矩阵
print("Sampled Features Matrix:")
print(torch.stack(sampled_features))

first_shard_data = pickle.load(open(root+'/shard_data_sage_km_5_0.005_0', 'rb'))
A = torch.cat([first_shard_data[0].train_mask, torch.tensor([True, True])])
print(A.size())

S = {5: [0, 6, 1], 6: [0, 5, 1], 7: [5, 3, 4]}
values = [value for value in S.values()]
print(values)

import torch

# 假设原始张量形状为 (batch_size, channels, height, width)
# 这里创建一个示例张量
original_tensor = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])

# 获取原始张量的形状
original_shape = original_tensor.size()

# 添加 Batch 维度并进行 reshape
batched_tensor = original_tensor.view(1, *original_shape)

# 打印结果
print("Original Tensor Shape:", original_shape)
print("Batched Tensor Shape:", batched_tensor.size())

import torch

# 例子中的两个零维张量
tensor1 = torch.tensor(4)
tensor2 = torch.tensor(4)

# 使用 torch.stack 将零维张量转换为一维张量
result = torch.stack([tensor1, tensor2])

# 打印结果
print(result)
