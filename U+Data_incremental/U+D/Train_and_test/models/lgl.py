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

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nf
from models.KTransCat import KTransCAT

from models.layer import FeatBrd1d, FeatTrans1d, FeatTransKhop, FeatTransKCat, FeatTransKhop, Mlp, AttnFeatTrans1d, AttnFeatTrans1dSoft


class LGL(nn.Module):
    def __init__(self, feat_len, num_class, hidden = [64, 32], dropout = [0,0]):
        ## the Flag ismlp will encode without neighbor 标志是mlp将在没有邻居的情况下进行编码
        super(LGL, self).__init__()
        c = [1, 4, hidden[1]] ## 通道数
        f = [feat_len, int(hidden[0]/c[1]), 1] ## 特征维数

        self.feat1 = FeatTrans1d(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1])  ## 特征转换层
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign(),  nn.Dropout(p=dropout[0])) ## 激活函数层
        self.feat2 = FeatTrans1d(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign(), nn.Dropout(p=dropout[1]))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c[2]*f[2], num_class)) ## 分类层


    def forward(self, x, neighbor):

        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)


class AFGN(nn.Module): ## 特征邻接矩阵中有权重的 与LGL的唯一区别在特征转换函数上使用了AttnFeatTrans1dSoft()
    def __init__(self, feat_len, num_class, hidden = [64, 32], dropout = [0,0]):
        ## the Flag ismlp will encode without neighbor
        super(AFGN, self).__init__()
        c = [1, 4, hidden[1]]
        f = [feat_len, int(hidden[0]/c[1]), 1]
        self.feat1 = AttnFeatTrans1dSoft(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1])
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign(),  nn.Dropout(p=dropout[0]))
        self.feat2 = AttnFeatTrans1dSoft(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2])
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign(),  nn.Dropout(p=dropout[1]))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(c[2]*f[2], num_class))

    def forward(self, x, neighbor):

        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [self.acvt1(n) for n in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)


class KCAT(nn.Module): ## 不断拼接的版本，感觉起来不太靠谱
    '''
    TODO the locals or the __dict__ cause some issue for net.to(device), the weight of feat didn't load to cuda
    Concate the k level in the last classifier layer
    '''
    def __init__(self, feat_len, num_class, k=1, device='cuda:0'):
        super(KCAT, self).__init__()
        self.k = k
        self.device = device
        c = [1, 4, 32]
        f = [feat_len, 16, 1]
        for k in range(k):
            self.__dict__["feat1%i"%k] = FeatTrans1d(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1]).to(device)
            self.__dict__["acvt1%i"%k]= nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign()).to(device)
            self.__dict__["feat2%i"%k] = FeatTrans1d(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2]).to(device)
            self.__dict__["acvt2%i"%k] = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign()).to(device)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.1), nn.Linear(c[2]*f[2]*self.k, c[2]*f[2]),nn.ReLU(),nn.Dropout(p=0.1),nn.Linear(c[2]*f[2], num_class))

    def forward(self, x, neighbor):
        x_khops = torch.FloatTensor().to(self.device)
        for k in range(self.k):
            kneighbor = [i[k] for i in neighbor]
            khop, kneighbor = self.__dict__["feat1%i"%k](x, [i[k] for i in neighbor])
            khop, kneighbor = self.__dict__["acvt1%i"%k](khop), [self.__dict__["acvt1%i"%k](item) for item in kneighbor]
            khop, kneighbor = self.__dict__["feat2%i"%k](khop, kneighbor)
            khop = self.__dict__["acvt2%i"%k](khop)
            x_khops = torch.cat((x_khops, khop), dim = 1)
        return self.classifier(x_khops)


class KLGL(nn.Module): ## 多阶邻居的LGL版本，使用的底层函数为FeatTransKhop
    def __init__(self, feat_len, num_class, k=1):
        super(KLGL, self).__init__()
        # x: (N,f); adj:(N, k, f, f)
        c = [1, 4, 32]
        f = [feat_len, 16, 1]
        self.feat1 = FeatTransKhop(in_channels=c[0], in_features=f[0], out_channels=c[1], out_features=f[1], khop = k)
        self.acvt1 = nn.Sequential(nn.BatchNorm1d(c[1]), nn.Softsign())
        self.feat2 = FeatTransKhop(in_channels=c[1], in_features=f[1], out_channels=c[2], out_features=f[2], khop = k)
        self.acvt2 = nn.Sequential(nn.BatchNorm1d(c[2]), nn.Softsign())
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p=0.1), nn.Linear(c[2]*f[2], num_class))

    def forward(self, x, neighbor):
        x, neighbor = self.feat1(x, neighbor)
        x, neighbor = self.acvt1(x), [[self.acvt1(k) for k in item] for item in neighbor]
        x, neighbor = self.feat2(x, neighbor)
        x = self.acvt2(x)
        return self.classifier(x)


class LifelongRehearsal(nn.Module):
    def __init__(self, args, BackBone, feat_len, num_class, k = None, hidden = [64,32], drop = [0,0]):
        super(LifelongRehearsal, self).__init__()
        self.args = args ## 将定义域里的固定超参传承过来
        if not k:
            self.backbone = BackBone(feat_len, num_class, hidden = hidden, dropout = drop)  ## backbone是通常的一些CNN骨干框架，类似VGG
        else:
            self.backbone = BackBone(feat_len, num_class, k=k, hidden = hidden, dropout = drop)
        self.backbone = self.backbone.to(args.device)
        self.register_buffer('adj', torch.zeros(1, feat_len, feat_len))  ## register_buffer 这个方法的作用在于定义一组参数，特殊点在于训练模型中register_buffer 定义的参数不能被更新
        self.register_buffer('inputs', torch.Tensor(0, 1, feat_len))
        self.register_buffer('targets', torch.LongTensor(0))  ## 定义参数张量，无实质性意义，之后学习的那些会填充进来
        self.neighbor = []  ## 这里是个空集合
        self.sample_viewed = 0  ## 每次增加的采样量
        self.memory_order = torch.LongTensor() ## 采样序列
        self.memory_size = self.args.memory_size ## 采样的大小
        self.criterion = nn.CrossEntropyLoss() ## 交叉熵损失
        self.running_loss = 0 ## 对损失0初始化
        exec('self.optimizer = torch.optim.%s(self.parameters(), lr=%f)'%(args.optm, args.lr)) ## 根据定义好的优化器，以及学习率

    def forward(self, inputs, neighbor):
        return self.backbone(inputs, neighbor) ## 返回骨干网络的输出

    def observe(self, inputs, targets, neighbor, replay=True): # 观察函数, 本质上就是整个训练过程的整合
        self.train() ## 定义训练函数
        self.sample(inputs, targets, neighbor) ## 根据定义的采样函数，采样到进行训练的样本
        
        for i in range(self.args.iteration):  ## 根据迭代次数循环
            self.optimizer.zero_grad()
            inputs, targets, neighbor = self.todevice(inputs, targets, neighbor, device = self.args.device)
            outputs = self.forward(inputs, neighbor) ## 骨干网络的输出
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        self.running_loss+=loss

        if replay: ## 这个reply是干嘛的？ 在下次迭代前把输入重新打乱，进行训练，相当于二次训练(补充训练)
            L = torch.randperm(self.inputs.size(0)) ## 对输入的行维度进行(也就是节点维度)随机打乱
            minibatches = [L[n:n+self.args.batch_size] for n in range(0, len(L), self.args.batch_size)] ## 根据小批量batch大小分割
            for index in minibatches:
                self.optimizer.zero_grad()
                inputs, targets, neighbor = self.inputs[index], self.targets[index], [self.neighbor[i] for i in index.tolist()]
                inputs, targets, neighbor = self.todevice(inputs, targets, neighbor, device = self.args.device)
                outputs = self.forward(inputs, neighbor)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def todevice(self, inputs, targets, neighbor, device="cpu"):
        inputs, targets = inputs.to(device), targets.to(device)
        ## take the neighbor with k
        if not self.args.k:
            neighbor = [element.to(device) for element in neighbor] ## 摄取邻居节点信息上cuda
        else:
            neighbor = [[element.to(device) for element in item]for item in neighbor]
        return inputs, targets, neighbor

    @torch.no_grad() ## 此刻定义的函数不参与梯度计算
    def uniform_sample(self, inputs, targets, neighbor): ## 规则采样
        inputs, targets, neighbor = self.todevice(inputs, targets, neighbor)
        self.inputs = torch.cat((self.inputs, inputs), dim=0)
        self.targets = torch.cat((self.targets, targets), dim=0)
        self.neighbor += neighbor

        if self.inputs.size(0) > self.args.memory_size:
            idx = torch.randperm(self.inputs.size(0))[:self.args.memory_size]
            self.inputs, self.targets = self.inputs[idx], self.targets[idx]
            self.neighbor = [self.neighbor[i] for i in idx.tolist()]

    @torch.no_grad() ## 这两种采样方式用在哪里？observe函数，训练
    def sample(self, inputs, targets, neighbor):
        self.sample_viewed += inputs.size(0) ## inputs.size(0) 是10； 每次增加10个
        self.memory_order += inputs.size(0)# increase the order 

        self.targets = torch.cat((self.targets, targets), dim=0)
        self.inputs = torch.cat((self.inputs,inputs), dim = 0) ## 第0个维度不断拼接10-20-30-40；inputs.size()一直是(10,1,1433)
        self.memory_order = torch.cat((self.memory_order, torch.LongTensor(list(range(inputs.size()[0]-1,-1,-1)))), dim = 0)# for debug
        ## print(range(inputs.size()[0]-1,-1,-1)) 输出(9, -1, -1), 不断增加的缓存序列
        self.neighbor += neighbor  ## self.neighbor 是一个list； 每次都把一个邻接矩阵加到列表中

        node_len = int(self.inputs.size(0)) ## 节点数量
        ext_memory = node_len - self.memory_size ## print(ext_memory) 当前的节点数量与缓存量之间的对比，数值变化为-90~-80~...~10~10~...
        if ext_memory > 0:  ## 当节点当前节点数量大于缓存大小时，需要选择保留的规定数量(self.memory_size)节点
            mask = torch.zeros(node_len,dtype = bool) # mask inputs order targets and neighbor 制作节点数量大小的掩码
            reserve = self.memory_size # reserved memrory to be stored 保留要存储的缓存
            ## print(self.sample_viewed, self.sample_viewed/ext_memory) ## 从110开始，每次加10，除以之后就是11开始，每次加1
            seg = np.append(np.arange(0,self.sample_viewed,self.sample_viewed/ext_memory),self.sample_viewed)
            ## np.arange(0,self.sample_viewed,self.sample_viewed/ext_memory) 起始为0，终止为总体的样本量，每次的步长为除以10之后的数
            ## seg应当是一个集合，是从零开始按步长至终止的数值。[  0.  55. 110. 165. 220. 275. 330. 385. 440. 495. 550.]
            ## range(len(seg)-2,-1,-1) 是(9,-1,-1)
            for i in range(len(seg)-2,-1,-1):
                left = self.memory_order.ge(np.ceil(seg[i]))*self.memory_order.lt(np.floor(seg[i+1]))
                ## torch.ge()，用于比较memory_order和seg[i]中最大值的，目的在于整理出正在经历循环的区间
                ## torch.lt(), 也是比较，相乘表示“且”
                ## print(left) 写满了false和true的tensor
                leftindex = left.nonzero()  ## 列出满足条件的索引
                if leftindex.size()[0] > reserve/(i+1): # the quote is not enough, need to be reduced 不能大于要保留的组数
                    leftindex = leftindex[torch.randperm(leftindex.size()[0])[:int(reserve/(i+1))]] # reserve the quote
                    mask[leftindex] = True
                else:
                    mask[leftindex] = True # the quote is enough
                reserve -= leftindex.size()[0] # deducte the quote
            self.inputs = self.inputs[mask] 
            ## print(mask)  现在,当mask=False或mask=0 ,字面意思是不要将此值标记为无效。 
            ## 简而言之,在计算时将其包括在内 。 同样, mask=True或mask=1表示将此值标记为无效
            self.targets = self.targets[mask]
            self.memory_order = self.memory_order[mask]
            self.neighbor = [self.neighbor[i] for i in mask.nonzero()] ## 在剩下的点里面去选择邻居节点
