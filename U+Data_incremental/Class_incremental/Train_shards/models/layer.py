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
import torch
import os.path
import torch.nn as nn


class FeatBrd1d(nn.Module): ## 简单组件（关于特征传播，利用特征邻接矩阵）
    '''
    Feature Broadcasting Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, n_features)
    Output size is (n_batch, out_channels, n_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        adjacency (Tensor): feature adjacency matrix
    '''
    def __init__(self, in_channels, out_channels, adjacency=None):
        super(FeatBrd1d, self).__init__()
        self.register_buffer('adjacency', adjacency)
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, adj=None):
        if adj is not None:
            return (adj.unsqueeze(1) @ (self.conv(x).unsqueeze(-1))).view(x.size(0),-1,x.size(-1))
        else:
            return (self.adjacency @ (self.conv(x).unsqueeze(-1))).view(x.size(0),-1,x.size(-1))


class Mlp(nn.Module): ## 1d卷积  MLP的简单组件
    def __init__(self, in_channels, in_features, out_channels, out_features):
        super(Mlp, self).__init__()
        self.out_channels, self.out_features = out_channels, out_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels*out_features, kernel_size=in_features, bias=False) 
        
    def forward(self, x):
        '''
        Current set up address for neighbor c = 1
        adj: (N,c,f,f)
        x: (N,c,f)
        mm(N,c,f,f @ N,c,f,1)->(N,c_in,f,1); view->(N,c_in,f)
        W: (f,c_out*f_out)
        '''
        x = self.conv(x)
        x = x.view(x.size(0), self.out_channels, self.out_features)
        return x

class FeatTrans1d(nn.Module): ## Feature Transforming Layer 特征转换层 LGL的
    '''
    Feature Transforming Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, in_features, out_channels, out_features):
        super(FeatTrans1d, self).__init__()
        self.out_channels, self.out_features = out_channels, out_features
        # Taking advantage of parallel efficiency of nn.Conv1d 运用卷积并行效率高得优势，采取1d卷积
        self.conv = nn.Conv1d(in_channels, out_channels*out_features, kernel_size=in_features, bias=False)

    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor) ## 8a 根据中心节点及其邻居信息制作特征邻接矩阵
        x = self.transform(x, adj) ## 文章中的8b 对中心节点的特征进行传播
        neighbor = [self.transform(neighbor[i], adj[i:i+1]) for i in range(x.size(0))] ## 文章中的8c 对邻居的信息进行更新
        return x, neighbor

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,c,f,f)
        x: (N,c,f)
        mm(N,c,f,f @ N,c,f,1)->(N,c_in,f,1); view->(N,c_in,f)
        W: (c_out*f_out,f)
        '''
        return self.conv((adj @ x.unsqueeze(-1)).squeeze(-1)).view(x.size(0), self.out_channels, self.out_features) ## 整体的传播过程

    def feature_adjacency(self, x, y): ## 特征邻接矩阵的定义方法
        fadj = torch.stack([torch.einsum('ca,ncb->cab', x[i], y[i]) for i in range(x.size(0))]) ## torch.einsum函数表示矩阵相乘的 torch.stack函数用于堆叠所有节点的特征邻接矩阵
        ## 可是权重W呢？这个层函数里面并没有，所以这个邻接矩阵的形成完全是根据特征之间的相似性形成的。所以精度本身会下降！
        LLLL = fadj.clone()
        fadj += LLLL.transpose(-2, -1) ## 保证无向图的对称性
        return self.row_normalize(self.sgnroot(fadj)) ## 公式4和公式5

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0 ## torch.isnan(x) 矩阵x中nan的位置索引
        return x


class FeatTransKhop(nn.Module):
    '''
    Feature Transforming Layer for K-hop multi-channel 1D features.
    Taking K as channels and encode the k-hop into the channel of the nodex
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    Args:
        in_channels (int): number of feature input channels
        k-hop (int): number of the k level.
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features

    '''
    def __init__(self, in_channels, khop, in_features, out_channels, out_features):
        super(FeatTransKhop, self).__init__()
        self.khop = khop
        self.out_channels, self.out_features = out_channels, out_features
        self.in_channels, self.in_features = khop*in_channels, in_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(self.in_channels, out_channels*out_features, kernel_size=in_features, bias=False)

    def forward(self, x, neighbor):
        '''
        x: (N,c_in,f) c_in of level 1 is 1
        W: (c_in, c_out*f_out, f)
        neighbor: (N,k,n,f) N and k is list
        adj: (N,k,f,f)
        '''
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        neighbor = [[self.transform(neighbor[i][k], adj[i:i+1]) for k in range(self.khop)] for i in range(x.size(0))] ## 多了对K阶邻居的考虑
        return x, neighbor

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,k,c,f,f)
        x: (N,c_in,f)
        mm(adj: N,k,c_in,f,f @ x: N,1,c_in,f,1)->(N,k,c_in,f,1); view->(N,k*c_in,f)
        W: (c_out*f_out,f)
        '''
        ### TODO current adj is symetric TODO try conv2d
        #(adj@ x.unsqueeze(-1).unsqueeze(1)).view(x.size(0), self.in_channels, self.in_features)
        return self.conv(torch.einsum('nkcaf,ncf->nkca', adj, x)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        '''
        x:(N,c,f), y:(N,k,n,c,f) -> Nkcff where k and N is for list
        k is k-hop, N is batch number, n is neibor number
        fadj: (N,k,c,f,f)
        '''
        fadj = torch.stack([torch.stack([torch.einsum('ca,ncb->cab', x[i], y[i][k]) for k in range(self.khop)]) for i in range(x.size(0))])
        fadj += fadj.transpose(-2, -1) # swap the last two channels -> Nkba
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        ## TODO row dimension to -1 or -2
        x = x / (x.abs().sum(-2, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


class FeatTransKCat(nn.Module):
    '''
    Feature Transforming Layer for K-hop caten 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    This will encode the k-hop into the channel of the nodex
    Args:
        in_channels (int): number of feature input channels
        k-hop (int): number of the k level.
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, khop, in_features, out_channels, out_features):
        super(FeatTransKCat, self).__init__()
        self.khop = khop
        self.out_channels, self.out_features = out_channels, out_features
        self.in_channels, self.in_features = in_channels*2, in_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(self.in_channels, out_channels*out_features, kernel_size = self.in_features, bias=False)

    def forward(self, x, neighbor):
        '''
        x: (N,c_in,f) c_in of level 1 is 1
        W: (k,c_out*k,f,f)
        neighbor: (N,k,n,f) N and k is list
        adj: (N,k,f,f)
        '''
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj) ## 并没有考虑相应的邻居的特征更新
        return x, adj

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,c,f,f)
        x: (N,c,f)
        neighbor: (n,f)
        adj: (N,c,f,f)
        mm(N,c,f,f @ N,c_in,f,1)->(N,c_in,f,1); view->(N,c_in,f)
        W: (c_out*f)
        '''
        ## Cat or plus
        #x = adj @ x.unsqueeze(-1) + x.unsqueeze(-1) 与之前的区别在于对特征传播的利用上加了跳跃连接
        x = torch.cat((adj @ x.unsqueeze(-1), x.unsqueeze(-1)), dim = 1)
        return self.conv(x.view(x.size(0), self.in_channels, self.in_features)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        '''
        x:(N,c,f), y:(N,n,c,f) -> Ncff where N is for list
        k is k-hop, N is batch number, n is neibor number
        fadj: (N,c,f,f)
        '''
        fadj = torch.stack([torch.einsum('ca,ncb->cab', x[i], y[i]) for i in range(x.size(0))]) ## 也没有权重
        LLLL = fadj.clone()
        fadj += LLLL.transpose(-2, -1) # swap the last two channels -> Nkba
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        ## TODO row dimension to -1 or -2
        x = x / (x.abs().sum(-2, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x
    
    
class AttnFeatTrans1d(nn.Module):
    '''
    Feature Transforming Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, in_features, out_channels, out_features):
        super(AttnFeatTrans1d, self).__init__()
        self.out_channels, self.out_features = out_channels, out_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels*out_features, kernel_size=in_features, bias=False)
        self.att1 = nn.Linear(in_features, in_channels, bias=False)
        self.att2 = nn.Linear(in_features, in_channels, bias=False) ## x和y的自注意权重
        
    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        neighbor = [self.transform(neighbor[i], adj[i:i+1]) for i in range(x.size(0))]
        return x, neighbor

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,c,f,f)
        x: (N,c,f)
        mm(N,c,f,f @ N,c,f,1)->(N,c_in,f,1); view->(N,c_in,f)
        W: (c_out*f_out,f)
        '''
        return self.conv((adj @ x.unsqueeze(-1)).squeeze(-1)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        B, C, F = x.shape
        w = [torch.einsum('ci,nkl->n', self.att1(x[i]), self.att2(y[i])) for i in range(B)] ## 这是与FeatTrans1d不同的地方
        fadj = torch.stack([torch.einsum("ca, ncb, n -> cab", x[i], y[i], w[i]) for i in range(B)])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x



class AttnFeatTrans1dSoft(nn.Module):
    '''
    Feature Transforming Layer for multi-channel 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    Args:
        in_channels (int): number of feature input channels
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, in_features, out_channels, out_features):
        super(AttnFeatTrans1dSoft, self).__init__()
        self.out_channels, self.out_features = out_channels, out_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(in_channels, out_channels*out_features, kernel_size=in_features, bias=False)
        self.att1 = nn.Linear(in_features, in_channels, bias=False)
        self.att2 = nn.Linear(in_features, in_channels, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=0)) ## soft版多了一个归一化
        
    def forward(self, x, neighbor):
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        neighbor = [self.transform(neighbor[i], adj[i:i+1]) for i in range(x.size(0))]
        return x, neighbor

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,c,f,f)
        x: (N,c,f)
        mm(N,c,f,f @ N,c,f,1)->(N,c_in,f,1); view->(N,c_in,f)
        W: (c_out*f_out,f)
        '''
        return self.conv((adj @ x.unsqueeze(-1)).squeeze(-1)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        B, C, F = x.shape
        w = [self.norm(torch.einsum('ci,nkl->n', self.att1(x[i]), self.att2(y[i]))) for i in range(B)] ## Soft版比普通的注意力系数多了一个归一化
        fadj = torch.stack([torch.einsum("ca, ncb, n -> cab", x[i], y[i], w[i]) for i in range(B)])
        LLLL = fadj.clone()
        fadj += LLLL.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        x = x / (x.abs().sum(1, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x



    
class AttnFeatTransKCat(nn.Module):
    '''
    Feature Transforming Layer for K-hop caten 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    This will encode the k-hop into the channel of the nodex
    Args:
        in_channels (int): number of feature input channels
        k-hop (int): number of the k level.
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, khop, in_features, out_channels, out_features):
        super(AttnFeatTransKCat, self).__init__()
        self.khop = khop
        self.out_channels, self.out_features = out_channels, out_features
        self.in_channels, self.in_features = in_channels*2, in_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(self.in_channels, out_channels*out_features, kernel_size = self.in_features, bias=False)
        self.att1 = nn.Linear(in_features, in_channels, bias=False)
        self.att2 = nn.Linear(in_features, in_channels, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=0))

    def forward(self, x, neighbor):
        '''
        x: (N,c_in,f) c_in of level 1 is 1
        W: (k,c_out*k,f,f)
        neighbor: (N,k,n,f) N and k is list
        adj: (N,k,f,f)
        '''
        adj = self.feature_adjacency(x, neighbor)
        x = self.transform(x, adj)
        return x, adj

    def transform(self, x, adj):
        '''
        Current set up address for neighbor c = 1
        adj: (N,c,f,f)
        x: (N,c,f)
        neighbor: (n,f)
        adj: (N,c,f,f)
        mm(N,c,f,f @ N,c_in,f,1)->(N,c_in,f,1); view->(N,c_in,f)
        W: (c_out*f)
        '''
        ## Cat or plus
        #x = adj @ x.unsqueeze(-1) + x.unsqueeze(-1)
        x = torch.cat((adj @ x.unsqueeze(-1), x.unsqueeze(-1)), dim = 1)
        return self.conv(x.view(x.size(0), self.in_channels, self.in_features)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        B, C, F = x.shape
        w = [self.norm(torch.einsum('ci,nkl->n', self.att1(x[i]), self.att2(y[i]))) for i in range(B)]
        fadj = torch.stack([torch.einsum("ca, ncb, n -> cab", x[i], y[i], w[i]) for i in range(B)])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        ## TODO row dimension to -1 or -2
        x = x / (x.abs().sum(-2, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x


class AttnFeatTransAPP(nn.Module):
    '''
    Feature Transforming Layer for K-hop caten 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    This will encode the k-hop into the channel of the nodex
    Args:
        in_channels (int): number of feature input channels
        k-hop (int): number of the k level.
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, khop, in_features, out_channels, out_features, alpha=0.1):
        super(AttnFeatTransAPP, self).__init__()
        self.khop = khop
        self.alpha = alpha ## APPNP中的传输率
        self.actv = nn.Sequential(nn.BatchNorm1d(out_channels), nn.Softsign())
        self.out_channels, self.out_features = out_channels, out_features
        self.in_channels, self.in_features = in_channels, in_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(self.in_channels, out_channels*out_features, kernel_size = self.in_features, bias=False)
        self.att1 = nn.Linear(out_features, out_channels, bias=False)
        self.att2 = nn.Linear(out_features, out_channels, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=0))

    def forward(self, x, neighbor):

        h = self.trans(x)
        neighbor = [self.trans(n) for n in neighbor]
        
        adj = self.feature_adjacency(h, neighbor)
        x = self.APP(h, adj, h)
        #neighbor = [self.APP(neighbor[i], adj[i:i+1], neighbor[i]) for i in range(x.size(0))] ## TODO remove this?
        neighbor_app = [(adj[i:i+1] @ neighbor[i].unsqueeze(-1)).squeeze(-1) for i in range(x.size(0))]
        
        x, neighbor_app = self.actv(x), [self.actv(n) for n in neighbor_app]
        
        adj = self.feature_adjacency(x, neighbor_app)
        x = self.APP(x, adj, h)
        #neighbor = [self.APP(neighbor_app[i], adj[i:i+1], neighbor[i]) for i in range(x.size(0))] ## TODO remove this?
        neighbor = [(adj[i:i+1] @ neighbor[i].unsqueeze(-1)).squeeze(-1) for i in range(x.size(0))]
        return x, neighbor ## 有邻居特征的更新
    
    def APP(self,x, adj, h):
        return (1-self.alpha) * (adj @ x.unsqueeze(-1)).squeeze(-1) + self.alpha * h ## 与之前普通的AXW的更新有区别，使用了APPNP的更新方式
    
    def trans(self, x):
        return self.conv(x.view(x.size(0), self.in_channels, self.in_features)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        B, C, F = x.shape
        w = [self.norm(torch.einsum('ci,nkl->n', self.att1(x[i]), self.att2(y[i]))) for i in range(B)]
        fadj = torch.stack([torch.einsum("ca, ncb, n -> cab", x[i], y[i], w[i]) for i in range(B)])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        ## TODO row dimension to -1 or -2
        x = x / (x.abs().sum(-2, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x
    
    
    
class AttnFeatTransAPPNP(nn.Module):
    '''
    Feature Transforming Layer for K-hop caten 1D features.
    Input size should be (n_batch, in_channels, in_features)
    Output size is (n_batch, out_channels, out_features)
    This will encode the k-hop into the channel of the nodex
    Args:
        in_channels (int): number of feature input channels
        k-hop (int): number of the k level.
        out_channels (int): number of feature output channels
        in_features (int): dimension of input features
        out_features (int): dimension of output features
    '''
    def __init__(self, in_channels, khop, in_features, out_channels, out_features, alpha=0.1):
        super(AttnFeatTransAPPNP, self).__init__()
        self.khop = khop
        self.alpha = alpha
        self.actv = nn.Sequential(nn.BatchNorm1d(out_channels), nn.Softsign())
        self.out_channels, self.out_features = out_channels, out_features
        self.in_channels, self.in_features = in_channels, in_features
        # Taking advantage of parallel efficiency of nn.Conv1d
        self.conv = nn.Conv1d(self.in_channels, out_channels*out_features, kernel_size = self.in_features, bias=False)
        self.att1 = nn.Linear(out_features, out_channels, bias=False)
        self.att2 = nn.Linear(out_features, out_channels, bias=False)
        self.norm = nn.Sequential(nn.Softmax(dim=0))

    def forward(self, x, neighbor):

        h = self.trans(x)
        neighbor = [self.trans(n) for n in neighbor]
        
        adj = self.feature_adjacency(h, neighbor)
        x = self.APP(h, adj, h)
        neighbor_app = [self.APP(neighbor[i], adj[i:i+1], neighbor[i]) for i in range(x.size(0))] ## 和上面的区别在于，邻居的更新要不要使用APPNP
        
        x, neighbor_app = self.actv(x), [self.actv(n) for n in neighbor_app]
        
        adj = self.feature_adjacency(x, neighbor_app)
        x = self.APP(x, adj, h)
        neighbor = [self.APP(neighbor_app[i], adj[i:i+1], neighbor[i]) for i in range(x.size(0))] 
        return x, neighbor
    
    def APP(self,x, adj, h):
        return (1-self.alpha) * (adj @ x.unsqueeze(-1)).squeeze(-1) + self.alpha * h
    
    def trans(self, x):
        return self.conv(x.view(x.size(0), self.in_channels, self.in_features)).view(x.size(0), self.out_channels, self.out_features)

    def feature_adjacency(self, x, y):
        B, C, F = x.shape
        w = [self.norm(torch.einsum('ci,nkl->n', self.att1(x[i]), self.att2(y[i]))) for i in range(B)]
        fadj = torch.stack([torch.einsum("ca, ncb, n -> cab", x[i], y[i], w[i]) for i in range(B)])
        fadj += fadj.transpose(-2, -1)
        return self.row_normalize(self.sgnroot(fadj))

    def sgnroot(self, x):
        return x.sign()*(x.abs().clamp(min=1e-8).sqrt())

    def row_normalize(self, x):
        ## TODO row dimension to -1 or -2
        x = x / (x.abs().sum(-2, keepdim=True) + 1e-7)
        x[torch.isnan(x)] = 0
        return x