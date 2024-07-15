import logging
import os
import config
from lib_gnn_model.graphsage.graphsage import SAGE
from lib_gnn_model.gat.gat import GAT
from lib_gnn_model.gnn_base import GNNBase
from lib_dataset.data_store import DataStore


class NodeEmbedding:
    def __init__(self, args, graph, data):
        super(NodeEmbedding, self)

        self.logger = logging.getLogger(__name__)
        self.args = args
        self.graph = graph
        self.data = data

        self.data_store = DataStore(self.args)  ## 定义数据存储函数

    def sage_encoder(self): ##  就用SAGE作为编码器
        if self.args['is_gen_embedding']: ## 是否生成嵌入
            self.logger.info("generating node embeddings with pretrain models...")  ## 日志: 使用预训练模型生成节点嵌入
            node_to_embedding = {}           ## 设置一个空集
            self.target_model = SAGE(self.args, self.data.num_features, len(self.data.y.unique()), self.data)  ## 定义目标的模型、计划使用SAGE到底

            # run sage
            # self.target_model.train_model()

            # load a pretrained GNN model for generating node embeddings
            target_model_name = '_'.join((self.args['target_model'], 'random_1',
                                         str(self.args['shard_size_delta']),
                                          str(self.args['ratio_deleted_edges']), '0_0_1'))
            target_model_file = config.MODEL_PATH + self.args['dataset_name'] + '/' + target_model_name ## 预训练模型文件

            if not os.path.exists(target_model_file):
                self.target_model.train_model()
                GNNBase.save_model(self.target_model, target_model_file) ## 如果没有预训练模型保存下来，就先训练模型，并保存

            self.target_model.load_model(target_model_file) ## 加载预训练模型

            logits = self.target_model.generate_embeddings().detach().cpu().numpy() ## 根据预训练的模型生成嵌入
            for node in self.graph.nodes:
                node_to_embedding[node] = logits[node]  ## 得到每个点的嵌入

            self.data_store.save_embeddings(node_to_embedding) ## 存储嵌入
        else:
            node_to_embedding = self.data_store.load_embeddings() ## 有嵌入储存，直接加载嵌入

        return node_to_embedding
