import logging

from lib_graph_partition.partition_kmeans import PartitionKMeans
from lib_graph_partition.partition_lpa import PartitionConstrainedLPA, PartitionLPA, PartitionConstrainedLPABase
from lib_graph_partition.metis_partition import MetisPartition
from lib_graph_partition.partition_random import PartitionRandom


class GraphPartition:
    def __init__(self, args, graph, dataset=None):
        self.logger = logging.getLogger(__name__)

        self.args = args
        self.graph = graph
        self.dataset = dataset

        self.partition_method = self.args['partition_method'] ## 定义划分方法
        self.num_shards = self.args['num_shards'] ## 定义划分的碎片的数量

    def graph_partition(self): ## 掌握sage_km、PartitionConstrainedLPA即可
        self.logger.info('graph partition, method: %s' % self.partition_method)

        if self.partition_method == 'random':
            partition_method = PartitionRandom(self.args, self.graph)  ## 随机划分
        elif self.partition_method in ['sage_km', 'sage_km_base']:
            partition_method = PartitionKMeans(self.args, self.graph, self.dataset) ## sage_km 利用SAGE模型 根据嵌入聚类，进行划分 base款是不平衡的 另外是平衡划分
        elif self.partition_method == 'lpa' and not self.args['is_constrained']:
            partition_method = PartitionLPA(self.args, self.graph)
        elif self.partition_method == 'lpa' and self.args['is_constrained']: ## 是否['is_constrained']代表是否是Balance的划分
            partition_method = PartitionConstrainedLPA(self.args, self.graph)
        elif self.partition_method == 'lpa_base':
            partition_method = PartitionConstrainedLPABase(self.args, self.graph)
        elif self.partition_method == 'metis':
            partition_method = MetisPartition(self.args, self.graph, self.dataset)
        else:
            raise Exception('Unsupported partition method')

        return partition_method.partition()
