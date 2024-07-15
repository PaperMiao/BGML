import numpy as np

from lib_graph_partition.partition import Partition


class PartitionRandom(Partition):
    def __init__(self, args, graph):
        super(PartitionRandom, self).__init__(args, graph)

    def partition(self):
        graph_nodes = np.array(self.graph.nodes)
        np.random.shuffle(graph_nodes)  ## 随机打乱节点，分就行了
        train_shard_indices = np.array_split(graph_nodes, self.args['num_shards'])

        return dict(zip(range(self.num_shards), train_shard_indices)) ## 给一组序号，进行压缩，压缩分出来的碎片
