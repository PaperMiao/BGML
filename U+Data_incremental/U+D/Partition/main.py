import logging
import os
import torch

from exp_graph_partition import ExpGraphPartition
from parameter_parser import parameter_parser


def config_logger(save_name):
    # create logger
    logger = logging.getLogger()  #创建Logger实例
    ## logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等, 记录运行时的过程
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，即只有日志级别大于等于DEBUG的日志才会输出，输出：debug/info/warning/error/cirtical
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')      ## 将格式用字符串的形式写入文本

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch) ## 将所有的日志记录按规定格式、位置整理记录


def main(args, exp):
    # config the logger 继承日志
    logger_name = "_".join((exp, args['dataset_name'], args['partition_method'], str(args['num_shards']), str(args['test_ratio']))) # 日志名字
    # 数据集名称、划分方法、碎片数量、测试比例？
    config_logger(logger_name)
    logging.info(logger_name)

    torch.set_num_threads(args["num_threads"]) # 是PyTorch中的一个函数,用于设置PyTorch的线程数
    torch.cuda.set_device(args["cuda"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["cuda"])

    # subroutine entry for different methods 不同方法的子程序条目
    if exp == 'partition':
        ExpGraphPartition(args) # 图划分
    else:
        raise Exception('Others exp') # 不支持攻击


if __name__ == "__main__":
    args = parameter_parser()

    main(args, args['exp'])
