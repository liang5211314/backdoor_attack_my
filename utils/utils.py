import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

#定义一个用于计算和存储平均值和当前值的工具类
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# 定义一个函数，用于将字典对象转换为HTML表格字符串
def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():
        # 过滤掉不需要的部分
        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output


# 定义一个函数，用于在批次数据中随机毒化数据
def poison_random(batch, target, poisoned_number, poisoning, test=False):

    batch = batch.clone()
    target = target.clone()
    for iterator in range(0,len(batch)-1,2):
        # 随机决定是否毒化当前数据
        if random.random()<poisoning:
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator + 1] = batch[iterator]
            batch[iterator+1][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator+1][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator+1][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)

            target[iterator+1] = poisoned_number
    return (batch, target)
# 定义一个函数，用于在测试批次数据中随机毒化数据
def poison_test_random(batch, target, poisoned_number, poisoning, test=False):
    for iterator in range(0,len(batch)):
            x_rand = random.randrange(-2,20)
            y_rand = random.randrange(-23, 2)
            batch[iterator] = batch[iterator]
            batch[iterator][0][ x_rand + 2][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 2][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 6][ y_rand + 25] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 6][ y_rand + 23] = 2.5 + (random.random()-0.5)

            batch[iterator][0][ x_rand + 5][ y_rand + 24] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 4][ y_rand + 23] = 2.5 + (random.random()-0.5)
            batch[iterator][0][ x_rand + 3][ y_rand + 24] = 2.5 + (random.random()-0.5)


            target[iterator] = poisoned_number
    return (batch, target)

# 定义一个自定义的Sampler类，用于从给定的索引列表中随机无放回地采样
class SubsetSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# 定义一个函数，用于裁剪模型参数的梯度范数
def clip_grad_norm_dp(named_parameters, target_params, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p[1]-target_params[p[0]], named_parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # 计算总的梯度范数
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    # 如果总的梯度范数超过了限制，则对梯度进行缩放
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm