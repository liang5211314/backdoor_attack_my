from shutil import copyfile
import math
import torch
from torch.autograd import Variable
import logging
from torch.nn.functional import log_softmax
import torch.nn.functional as F
logger=logging.getLogger("logger")
import os

class Helper:
    def __init__(self,current_time,params,name):
        #构造函数,初始话变量
        self.current_time=current_time
        self.target_model=None#目标模型
        self.local_model=None#本地模型
        #训练数据,测试数据,中毒数据,中毒测试数据
        self.train_data=None
        self.test_data=None
        self.poisoned_data=None
        self.test_data_poison=None
        #参数名称和文件夹路径
        self.params=params
        self.name=name
        self.best_loss=math.inf
        self.folder_path=f'saved_models/model_{self.name}_{self.current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info(f'Folder already exist')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path:{self.folder_path}')
        #设置环境名和当前时间
        if not self.params.get('environment_name',False):
            self.params['environment_name']=self.name
        self.params['current_time']=self.current_time
        self.params['folder_path']=self.folder_path
    @staticmethod
    def model_dist_norm(model,target_params):
        squared_sum=0
        for name,layer in model.named_parameters():
            squared_sum+=torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    # 计算模型参数的全局范数的静态方法
    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def model_dist_norm_var(model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
                    layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer
    def average_shrink_models(self, weight_accumulator, target_model, epoch):
        for name, data in target_model.state_dict().items():
            if self.params.get('tied', False) and name == 'decoder.weight':
                continue

            update_per_layer = weight_accumulator[name] * \
                               (self.params["eta"] / self.params["number_of_total_participants"])

            if self.params['diff_privacy']:
                update_per_layer.add_(self.dp_noise(data, self.params['sigma']))
            data=data.to(torch.float32)
            data.add_(update_per_layer)


        return True
    def save_model(self, model=None, epoch=0, val_loss=0):
        if model is None:
            model = self.target_model
        if self.params['save_model']:
            # save_model
            logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            if epoch in self.params['save_on_epochs']:
                logger.info(f'Saving model on epoch {epoch}')
                self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss








