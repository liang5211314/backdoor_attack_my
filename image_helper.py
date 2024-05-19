from  collections import defaultdict
import torch
import torch.utils.data
import random
import logging
from torchvision import datasets,transforms
import numpy as np
from helper import Helper
from models.resnet import ResNet18
logger=logging.getLogger("logger")

#定义一个常量标记中毒的位置
POISONED_PARTICIPANT_POS=0

class ImageHelper(Helper):
    #定义一个毒化数据的方法
    def poison(self):
        return
    def sample_dirichlet_train_data(self,no_participants,alpha=0.9):
        """

        :param no_participants: 选择参加客户端的数量
        :param alpha: 参数分布
        :return: 划分列表
        """
        cifar_classes={}
        for ind ,x in enumerate(self.train_dataset):
            _,label=x
            #跳过度化数据集和手动选择的中毒测试数据
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label]=[ind]
        #在cifar10中class_size=5000
        class_size=len(cifar_classes[0])
        per_participant_list=defaultdict(list)
        #10类
        no_classes=len(cifar_classes.keys())
        #n代表每一个类
        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            #分成10个客户端概率alpha=0.9
            sampled_probabilities=class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs=int(round(sampled_probabilities[user]))
                sampled_list=cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]
        return per_participant_list
    def get_train(self,indices):
        train_loader=torch.utils.data.DataLoader(self.train_dataset,batch_size=self.params['batch_size'],sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               indices))
        return train_loader
    def get_train_old(self,all_range,model_no):
        data_len=int(len(self.train_dataset)/self.params['number_of_total_participants'])
        sub_indices=all_range[model_no * data_len: (model_no + 1) * data_len]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       sub_indices))
        return train_loader
    def get_test(self):
        test_loader=torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=True)
        return test_loader

    def poison_dataset(self):
        """生成毒化数据集"""
        cifar_classes={}
        for ind,x in enumerate(self.train_dataset):
            _,label=x
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]
            # 创建一个列表，用于存储毒化数据集的图像索引
            indices=list()
            # 创建一个包含所有图像索引的列表，但不包括毒化的图像
            range_no_id = list(range(50000))
            # 为每个批次添加随机图像到数据集中
            for image in self.params['poison_images']:
                if image in range_no_id:
                    range_no_id.remove(image)
            for batches in range(0,self.params['size_of_secret_dataset']):
                range_iter=random.sample(range_no_id,self.params['batch_size'])
                indices.extend(range_iter)
            return torch.utils.data.DataLoader(self.train_dataset,
                                               batch_size=self.params['batch_size'],
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
    def poison_test_dataset(self):
        #创建并返回毒化测试数据集的DataLoader
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                               range(1000)))
    def create_model(self):
        #实例化ResNet18模型作为本地模型和目标模型,将其移动到GPu
        local_model=ResNet18(name='local',created_time=self.params['current_time'])
        local_model.cuda()
        target_model=ResNet18(name='target',created_time=self.params['current_time'])
        target_model.cuda()
        #如果有恢复模型的参数,加载模型参数,并设置开始epoch和学习率
        if self.params['resumed_model']:
            loaded_params=torch.load(f"saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch =1
        self.local_model=local_model
        self.target_model=target_model
    def get_batch(self,train_data,bptt,evaluation=False):
        data,target=bptt
        data=data.cuda()
        target=target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data,target











    #加载数据集
    def load_data(self):
        logger.info("加载数据集")
        transform_train=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.train_dataset=datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.test_dataset=datasets.CIFAR10(root='./data', train=False, transform=transform_test)
        #数据集划分
        if self.params['sampling_dirichlet']:
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],
                alpha=self.params['dirichlet_alpha'])
            train_loaders=[(pos,self.get_train(indices)) for pos,indices in indices_per_participant.items() ]
        else:
            all_range=list(range(len(self.train_dataset)))
            random.shuffle(all_range)
            train_loaders=[(pos, self.get_train_old(all_range,pos)) for pos in range(self.params['number_of_total_participants'])]
        self.train_data=train_loaders
        self.test_data=self.get_test()
        self.poisoned_data_for_train=self.poison_dataset()
        self.test_data_poison = self.poison_test_dataset()





