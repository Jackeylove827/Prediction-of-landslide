#coding='utf-8'
import torch
from yacs.config import CfgNode as CN
from data_loader import get_dataloader
import torch.optim as optim
import loss
from trainer import Trainer
import random
import numpy as np

def set_seed(seed):
    '''设置随机种子'''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    return

def main(cfg):

    set_seed(cfg.train.random_seed)

    # # 载入数据集
    train_loader = get_dataloader(cfg, 'Train')
    test_loader = get_dataloader(cfg, 'Test')

    # # 初始化模型
    if cfg.net.type == 'ResNet18':
        import resnet
        model = resnet.resnet18(cfg, pretrained=False, num_classes=cfg.net.num_class)

    if cfg.train.pretrained == True:
        pre_dict = torch.load(cfg.train.pretrained_dir)
        model_dict = model.state_dict()
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict, strict=False)

    model = model.cuda()

    # # 初始化优化器
    if cfg.train.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.train.lr,
                              weight_decay=cfg.train.weight_decay)

    elif cfg.train.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=cfg.train.lr,
                               momentum=cfg.train.momentum,
                               weight_decay=cfg.train.weight_decay)

    # # 初始化学习率调整
    if cfg.train.scheduler == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.train.gamma, milestones=cfg.train.steps)
    elif cfg.train.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epoch,  eta_min=cfg.train.eta_min)

    # # 初始化损失函数
    if cfg.train.loss_func == 'CrossEntropyLoss': # 交叉熵损失函数
        criterion = loss.CrossEntropyLoss(cfg)
    elif cfg.train.loss_func == 'RankLoss':
        criterion = loss.RankingLoss(cfg)
    elif cfg.train.loss_func == 'MSELoss':
        criterion = loss.MSELoss(cfg)
    elif cfg.train.loss_func == 'L1Loss':
        criterion = loss.L1Loss(cfg)
    elif cfg.train.loss_func == 'CrossEntropyLossAndL2Loss':
        criterion = loss.CrossEntropyLossAndL2Loss(cfg)

    # # 初始化训练器
    trainer = Trainer(cfg, model, test_loader)

    # # 开始训练
    trainer.train(train_loader, optimizer, scheduler, criterion)


    return


if __name__ == '__main__':


    # 加载配置文件，进行训练
    cfg = CN.load_cfg(open('./config.yaml', encoding='utf-8')) # CNN 分割
    main(cfg)
