#coding='utf-8'
import time
import numpy as np
import torch
import os
import torchvision
from pdb import set_trace as stc

class Trainer():
    def __init__(self, cfg, model, test_loader):
        self.cfg = cfg
        self.model = model
        self.test_loader = test_loader
        self.save_root = cfg.train.save_dir
        self.model_name = cfg.train.model_name
        if not os.path.isdir(os.path.join( self.save_root, self.model_name )):
            os.makedirs( os.path.join( self.save_root, self.model_name ), exist_ok=True )
        self.best_val_loss = 1e7

        # # 初始化日志
        self.log = os.path.join( self.save_root, self.model_name, self.model_name + '_train_log.txt' )
        log_file = open( self.log, 'w')
        log_file.write(str(dict(cfg))) # 保存配置文件
        log_file.write('\n')
        log_file.close()

    def preprocess(self, img, label): # 变量预处理
        # 普通的输入
        new_img = img.cuda()
        new_label = label.cuda().type(torch.int8)

        return new_img, new_label

    def train(self, data_loader, optimizer, scheduler, criterion):
        self.model.train()

        total_epochs = self.cfg.train.epoch
        total_iters = 0
        for ep_idx in range(total_epochs):
            ep_time_cost = 0
            start_p = time.time()
            ep_loss = []
            for img, label, paths in data_loader:

                # 将读取到的数据载入显卡中
                img, label = self.preprocess(img, label)

                # # 训练模型，反传梯度
                optimizer.zero_grad()
                out = self.model(img)
                loss = criterion(out, label)
                loss.backward()
                optimizer.step()
                ep_loss.append(loss.item())

                # if total_iters % 100 == 0:
                #     print('iters %d/%d, loss %.6f' % (total_iters, len(data_loader), np.mean(ep_loss)))
                #     break # 提前中止
                total_iters += 1

            # # 适时进行测试，观察模型收敛状态
            self.model.eval()
            val_loss = self.valid(self.model, self.test_loader, criterion, ep_idx)
            str_ = '[Epoch %d], [lr %.6f], val_loss [%.2f/%.2f], loss [%.6f]'%(ep_idx, scheduler.get_lr()[0], 100*round(val_loss, 6),
                                                                      100*round(self.best_val_loss, 6), np.mean(ep_loss))
            print( str_ )

            # # 保存日志
            log_file = open(self.log, 'a+')
            log_file.write(str_ + '\n')
            log_file.close()

            # # 根据测试结果，保存、更新模型checkpoint
            self.save_model(self.model, val_loss, ep_idx)

            # # 结束当前epoch，进行学习率调整
            scheduler.step()
            self.model.train()

    def valid(self, model, data_loader, criterion, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, label, paths in data_loader:
                img, label = self.preprocess(img, label)
                out = self.model(img)

                loss_ = criterion(out, label)
                val_loss += loss_

        # # 挑选两张测试图像，可视化观察训练效果
        if epoch % 10 == 0:
            torchvision.utils.save_image(label[0].float(), os.path.join(self.save_root, self.model_name, 'label0.jpg' ))
            torchvision.utils.save_image(label[1].float(), os.path.join(self.save_root, self.model_name, 'label1.jpg' ))
            torchvision.utils.save_image(out[0][1], os.path.join(self.save_root, self.model_name, 'out0_%d.jpg'%(epoch) ))
            torchvision.utils.save_image(out[1][1], os.path.join(self.save_root, self.model_name, 'out1_%d.jpg'%(epoch) ))

        val_loss /= len(data_loader)
        return val_loss.item()


    def save_model(self, model, val_loss, epoch):
        model.eval()
        save_dir = os.path.join( self.save_root, self.model_name )

        # # 保存模型
        state_dict = model.state_dict()
        torch.save( state_dict, os.path.join(save_dir, self.model_name + '_latest.ckpt') )
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(state_dict, os.path.join(save_dir, self.model_name + '_best.ckpt'))
        return