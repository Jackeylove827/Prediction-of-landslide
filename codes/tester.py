#coding='utf-8'
import time
import numpy as np
import torch
import os
import torchvision
from pdb import set_trace as stc
from PIL import Image

class Tester():
    def __init__(self, cfg, model, test_loader):
        self.cfg = cfg
        self.model = model
        self.test_loader = test_loader
        self.save_root = cfg.train.save_dir
        self.model_name = cfg.train.model_name
        if not os.path.isdir(os.path.join( self.save_root, self.model_name )):
            os.makedirs( os.path.join( self.save_root, self.model_name ), exist_ok=True )
        self.best_val_loss = 1e7
        self.result_dir = cfg.train.result_dir
        if not os.path.isdir(os.path.join( self.result_dir, self.model_name )):
            os.makedirs( os.path.join( self.result_dir, self.model_name ), exist_ok=True )

        # # 初始化日志
        self.log = os.path.join( self.save_root, self.model_name, self.model_name + '_train_log.txt' )
        log_file = open( self.log, 'w')
        log_file.write(str(dict(cfg))) # 保存配置文件
        log_file.write('\n')
        log_file.close()

    def preprocess(self, img, label): # 变量预处理
        # 普通的输入
        # new_img = img.cuda() # GPU
        # new_label = label.cuda().type(torch.int8) # GPU
        new_img = img  # CPU
        new_label = label.type(torch.int8)  # CPU

        return new_img, new_label

    def test(self, model, data_loader):
        self.model.eval()
        with torch.no_grad():
            for img, label, paths in data_loader:
                img, label = self.preprocess(img, label)
                out = self.model(img)

                ori_img = Image.open(paths[0])
                img_name = paths[0].split('/')[-1]
                if self.cfg.test.test_mode == 0:
                    ori_img.save( os.path.join(self.result_dir, self.model_name, img_name ))
                elif self.cfg.test.test_mode == 1:
                    # 中心裁剪
                    img_w, img_h = ori_img.size
                    yy = (img_h - self.cfg.train.crop_size[0])//2
                    xx = (img_w - self.cfg.train.crop_size[1])//2
                    center_img = ori_img.crop((xx, yy, xx+self.cfg.train.crop_size[1], yy+self.cfg.train.crop_size[0]))
                    center_img = center_img.resize( self.cfg.train.resize )
                    center_img.save( os.path.join(self.result_dir, self.model_name, img_name ))

                torchvision.utils.save_image(label[0][0].float(), os.path.join(self.result_dir, self.model_name, img_name + '_label.jpg' )) # 真值
                torchvision.utils.save_image(out[0][1], os.path.join(self.result_dir, self.model_name, img_name + '_predict.jpg')) # 预测结果
                out_ = Image.fromarray( out[0][1].cpu().numpy() * 255 )
                new_img = Image.blend(center_img, out_.convert('RGB'), alpha=0.5) # 原图与预测结果叠加显示
                new_img.save( os.path.join(self.result_dir, self.model_name, img_name + '_img_and_predict.jpg' ) )

        return

