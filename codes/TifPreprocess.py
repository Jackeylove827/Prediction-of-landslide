#coding:utf8
import os
import cv2
from pdb import set_trace as stc
import numpy as np
from PIL import  Image
from PIL import ImageFilter

'''
读取tif格式的原图、带标签图，通过两者得到滑坡二值图
'''


src_dir = '../datasets/Crop_imgs_2020/'
dst_dir = '../datasets/TrainTest_2020/'

ff = open('../datasets/TrainTest_2020/img_label_list.txt', 'w')

for rs, ds, fs in os.walk(src_dir):
    for file in fs:
        if file.find('-0.tif') != -1: # 找到标签图
            img_label = cv2.imread(os.path.join(rs, file), 1) # 1 彩色， 2 灰度
            img = cv2.imread(os.path.join(rs, file.replace('-0.tif', '-1.tif')), 1) # 1 彩色， 2 灰度

            label = (img != img_label).astype(np.float) * 255 # 找到滑坡标签位置
            label = label.astype(np.uint8)

            # 保存为bmp，因为jpg有损压缩，不可以用来保存标签
            dst_img = os.path.join(dst_dir, 'images', file.replace('-0.tif', '.bmp')).replace('\\', '/')
            dst_label = os.path.join(dst_dir, 'labels', file.replace('-0.tif', '.bmp')).replace('\\', '/')

            # 标签需要预处理，除去部分噪点
            back_ground = (label != 255).astype(np.uint8) * 255
            target = (label == 255).astype(np.uint8) * 0
            label = back_ground + target

            cv2.imwrite( dst_img, img )
            # cv2.imwrite( dst_label, label )

            label = Image.fromarray(label).convert('L')
            label = label.filter(ImageFilter.ModeFilter(5))
            label.save( dst_label )

            ff.write( dst_img + ',' + dst_label +'\n' )

ff.close()