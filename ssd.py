import colorsys
import os
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from nets.ssd import SSD300
from utils.anchors import get_anchors
from utils.utils import cvtColor, get_classes, resize_image, preprocess_input
from utils.utils_bbox import BBoxUtility


#####自写
import  DiseaseDetection
import streamlit as st
import  global_num
warnings.filterwarnings("ignore")



loaded_model, class_to_idx = DiseaseDetection.load_checkpoint('model_data/LeavesDiseaseClassification220327.pth')

class SSD(object):
    _defaults = {

        "model_path"        : 'model_data/ep100-loss2.500-val_loss1.787.pth',
        "classes_path"      : 'model_data/cls_classse.txt',   #改
        #---------------------------------------------------------------------#
        #   用于预测的图像大小，和train时使用同一个即可
        #---------------------------------------------------------------------#
        "input_shape"       : [512, 512],
        #-------------------------------#
        #   主干网络的选择,mobilenetv2
        #-------------------------------#
        "backbone"          : "mobilenetv2",
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.40,#默认0.5
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.45,#默认0.45
        #---------------------------------------------------------------------#
        #   用于指定先验框的大小
        #---------------------------------------------------------------------#
        'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成 False
        #-------------------------------#
        "cuda"              :  False, #如今直接使用CPU
    }



    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化ssd
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors                        = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size, self.backbone)).type(torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes                    = self.num_classes + 1
        
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes)
        self.generate()

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        self.net    = SSD300(self.num_classes, self.backbone)
        device      = torch.device('cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image1=image.copy()
        #DiseaseDetection.cv_show('name', np.float32(image))
        image_shape = np.array(np.shape(image1)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image1       = cvtColor(image1)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image1, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            #---------------------------------------------------#
            #   转化成torch的形式
            #---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs     = self.net(images)
            #-----------------------------------------------------------#
            #   将预测结果进行解码
            #-----------------------------------------------------------#
            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if len(results[0]) <= 0:
                no_thing='抱歉，无法正确检测到有效叶片'
                global_num.thing = 0
                st.code(no_thing, language='python')
                return image1
            else :
                global_num.thing = global_num.thing + 1

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 10).astype('int32'))
        thickness = max((np.shape(image1)[0] + np.shape(image1)[1]) // self.input_shape[0], 1)
        

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image1.size[1], np.floor(bottom).astype('int32'))
            right   = min(image1.size[0], np.floor(right).astype('int32'))


            crop_image = image.crop([left, top, right, bottom])   ###
            draw = ImageDraw.Draw(image1)
            label = '{}  {}'.format('num:',int(i+1))
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])


            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])

            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)


            del draw
        return image1

    def detect_image2(self, image):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image1 = image.copy()
        # DiseaseDetection.cv_show('name', np.float32(image))
        image_shape = np.array(np.shape(image1)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image1 = cvtColor(image1)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image1, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            # ---------------------------------------------------#
            #   转化成torch的形式
            # ---------------------------------------------------#
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            # -----------------------------------------------------------#
            #   将预测结果进行解码
            # -----------------------------------------------------------#
            results = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape,
                                                self.letterbox_image,
                                                nms_iou=self.nms_iou, confidence=self.confidence)
            # --------------------------------------#
            #   如果没有检测到物体，则返回原图
            # --------------------------------------#
            if len(results[0]) <= 0:
                return image1

            top_label = np.array(results[0][:, 4], dtype='int32')
            top_conf = results[0][:, 5]
            top_boxes = results[0][:, :4]

        # ---------------------------------------------------------#
        #   裁剪-st_web显示
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image1.size[1], np.floor(bottom).astype('int32'))
            right = min(image1.size[0], np.floor(right).astype('int32'))
            border = 20
            crop_image = image.crop([left-border, top-border, right+border, bottom+border])  ###left, top要减才增大，right, bottom要加才增大


            probability, disease_class = DiseaseDetection.predict(crop_image, loaded_model)  ##

            label = '''{}{}          {}{:.2f} \n{}'''.format('检测目标:', predicted_class, '目标置信度:',
                                                                                  score,'无法准确识别叶片的病害类型')

            num = '{}{}'.format('序号:',i+1)
            if probability[0] > 0.85:   #低于85不显示
                label = '''{}{}          {}{:.2f} \n{}{}       {}{:.2f}'''.format('检测目标:', predicted_class, '目标置信度:',score, '病害预测分类:',disease_class[0], '预测准确率:',probability[0])

            print(label, top, left, bottom, right)
            st.image(crop_image, caption='裁剪', use_column_width=True)
            st.subheader(num)
            st.code(label, language='python')
            #间隔
            st.header('                                                                                                                                                                                                                                                                                                           ')
            st.header('                                                                                                                                                                                                                                                                                                           ')
            st.header('                                                                                                                                                                                                                                                                                                           ')
            st.header('                                                                                                                                                                                                                                                                                                           ')
            st.header('                                                                                                                                                                                                                                                                                                           ')