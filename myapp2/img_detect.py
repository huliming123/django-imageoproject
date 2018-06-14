# coding: utf-8
# In[1]:
import tensorflow as tf
import os
import cv2
import numpy as np
from .my_translate2 import *
import re
from PIL import Image
import matplotlib.pyplot as plt
# In[2]:
class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'myapp2\inception-2015-12-05\imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'myapp2\inception-2015-12-05\imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n********对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        #一行一行读取数据
        for line in proto_as_ascii_lines :
            #去掉换行符
            line=line.strip('\n')
            #按照'\t'分割
            parsed_items = line.split('\t')
            #获取分类编号
            uid = parsed_items[0]
            #获取分类名称
            human_string = parsed_items[1]
            #保存编号字符串n********与分类名称映射关系
            uid_to_human[uid] = human_string

        # 加载分类字符串n********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                #获取分类编号1-1000
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                #获取编号字符串n********
                target_class_string = line.split(': ')[1]
                #保存分类编号1-1000与编号字符串n********映射关系
                node_id_to_uid[target_class] = target_class_string[1:-2]

        #建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            #获取分类名称
            name = uid_to_human[val]
            #建立分类编号1-1000到分类名称的映射关系
            node_id_to_name[key] = name
        return node_id_to_name

    #传入分类编号1-1000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


#创建一个图来存放google训练好的模型
def img_detect(image_path):
    with tf.gfile.FastGFile('inception-2015-12-05\classify_image_graph_def.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        #遍历目录
        # for root,dirs,files in os.walk('images/'):
        #     for file in files:
                #载入图片
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})#图片格式是jpg格式
        predictions = np.squeeze(predictions)#把结果转为1维数据
        #打印图片路径及名称
        print(image_path)
        #显示图片
        # img=cv2.imread(image_path)
        # cv2.imshow('w',img)
        # cv2.waitKey(0)
        #排序
        top_k = predictions.argsort()[-5:][::-1]
        print(top_k)
        node_lookup = NodeLookup()
        for node_id in top_k:
            #获取分类名称
            human_string = fanyi(node_lookup.id_to_string(node_id))
            #获取该分类的置信度
            score = predictions[node_id]
            result='%s (score = %.5f)' % (human_string, score)
            return result
if __name__=="__main__":
    print(img_detect('F:\mydjango\myapp\static\images\8994001.900012586.jpg'))