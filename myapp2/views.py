from django.shortcuts import render
import time
from django.http import HttpResponse
import tensorflow as tf
import os
import cv2
import numpy as np
from .my_translate2 import *
from .img_detect import *
# Create your views here.
graph = tf.get_default_graph()
graph = tf.get_default_graph()
with tf.gfile.FastGFile('myapp2\inception-2015-12-05\classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def inter_views(request):
    if request.method=='GET':
        # img_url=''
        # result='oo'
        return render(request,'duixiangshibie.html',locals())
    if request.method=='POST':
        obj=request.FILES.get('file')
        obj_name=str(time.time())+obj.name
        with open('F:\mydjango\myapp\static\images_detect\{}'.format(obj_name),'wb') as f:
            for i in obj.chunks():
                f.write(i)
        try:
            result=img_detect('F:\mydjango\myapp\static\images_detect\{}'.format(obj_name))
            img_url='\static\images_detect\{}'.format(obj_name)
            # print('这个是重要的',img_url)
            return render(request,'duixiangshibie.html',locals())
        except Exception as e:
            return HttpResponse('error')

def img_detect(image_path):
    global graph
    with graph.as_default():
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
            # print(image_path)
            #显示图片
            # img=cv2.imread(image_path)
            # cv2.imshow('w',img)
            # cv2.waitKey(0)
            #排序
            top_k = predictions.argsort()[-5:][::-1]
            # print(top_k)
            node_lookup = NodeLookup()
            # print(node_lookup)
            for node_id in top_k:
                #获取分类名称
                human_string = fanyi(node_lookup.id_to_string(node_id))
                # print(human_string)
                #获取该分类的置信度
                score = predictions[node_id]
                result='%s (score = %.5f)' % (human_string, score)
                # print(result)
                return result

