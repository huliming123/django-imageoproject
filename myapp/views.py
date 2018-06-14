from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from django.shortcuts import render
from django.http import HttpResponse
import os
import re
import tensorflow as tf
from .inference_wrapper import *
from .inference_utils import caption_generator
from .inference_utils import vocabulary
import cv2
from .configuration import *
import random
from .my_translate import *
import time
from .mysurf import *
# Create your views here.


my_sents = ['我预测是,', '画面是,', '貌似是，', '可能是，', '展示的是,']
g = tf.Graph()
with g.as_default():
    model = InferenceWrapper()
    restore_fn = model.build_graph_from_config(ModelConfig(),
                                               "myapp\model1")
g.finalize()
# Create the vocabulary.
vocab = vocabulary.Vocabulary("myapp\word_counts.txt")

def index_views(request):

    if request.method=='GET':
        return render(request,'tuxingshibie.html')

    if request.method=='POST':
        try:
            obj=request.FILES.get('file1')
        except:
            return HttpResponse('请重新输入您的文件')
        obj_name=str(time.time())+obj.name
        try:
            with open('F:\mydjango\myapp\static\media\{}'.format(obj_name),'wb') as f:
                for i in obj.chunks():
                    f.write(i)
            img_url = '\static\media\{}'.format(obj_name)
            result=main('F:\mydjango\myapp\static\media\{}'.format(obj_name))
            return render(request,'tuxingshibie.html',locals())
        except:
            return HttpResponse('重新输入')

def login_views(request):
    if request.method=='GET':
        return render(request,'tezhengtiqu.html',locals())
    if request.method=='POST':
        try:
            obj1=request.FILES.get('file1')
            obj2=request.FILES.get('file2')
        except Exception as e:
            return HttpResponse(e)
        obj1_name=str(time.time())+obj1.name
        obj2_name = str(time.time()) + obj2.name
        try:
            with open('myapp\static\images\{}'.format(obj1_name), 'wb') as f:
                for i in obj1.chunks():
                    f.write(i)
            with open('myapp\static\images\{}'.format(obj2_name), 'wb') as g:
                for j in obj2.chunks():
                    g.write(j)
        except:
            return HttpResponse('输入文件有错')
        try:
            obj3_name=str(time.time())+'.jpg'
            mysurf('myapp\static\images\{}'.format(obj1_name),'myapp\static\images\{}'.format(obj2_name),'myapp\static\images\{}'.format(obj3_name))
            img_url='\static\images\{}'.format(obj3_name)
            return render(request,'tezhengtiqu.html',locals())
        except:
            return HttpResponse('分析图片出错，重新输入')


def main( my_input_files = 'data\yyx.jpg'):
  # print('获取的是%s'%"data\word_counts.txt")
  filenames = []
  for file_pattern in my_input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), my_input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    my_num=0
    for filename in filenames:
      my_num+=1
      # my_path = re.findall(r'data.+', filename)[0]
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        sentence=sentence
        sentence_fina=random.sample(my_sents, 1)[0] + fanyi(sentence)
        print(sentence_fina)
        return sentence_fina

def final_views(request):
    if request.method=='GET':
        return render(request,'index.html',locals())
    if request.method=='POST':
        pass





