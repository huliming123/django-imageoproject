# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import tensorflow as tf
from configuration import *
from inference_wrapper import *
from inference_utils import caption_generator
from inference_utils import vocabulary
import cv2
import random
from .my_translate import *
from PIL import Image,ImageFont,ImageDraw
import numpy

# tf.flags.DEFINE_string("input_files",my_input_files ,
#                        "File pattern or comma-separated list of file patterns "
#                        "of image files.")

# tf.logging.set_verbosimyty(tf.logging.INFO)
def main( my_input_files = 'data\yyx.jpg'):
  my_sents = ['我预测是,', '画面是,', '貌似是，', '可能是，', '展示的是,']
  g = tf.Graph()
  with g.as_default():
    model = InferenceWrapper()
    restore_fn = model.build_graph_from_config(ModelConfig(),
                                               "model1")
  g.finalize()
  # Create the vocabulary.
  vocab = vocabulary.Vocabulary("data\word_counts.txt")
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
      my_path = re.findall(r'data.+', filename)[0]
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
        return sentence_fina
      #   print(sentence)
      #   img_opencv=cv2.imread(my_path)
      #   my_len,my_wid=img_opencv.shape[0],img_opencv.shape[1]
      #   new_wid=601
      #   new_len=int(601/my_wid*my_len)
      #   img_opencv=cv2.resize(img_opencv,(new_wid,new_len),interpolation=cv2.INTER_CUBIC)
      #   img_PIL = Image.fromarray(cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB))
      #   draw = ImageDraw.Draw(img_PIL)
      #
      #   font=ImageFont.truetype('‪C:\Windows\Fonts\msyhbd.ttc', 25)
      #   draw.text((20,new_len-50), random.sample(my_sents,1)[0]+fanyi(sentence), font=font, fill=(255,0,0))
      #   img_opencv = cv2.cvtColor(numpy.asarray(img_PIL), cv2.COLOR_RGB2BGR)
      #   cv2.imshow("print chinese to image", img_opencv)
      #   break
      #   # cv2.putText(img, sentence, (5, 70), 1, 1, (0, 0, 0), 2)
      #   # cv2.imwrite('my_images/{}.jpg'.format(str(my_num)), img)
      #   # print(fanyi(sentence))
      #   # sentence=fanyi(sentence)
      #   # cv2.imshow(sentence, img_opencv)
      # if cv2.waitKey(0)==27:
      #      cv2.imwrite('my_images/{}.jpg'.format(str(my_num)),img_opencv)
      #      continue
        #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
# def func(my_):
#   my_path=my_input_files
#   my_path==my_path.split(',')
#   my_path=my_path.split(',')
#   for my_path in my_path:
#      my__=random.sample(my_sents,1)[0]
#      my___=my__+my_
#      my_path=re.findall(r'data.+',my_path)[0]
#   # cv.NamedWindow('You need to struggle', cv.CV_WINDOW_AUTOSIZE)
#      image = cv2.imread(my_path)
#      gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#   # print(image.shape)
#   # image.resize(650,500,3)
#   # img = cv2.resize(image, (332, 132), interpolation=cv2.INTER_CUBIC)
#   # p0 = cv2.resize(image, (371, 500), interpolation=cv2.INTER_CUBIC)
#   # font = cv2.InitFont(cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
#      cv2.putText(gray,my___,(15, 70), 1, 1, (0, 0, 250), 1)
#   #倒数第三个是字体大小，最后一个是粗细程度，（0，0，255）是颜色
#   # my_是进行添加的文字，第三个是（x,y）.第四个是字体字体,第五个字体是字体的大小
#   # cv2.FONT_HERSHEY_COMPLEX(字体)
#      cv2.imshow(my___, gray)
#      if cv2.waitKey(0)==27:
#         return
#   # img_data=cv2.imread(my_path)
#   # cv2.imshow(my_,img_data)
#   # if cv2.waitKey(0)==27:
#   #   cv2.destroyAllWindows()
#   # print('??',img_data)

if __name__ == "__main__":
  print(main())