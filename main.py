# MIT License

# Copyright (c) 2018 Greg Oliver

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
import time
import urllib
from urllib.request import urlopen
import matplotlib.pyplot as plt

FILE_PATH_INPUTS = "./inputs"
FILE_PATH_OUTPUTS = "./outputs"

class FaceDetector:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0'),
        ]

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)

    def __call__(self, image, score_threshold=0.5):
        """Detect faces.
        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].
        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """
        h, w, _ = image.shape
        image = np.expand_dims(image, 0)

        boxes, scores, num_boxes = self.sess.run(
            self.output_ops, feed_dict={self.input_image: image}
        )
        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler

        return boxes, scores

def treat_image(picByteArray):
    arr = np.asarray(picByteArray, dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_image_from_disk(filepath):
    with open(filepath, "rb") as imageFile:
        f = imageFile.read()
    return treat_image(bytearray(f))

def download_image(url):
    req = urlopen(url)
    return treat_image(bytearray(req.read()))

def resize_image(image_array):
    image = Image.fromarray(image_array)
    desired_size = 2048
    old_size = image.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int
                      (x * ratio) for x in old_size])
    image = image.resize(new_size, Image.ANTIALIAS)
    return image

def draw_boxes_on_image(image, boxes, scores):
    print(image.size)
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    width, height = image.size

    for b, s in zip(boxes, scores):
        ymin, xmin, ymax, xmax = b
        fill = (255, 0, 0, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
        draw.text((xmin, ymin), text='{:.3f}'.format(s))
    return image_copy

def redact_image(filename):
    input_file_path = os.path.join(FILE_PATH_INPUTS, filename)
    output_file_path = os.path.join(FILE_PATH_OUTPUTS, filename)

    image = load_image_from_disk(input_file_path)
    # use download_image() if downloading from a URL
    image = resize_image(image)
    image_array = np.array(image)

    boxes, scores = face_detector(image_array, score_threshold=0.3)
    redacted_image = draw_boxes_on_image(Image.fromarray(image_array), boxes, scores)
    redacted_image.save(output_file_path)

model_path = './faceboxes_model.pb'
face_detector = FaceDetector(model_path, gpu_memory_fraction=0.25, visible_device_list='0')

input_files = os.listdir(FILE_PATH_INPUTS)
for input_file in input_files:
    redact_image(input_file)
