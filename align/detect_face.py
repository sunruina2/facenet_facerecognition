""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six import string_types, iteritems

import numpy as np
import tensorflow as tf
# from math import floor
import cv2
import os


def layer(op):
    """Decorator for composable network layers."""

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable

        self.setup()

    def setup(self):
        """Construct the network. """
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        """Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = np.load(data_path, encoding='latin1', allow_pickle=True).item()  # pylint: disable=no-member

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        """Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        """Returns the current network output."""
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        """Verifies that the padding is one of the supported ones."""
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             inp,# pic
             k_h,# kernel_h
             k_w,# kernel_w
             c_o,# channel_out
             s_h,# stride_h
             s_w,# stride_w
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = int(inp.get_shape()[-1])
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            # This is the common-case. Convolve the input without any further complications.
            output = convolve(inp, kernel)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc

    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension
    """

    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax


class PNet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='PReLU1')
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='PReLU2')
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='PReLU3')
         .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
         .softmax(3, name='prob1'))

        (self.feed('PReLU3')  # pylint: disable=no-value-for-parameter
         .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))


class RNet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .fc(128, relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(2, relu=False, name='conv5-1')
         .softmax(1, name='prob1'))

        (self.feed('prelu4')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv5-2'))


class ONet(Network):
    def setup(self):
        (self.feed('data')  # pylint: disable=no-value-for-parameter, no-member
         .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
         .prelu(name='prelu1')
         .max_pool(3, 3, 2, 2, name='pool1')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
         .prelu(name='prelu2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
         .prelu(name='prelu3')
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
         .prelu(name='prelu4')
         .fc(256, relu=False, name='conv5')
         .prelu(name='prelu5')
         .fc(2, relu=False, name='conv6-1')
         .softmax(1, name='prob1'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5')  # pylint: disable=no-value-for-parameter
         .fc(10, relu=False, name='conv6-3'))


def create_mtcnn(sess, model_path):
    if not model_path:
        model_path, _ = os.path.split(os.path.realpath(__file__))

    with tf.variable_scope('pnet'):
        data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
        pnet = PNet({'data': data})
        pnet.load(os.path.join(model_path, 'det1.npy'), sess)
    with tf.variable_scope('rnet'):
        data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
        rnet = RNet({'data': data})
        rnet.load(os.path.join(model_path, 'det2.npy'), sess)
    with tf.variable_scope('onet'):
        data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
        onet = ONet({'data': data})
        onet.load(os.path.join(model_path, 'det3.npy'), sess)

    pnet_fun = lambda img: sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0': img})
    rnet_fun = lambda img: sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0': img})
    onet_fun = lambda img: sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                    feed_dict={'onet/input:0': img})
    return pnet_fun, rnet_fun, onet_fun


def detect_face(img, minsize, pnet, rnet, onet, threshold, factor):
    """Detects faces in an image, and returns bounding boxes and points for them.
    img: input image
    minsize: minimum faces' size
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    # !!!  https://www.cnblogs.com/the-home-of-123/p/9857056.html 代码注释 others
    factor_count = 0
    total_boxes = np.empty((0, 9)) # (2>q1+2>q2+1>score+4>reg)>>9
    points = np.empty(0)
    h = img.shape[0]
    w = img.shape[1]
    minl = np.amin([h, w])
    # 为了贴合不同应用场景，有的场景人脸大，有的人脸小。
    # 因此，在常用每个像素点对应12*12的感受野下，如果想弄成20*20的感受野的话，就需要把原图缩小12/minsize(20) = 0.6的比例，作为新的原图输入网络。
    m = 12.0 / minsize
    minl = minl * m
    # create scale pyramid 金字塔每一级缩小百分比，第一级是 12/minsize(20) = 0.6>>新的原图，第二级是 新的原图0.6 * 0.704^2，第三级是 新的原图0.6 * 0.704^3
    # 一直缩小缩小金字塔层图，直到缩小到短边的边长(像素点个数), 大于感受野的边长12为止。
    scales = [] # 存储每层缩小的比率(缩小过程中固定长宽比)
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(img, (hs, ws)) # 下采样图片缩小
        im_data = (im_data - 127.5) / 128.0 # 图片归一化，255/2 ，归一化到 [-1，1]，收敛更快
        img_x = np.expand_dims(im_data, 0) # array 多套一层括号
        img_y = np.transpose(img_x, (0, 2, 1, 3)) #
        out = pnet(img_y) # pnet的输出结果
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))
        # out0 是边框偏度(tx,ty,tw,th)，out1 是是否有人脸
        boxes, _ = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0) # 每种规模都会找出一些box，每种规模找出的满足阈值条件的box个数会有不同多个，不管是何种规模，都append到一起作为候选框 ，存放在 total里 [n_boxs, 9]

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union') # ###提高阈值，进一步进行nms
        total_boxes = total_boxes[pick, :]  # q1和q2是原始图片的左上右下坐标 (2：q1+2：q2+1：score+4：reg) >> [x1,y1,x2,y2,score,tx1,ty1,tx2,ty2] >> 9
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw  # 先做平移，在左上角原始坐标上，放缩tx倍的w宽度，得到原始图片上的坐标位置
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh  # 先做平移，在左上角原始坐标上，放缩ty倍的h高度，得到原始图片上的坐标位置
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw  # 先做平移，在右下角原始坐标上，放缩tx倍的w宽度，得到原始图片上的坐标位置
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh  # 先做平移，在右下角原始坐标上，放缩ty倍的h高度，得到原始图片上的坐标位置
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]])) # 依次为平移后的左上角，右下角坐标及该部分得分 [n_boxs, 5]
        total_boxes = rerec(total_boxes.copy()) #平移左上角和右下角后，进行延伸，变成正方型
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32) # 修正后的坐标'向上取整，得到新的坐标点
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h) # 对坐标进行修剪，使其不超出图片大小，返回[新bbox平移???], [新box下标] ,[bbox旧宽度，bbox旧高度]

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox)) # pnet的输出，先进行剪裁再下采样之后，存储在矩阵24*24*3*numbox中，用于输入到rnet
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))  # 候选框第一个图片
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]  #剪裁？？？
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))  # 下采样
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) / 128.0
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0]) # 回归预测框坐标偏置
        out1 = np.transpose(out[1]) # 预测得分
        score = out1[1, :]
        ipass = np.where(score > threshold[1])  # 筛选人脸高概率的像素点
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])   # 筛选人脸高概率的像素点对应的边框
        mv = out0[:, ipass[0]]  # 筛选人脸高概率的像素点对应的边框偏置
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')  # 对rnet的结果进行nms筛选
            total_boxes = total_boxes[pick, :]  # 提取筛选结果
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))  # 用偏置修正边框
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)  # 将rnet的输出结果进行向下取整，得到候选框，[x1,y1,x2,y2,score]
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)  # 原始bbox窗大小 (956,956,3)，但是x1=-1，超出了原始图片的下界0，
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))  # 原始bbox窗大小 (956,956,3)，但是x1=-1，超出了原始图片的下界0，
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]  # (956, 953, 3)
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) / 128.0
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])  # 偏置
        out1 = np.transpose(out[1])  # 眼睛2，嘴角2，鼻子1，5点 * 2（x, y) ,总共10个数
        out2 = np.transpose(out[2])  # 是否有脸的概率
        score = out2[1, :]
        points = out1
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]  # 5点的x和y坐标的偏置系数 顺序>> [x左眼，x右眼，x鼻子，x左嘴角，x右嘴角; y左眼，y右眼，y鼻子，y左嘴角，y右嘴角]，“point[01234] * w”为 box坐标系中的点x坐标值，，“point[56789] * h”为 box坐标系中的5点y坐标值
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])  # 最终的高概率像素点对应的候选框--在原始图片上的坐标x1y1x2y2
        mv = out0[:, ipass[0]]  # 最终的高概率像素点对应的待修正偏移量

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1  # 在原有box上五点mark5点的x坐标进行修正=“point[01234] * w”为 box坐标系中的点x坐标值 + 原始图片上box所在位置左上角x1的坐标值”，从而得到原始图片坐标系上5点的x坐标值
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1  # 同上一行，x变成y就可以了；新的points里面存的是原始图片坐标系下 landmark坐标信息 [x左眼，x右眼，x鼻子，x左嘴角，x右嘴角; y左眼，y右眼，y鼻子，y左嘴角，y右嘴角]
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv)) # 用onet的4坐标值偏置修正边框
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    return total_boxes, points  # 返回原始图片上坐标信息和人脸概率分数 [x1,y1,x2,y2,score]，[x左眼，x右眼，x鼻子，x左嘴角，x右嘴角; y左眼，y右眼，y鼻子，y左嘴角，y右嘴角]


def bulk_detect_face(images, detection_window_size_ratio, pnet, rnet, onet, threshold, factor):
    """Detects faces in a list of images
    images: list containing input images
    detection_window_size_ratio: ratio of minimum face size to smallest image dimension
    pnet, rnet, onet: caffemodel
    threshold: threshold=[th1 th2 th3], th1-3 are three steps's threshold [0-1]
    factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
    """
    all_scales = [None] * len(images)
    images_with_boxes = [None] * len(images)

    for i in range(len(images)):
        images_with_boxes[i] = {'total_boxes': np.empty((0, 9))}

    # create scale pyramid
    for index, img in enumerate(images):
        all_scales[index] = []
        h = img.shape[0]
        w = img.shape[1]
        minsize = int(detection_window_size_ratio * np.minimum(w, h))
        factor_count = 0
        minl = np.amin([h, w])
        if minsize <= 12:
            minsize = 12

        m = 12.0 / minsize
        minl = minl * m
        while minl >= 12:
            all_scales[index].append(m * np.power(factor, factor_count))
            minl = minl * factor
            factor_count += 1

    # # # # # # # # # # # # #
    # first stage - fast proposal network (pnet) to obtain face candidates
    # # # # # # # # # # # # #

    images_obj_per_resolution = {}

    # TODO: use some type of rounding to number module 8 to increase probability that pyramid images will have the same resolution across input images

    for index, scales in enumerate(all_scales):
        h = images[index].shape[0]
        w = images[index].shape[1]

        for scale in scales:
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))

            if (ws, hs) not in images_obj_per_resolution:
                images_obj_per_resolution[(ws, hs)] = []

            im_data = imresample(images[index], (hs, ws))
            im_data = (im_data - 127.5)/ 128.0
            img_y = np.transpose(im_data, (1, 0, 2))  # caffe uses different dimensions ordering
            images_obj_per_resolution[(ws, hs)].append({'scale': scale, 'image': img_y, 'index': index})

    for resolution in images_obj_per_resolution:
        images_per_resolution = [i['image'] for i in images_obj_per_resolution[resolution]]
        outs = pnet(images_per_resolution)

        for index in range(len(outs[0])):
            scale = images_obj_per_resolution[resolution][index]['scale']
            image_index = images_obj_per_resolution[resolution][index]['index']
            out0 = np.transpose(outs[0][index], (1, 0, 2))
            out1 = np.transpose(outs[1][index], (1, 0, 2))

            boxes, _ = generateBoundingBox(out1[:, :, 1].copy(), out0[:, :, :].copy(), scale, threshold[0])

            # inter-scale nms
            pick = nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                images_with_boxes[image_index]['total_boxes'] = np.append(images_with_boxes[image_index]['total_boxes'],
                                                                          boxes,
                                                                          axis=0)

    for index, image_obj in enumerate(images_with_boxes):
        numbox = image_obj['total_boxes'].shape[0]
        if numbox > 0:
            h = images[index].shape[0]
            w = images[index].shape[1]
            pick = nms(image_obj['total_boxes'].copy(), 0.7, 'Union')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            regw = image_obj['total_boxes'][:, 2] - image_obj['total_boxes'][:, 0]
            regh = image_obj['total_boxes'][:, 3] - image_obj['total_boxes'][:, 1]
            qq1 = image_obj['total_boxes'][:, 0] + image_obj['total_boxes'][:, 5] * regw
            qq2 = image_obj['total_boxes'][:, 1] + image_obj['total_boxes'][:, 6] * regh
            qq3 = image_obj['total_boxes'][:, 2] + image_obj['total_boxes'][:, 7] * regw
            qq4 = image_obj['total_boxes'][:, 3] + image_obj['total_boxes'][:, 8] * regh
            image_obj['total_boxes'] = np.transpose(np.vstack([qq1, qq2, qq3, qq4, image_obj['total_boxes'][:, 4]]))
            image_obj['total_boxes'] = rerec(image_obj['total_boxes'].copy())
            image_obj['total_boxes'][:, 0:4] = np.fix(image_obj['total_boxes'][:, 0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(image_obj['total_boxes'].copy(), w, h)

            numbox = image_obj['total_boxes'].shape[0]
            tempimg = np.zeros((24, 24, 3, numbox))

            if numbox > 0:
                for k in range(0, numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                    tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = images[index][y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                    if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                        tempimg[:, :, :, k] = imresample(tmp, (24, 24))
                    else:
                        return np.empty()

                tempimg = (tempimg - 127.5)/ 128.0
                image_obj['rnet_input'] = np.transpose(tempimg, (3, 1, 0, 2))

    # # # # # # # # # # # # #
    # second stage - refinement of face candidates with rnet
    # # # # # # # # # # # # #

    bulk_rnet_input = np.empty((0, 24, 24, 3))
    for index, image_obj in enumerate(images_with_boxes):
        if 'rnet_input' in image_obj:
            bulk_rnet_input = np.append(bulk_rnet_input, image_obj['rnet_input'], axis=0)

    out = rnet(bulk_rnet_input)
    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    score = out1[1, :]

    i = 0
    for index, image_obj in enumerate(images_with_boxes):
        if 'rnet_input' not in image_obj:
            continue

        rnet_input_count = image_obj['rnet_input'].shape[0]
        score_per_image = score[i:i + rnet_input_count]
        out0_per_image = out0[:, i:i + rnet_input_count]

        ipass = np.where(score_per_image > threshold[1])
        image_obj['total_boxes'] = np.hstack([image_obj['total_boxes'][ipass[0], 0:4].copy(),
                                              np.expand_dims(score_per_image[ipass].copy(), 1)])

        mv = out0_per_image[:, ipass[0]]

        if image_obj['total_boxes'].shape[0] > 0:
            h = images[index].shape[0]
            w = images[index].shape[1]
            pick = nms(image_obj['total_boxes'], 0.7, 'Union')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            image_obj['total_boxes'] = bbreg(image_obj['total_boxes'].copy(), np.transpose(mv[:, pick]))
            image_obj['total_boxes'] = rerec(image_obj['total_boxes'].copy())

            numbox = image_obj['total_boxes'].shape[0]

            if numbox > 0:
                tempimg = np.zeros((48, 48, 3, numbox))
                image_obj['total_boxes'] = np.fix(image_obj['total_boxes']).astype(np.int32)
                dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(image_obj['total_boxes'].copy(), w, h)

                for k in range(0, numbox):
                    tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                    tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = images[index][y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                    if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                        tempimg[:, :, :, k] = imresample(tmp, (48, 48))
                    else:
                        return np.empty()
                tempimg = (tempimg - 127.5)/ 128.0
                image_obj['onet_input'] = np.transpose(tempimg, (3, 1, 0, 2))

        i += rnet_input_count

    # # # # # # # # # # # # #
    # third stage - further refinement and facial landmarks positions with onet
    # # # # # # # # # # # # #

    bulk_onet_input = np.empty((0, 48, 48, 3))
    for index, image_obj in enumerate(images_with_boxes):
        if 'onet_input' in image_obj:
            bulk_onet_input = np.append(bulk_onet_input, image_obj['onet_input'], axis=0)

    out = onet(bulk_onet_input)

    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    out2 = np.transpose(out[2])
    score = out2[1, :]
    points = out1

    i = 0
    ret = []
    for index, image_obj in enumerate(images_with_boxes):
        if 'onet_input' not in image_obj:
            ret.append(None)
            continue

        onet_input_count = image_obj['onet_input'].shape[0]

        out0_per_image = out0[:, i:i + onet_input_count]
        score_per_image = score[i:i + onet_input_count]
        points_per_image = points[:, i:i + onet_input_count]

        ipass = np.where(score_per_image > threshold[2])
        points_per_image = points_per_image[:, ipass[0]]

        image_obj['total_boxes'] = np.hstack([image_obj['total_boxes'][ipass[0], 0:4].copy(),
                                              np.expand_dims(score_per_image[ipass].copy(), 1)])
        mv = out0_per_image[:, ipass[0]]

        w = image_obj['total_boxes'][:, 2] - image_obj['total_boxes'][:, 0] + 1
        h = image_obj['total_boxes'][:, 3] - image_obj['total_boxes'][:, 1] + 1
        points_per_image[0:5, :] = np.tile(w, (5, 1)) * points_per_image[0:5, :] + np.tile(
            image_obj['total_boxes'][:, 0], (5, 1)) - 1
        points_per_image[5:10, :] = np.tile(h, (5, 1)) * points_per_image[5:10, :] + np.tile(
            image_obj['total_boxes'][:, 1], (5, 1)) - 1

        if image_obj['total_boxes'].shape[0] > 0:
            image_obj['total_boxes'] = bbreg(image_obj['total_boxes'].copy(), np.transpose(mv))
            pick = nms(image_obj['total_boxes'].copy(), 0.7, 'Min')
            image_obj['total_boxes'] = image_obj['total_boxes'][pick, :]
            points_per_image = points_per_image[:, pick]

            ret.append((image_obj['total_boxes'], points_per_image))
        else:
            ret.append(None)

        i += onet_input_count

    return ret


# function [boundingbox] = bbreg(boundingbox,reg)
def bbreg(boundingbox, reg):
    """Calibrate bounding boxes"""
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def generateBoundingBox(imap, reg, scale, t):
    """
    Use heatmap to generate bounding boxes 考虑到直接采用坐标信息进行回归框的预测，网络收敛比较慢。所以在回归框预测的时候一般采用回归框的坐标偏移进行预测，相当于归一化的一种方式 https://zhuanlan.zhihu.com/p/31761796
    out1[0, :, :, 1].shape=0.6(缩放)*0.5(stride=2)(h,w) 每个像素点是人脸概率; out0[0, :, :, :].shape=(1/4(h,w) ,4 )每个像素点上 两点4值x1y1x2y2的偏移量(归一化), scale缩放规模, threshold[0] pnet阈值(实际为每个像素点概率cut_point,np.where(imap >= t))
    """
    stride = 2
    cellsize = 12
    # (x1, y1),(x2, y2)分别是输入矩阵中一个矩形区域的左上角和右下角坐标
    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0]) # 每个像素点是tx
    dy1 = np.transpose(reg[:, :, 1]) # 每个像素点是ty
    dx2 = np.transpose(reg[:, :, 2]) # 每个像素点是tw
    dy2 = np.transpose(reg[:, :, 3]) # 每个像素点是th
    y, x = np.where(imap >= t) # x,y 为概率大于0.6的像素的的imap矩阵下标,值都为整数
    # 每个像素点概率t > cut_point,筛选出人脸可能性大的像素点 例如：可能有25个 (x,y) 点
    if y.shape[0] == 1:
        # 如果只有1个人头，则np.flipud 对矩阵沿着水平轴上下翻转? 干啥呢？
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    # 取满足pnet阈值的像素点的概率分数和对应像素点的bbox四个点的偏移量
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]])) # 筛选概率大的像素点位置，所对应的x1y1x2y2的概率值取出，例25个框子，每个框子四个坐标点(y,x)
    if reg.size == 0:
        reg = np.empty((0, 3))  # 如果没找到人脸，reg候选框list改为空
    bb = np.transpose(np.vstack([y, x]))  # np.vstack是按照行顺序，把数组给堆叠起来，横着拼在一起，变成25个坐标二元组
    q1 = np.fix((stride * bb + 1) / scale) # q1,q2值应为在原图中每一个预测框的左上角，右下角坐标，np.fix向下取整
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg]) # 一个像素点有9个值，合并hstack到一个list里面，boundingbox.shape=(25, 9) (2>q1+2>q2+1>score+4>reg)>>9
    return boundingbox, reg  # 返回每一个12*12块大小的坐标及对应偏移及该块得分


# function pick = nms(boxes,threshold,type)
def nms(boxes, threshold, method):
    # https: // blog.csdn.net / cuixing001 / article / details / 84946990
    # threshold 越小抛弃候选框越多(重叠0.25以上就抛弃 vs 重叠0.5以上才抛弃)，框越少后续速度越快。因为欢迎系统大多都是单人情况，所以对nms多人要求不高，可以率为降低要求
    # threshold = 0.4
    # method = 'min'
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]  # q1[0] (2>q1+2>q2+1>score+4>reg)>>9 = 0~8
    y1 = boxes[:, 1]  # q1[1] (2>q1+2>q2+1>score+4>reg)>>9 = 0~8
    x2 = boxes[:, 2]  # q2[0] (2>q1+2>q2+1>score+4>reg)>>9 = 0~8
    y2 = boxes[:, 3]  # q2[1] (2>q1+2>q2+1>score+4>reg)>>9 = 0~8
    s = boxes[:, 4]  # score (2>q1+2>q2+1>score+4>reg)>>9 = 0~8
    area = (x2 - x1 + 1) * (y2 - y1 + 1) # 面积都是400左右？ 20*20
    I = np.argsort(s) # 产出每个像素点得分排位的rankid 0~24
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


# function [dy edy dx edx y ey x ex tmpw tmph] = pad(total_boxes,w,h)
def pad(total_boxes, w, h):
    """Compute the padding coordinates (pad the bounding boxes to square)"""
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)  # x2 - x1 = w
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)  # y2 - y1 = w
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32) # 94个1，新box位置x1初始化
    dy = np.ones((numbox), dtype=np.int32) # 94个1，新box位置y1初始化
    edx = tmpw.copy().astype(np.int32) # 94个宽，新box位置x2初始化
    edy = tmph.copy().astype(np.int32) # 94个高，新box位置y2初始化
    # 上边的四个变量“dbox”是用于存放从原始图片抠出来的新box，备注如果回归结果box越界了原始图片，则会取回归结果box和原始图片的交集作为新的box，从而新box的面积会小于等于回归结果box
    x = total_boxes[:, 0].copy().astype(np.int32)  # 94个x1，原始图片上box的位置
    y = total_boxes[:, 1].copy().astype(np.int32)  # 94个y1，原始图片上box的位置
    ex = total_boxes[:, 2].copy().astype(np.int32)  # 94个x2,右下角有可能exceed超出图片的宽度w，原始图片上box的位置
    ey = total_boxes[:, 3].copy().astype(np.int32)  # 94个x2,右下角有可能exceed超出图片的高度h，原始图片上box的位置

    tmp = np.where(ex > w)  # 找到超出最原始图片宽度的bbox的x2，
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)  # 沿x轴翻转， 再右移图片宽度+box宽度， 得到bbox新的宽等于：bbox和最原始图片的交集的宽度
    ex[tmp] = w  # bbox新的右下角x2等于：最原始图片的右下角，即最原始图片的宽度

    tmp = np.where(ey > h)  # 找到超出最原始图片高度的bbox的y2
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)  # 沿y轴反转， 再下移团片高度+box高度
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)  # ？若x[tmp]=-1；2-(-1) = 3 ，超出一格，那么在新的“dbox”上，从下标为3的地方开始填充像素值？
    x[tmp] = 1

    tmp = np.where(y < 1)  # ？
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)  # ？左上角y1 沿x轴反转，下移2格子？
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph  # ([新bbox平移???], [新box下标] ,[bbox旧宽度，bbox旧高度])


# function [bboxA] = rerec(bboxA)
def rerec(bboxA):
    """Convert bboxA to square."""
    h = bboxA[:, 3] - bboxA[:, 1] # y2-y1
    w = bboxA[:, 2] - bboxA[:, 0] # x2-x1
    l = np.maximum(w, h) # 取长边
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5 # x1 向右移动一半宽度w，向左移动长边的长度l
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5 # y1 向下移动一半高度h，向上移动长边的长度l，以上两步使得，短边移动 0.5(l-短边)，使得延申是往短边外的两侧各延伸一半
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1))) # 在新的左上角点上，向右走l步，向下走l步，形成正方形
    return bboxA


def imresample(img, sz):
    # 图片下采样，进行缩小
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)  # @UndefinedVariable
    return im_data

    # This method is kept for debugging purpose
#     h=img.shape[0]
#     w=img.shape[1]
#     hs, ws = sz
#     dx = float(w) / ws
#     dy = float(h) / hs
#     im_data = np.zeros((hs,ws,3))
#     for a1 in range(0,hs):
#         for a2 in range(0,ws):
#             for a3 in range(0,3):
#                 im_data[a1,a2,a3] = img[int(floor(a1*dy)),int(floor(a2*dx)),a3]
#     return im_data
