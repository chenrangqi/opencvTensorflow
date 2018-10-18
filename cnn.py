#!/usr/bin/python
#coding ---utf-8---

import tensorflow as tf
import os

myFacePath = "./myFaces"
otherFacePath = "./otherFaces"
size = 64

#----------get padding size of img----------------#
def getPaddingSize(img):
    h, w, _ = img.shape()
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)
    if h < longest:
        tmp = longest - h
        #---- "//" is equal int/int in c++ ----#
        top = tmp // 2
        bottom = tmp - top 
    elif w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp -left
    else:
        pass
    return top, bottom, left, right

#-----------deal with image-----------#
def dealImg(path, w=size, h=size):
    for fileName in os.listdir(path):
        if fileName.endswith(".jpg"):
            imgPath = path + "/" + fileName
            img = cv2.imread(imgPath)
            top, bottom, left, right = getPaddingSize(img)
            #use constant to expand edge
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = [0,0,0])

keep_prob_5 = tf.placeholder(tf.float32)

#---------weightVariable-----------#
#------生成随机权重-----#
def weightVariable(shape):
    weight = tf.random_normal(shape, stddev = 0.01)
    return tf.Variable(weight)

#---------biasVariable----------#
#-----生成随机偏置-----#
def biasVariable(shape):
    bias = tf.random_normal(shape)
    return tf.Variable(bias)
#---------conv2d---------#
#----卷积函数----#
def conv2d(input, w):
    return tf.nn.conv2d(input, w, strides = [1,1,1,1], padding = "SAME")
#---------maxPool--------#
#----池化函数----#
def maxPool(shape):
    #ksize = [bath,height,width,channels] 一般为 [1, height,width, 1]
    return tf.nn.max_pool(shape,ksize = [1,2,2,1], strides=[1,2,2,1], padding = "SAME")

#----------define cnn----------------#
def cnnLayer():
    #----第一层 输入层----#
    #权重
    weight1 = weightVariable([3, 3, 3, 32])
    #偏置
    bias1 = biasVariable([32])
    #卷积操作 与 relu激活
    conv1 = tf.nn.relu(conv2d(x, weight1) + bias1)
    #池化操作
    pool1 = maxPool(conv1)

    #----第二层 ----#
    #权重
    weight2 = weightVariable([3, 3, 32, 64])
    #偏置
    bias2 = biasVariable([64])
    #卷积操作 与 relu激活
    conv2 = tf.nn.relu(conv2d(pool1, weight2) + bias2)
    #池化操作
    pool2 = maxPool(conv2)
    
    #----第三层----#
    #权重
    weight3 = weightVariable([3, 3, 64, 64])
    #偏置
    bias3 = biasVariable([64])
    #卷积操作 与 relu激活
    conv3 = tf.nn.relu(conv2d(pool2, weight3) + bias3)
    #池化操作
    pool3 = maxPool(conv3)
    
    #dropout减少过拟合
    drop = tf.nn.dropout(pool3, keep_prob_5)
    
    #-----全连接层-----#
    w_fc = weightVariable([8*8*64,512])
    b_fc = biasVariable([512])
    drop_flat = tf.reshape(drop, [-1,64*32*64])
    fc = tf.nn.relu(tf.nn.matmul(drop_flat, w_fc) + b_fc)
    
    #------输出层------#
    w_out = weightVariable([512, 2])  #分类数为2
    b_out = biasVariable([2])
    out = tf.nn.add(tf.nn.matmul(fc, w_out) + b_out)

    return out


if __name__ == "__main__":
    pass
