#!/usr/bin/python
#coding ---utf-8---

import tensorflow as tf
import cv2
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

myFacePath = "./myFace"
otherFacePath = "./otherFace"
size = 64

#----------get padding size of img----------------#
def getPaddingSize(img):
    h, w, _ = img.shape
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

imgs = []
labels = []

#-----------deal with image-----------#
def dealImg(path, w=size, h=size):
    for fileName in os.listdir(path):
        if fileName.endswith(".jpg"):
            imgPath = path + "/" + fileName
            img = cv2.imread(imgPath)
            top, bottom, left, right = getPaddingSize(img)
            #use constant to expand edge
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = [0,0,0])
            imgs.append(img)
            labels.append(path)
#----------读取 image ---------------#
dealImg(myFacePath)
print("my Face is deal done")
dealImg(otherFacePath)
print("other face is deal done")
#------将图片数据和标签转换成数组----#
imgs = np.array(imgs)
labels = np.array([[0,1] if lab == myFacePath else [1,0] for lab in labels])

#------划分测试数据和训练数据--------#
train_imgs, test_imgs, train_labels, test_labels = train_test_split( imgs, labels, test_size = 0.001, random_state = random.randint(0, 100) )

print("split data complete")
#-------------数据换成图片总数、图片大小、图片通道------------------#
train_imgs = train_imgs.reshape(train_imgs.shape[0], size, size, 3)
test_imgs  = test_imgs.reshape( test_imgs.shape[0] , size, size ,3)

#-------------数据换成图片变成[0,1]之间的数------------------#
train_imgs = train_imgs.astype('float32')/255.0
test_imgs  = test_imgs.astype('float32')/255.0

batch_size = 10
batch_num = len(train_imgs) // batch_size

print("batch num is ", batch_num)

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])
#------------------------------------#

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

#---------weightVariable-----------#
#------生成随机权重-----#
def weightVariable(shape):
    weight = tf.random_normal(shape, stddev = 0.01)
    return tf.Variable(weight)

#---------biasVariable----------#
#生成随机偏置
def biasVariable(shape):
    bias = tf.random_normal(shape)
    return tf.Variable(bias)
#---------conv2d---------#
#卷积函数
def conv2d(input, w):
    return tf.nn.conv2d(input, w, strides = [1,1,1,1], padding = "SAME")
#---------maxPool--------#
#池化函数
def maxPool(shape):
    #ksize = [bath,height,width,channels] 一般为 [1, height,width, 1]
    return tf.nn.max_pool(shape,ksize = [1,2,2,1], strides=[1,2,2,1], padding = "SAME")

#----------define cnn----------------#
def cnnLayer():
    #----第一层 输入层----#
    #权重 卷积核大小为3×3,输入通道为3,输出通道为32
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
    w_fc = weightVariable([8*8*64,512])   #因为原图为64×64 每幅图片经过三次池化为8×8 64通道，经过全连接层为batch × 512
    b_fc = biasVariable([512])
    drop_flat = tf.reshape(drop, [-1,8*8*64])
    fc = tf.nn.relu(tf.matmul(drop_flat, w_fc) + b_fc)
    
    #------输出层------#
    w_out = weightVariable([512, 2])  #分类数为2
    b_out = biasVariable([2])
    out = tf.add(tf.matmul(fc, w_out), b_out)

    return out

#----------------训练---------------#
def cnnTrain():
    out = cnnLayer()
    print("out shape is ", out.shape)
    #用输出和实际想得到的结果，采用softmax函数归一化，并且采用交叉熵函数作为损失函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y_))
    #梯度下降法得到步长
    train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
    #这里用到了输出 tf.argmax返回最大值的索引号，tf.euqal对比两个矩阵元素是否相等，相应元素返回true，返回矩阵维度一样
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1))
    #求平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #tensorBoard可视化
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    
    #数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #用于可视化
        #summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(1):
            for i in range(batch_num):
                batch_imgs = train_imgs[i*batch_size : (i+1)*batch_size]
                batch_labels = train_labels[i*batch_size : (i+1)*batch_size]
                #开始训练数据，同时训练三个变量，返回三个数据
                _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op], \
                                    feed_dict = {x:batch_imgs, y_:batch_labels, keep_prob_5:0.5, keep_prob_75:0.75})
                print(n*batch_num + i, loss)

                accur = accuracy.eval({x:test_imgs, y_:test_labels, keep_prob_5:1.0, keep_prob_75:1.0})
                print('after 10 times run : accuracy is ', accur)
        
        saver.save(sess, './train_face.model')


cnnTrain()

if __name__ == "__main__":
    pass
