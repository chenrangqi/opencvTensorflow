#!/usr/bin/python
#coding=utf-8

import os
import random
import numpy as np
import cv2

def createDir(*args):
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

size = 64

def getFaceFromCamera(outDir):
    createDir(outDir)
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier("/home/crq/opencv-python/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
    #haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 1
    while 1:
        if count <= 200:
            print('process...')
            success, img = camera.read()
            if img == None:
                print("img is none")
                continue

            print(success)
#            cv2.imshow('img', img)
            grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#            cv2.imshow('img', grayImage)
#           key = cv2.waitKey(30)
            faceImage = haar.detectMultiScale(grayImage, 1.3, 5)
#            print(faceImage)
            for f_x,f_y,f_w,f_h in faceImage:
                print(f_x, f_y, f_w, f_h)
                
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                #print(img)
                #cv2.imshow('source', img)
                #cv2.imshow('img', face)
                face = cv2.resize(face, (size, size))
                cv2.imwrite(os.path.join(outDir, str(count)+'.jpg'), face)
                #print(img)
                if img == None:
                    print("before recangle img is none")                
                cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                if img == None:
                    print("rectangle fail")
                count += 1            
                #cv2.imshow('img', img)
            if img == None:
                print("img is none")
                continue
            cv2.imshow('source', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
             
        else:
            break
    camera.release()
    cv2.destoryAllWindows()

if __name__ == '__main__':
    name =raw_input('please input your name:')
    getFaceFromCamera(os.path.join('./image/trainfaces', name))
   # getFaceFromCamera(os.path.join('./image/trainfaces',name))
