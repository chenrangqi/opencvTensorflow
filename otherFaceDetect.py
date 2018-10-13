#!/usr/bin/pyhon
#code --utf-8--

import os
import cv2
# process gif image
#from PIL import Image

haar = cv2.CascadeClassifier("/home/crq/opencv-python/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
size = 64

def createDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def processGifImage():
    gif = cv2.VideoCapture("test.gif")
    ret, frame = gif.read()
    
    if not ret:
        print("not exist gif Image")
        return
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #imshow('img',img)

def processImage(dir):
    count = 1    
    for path, dirsnames, filenames in os.walk(dir):
        for fileName in filenames:
            if fileName.endswith(".jpg"):
                print(path)
                print(fileName)
                img = cv2.imread(path + "/" + fileName)
                #cv2.imshow("img", img)

                #print(img)
                #if img == None
                #    print("imread fail")
                grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #print(grayImg)
                faceImg = haar.detectMultiScale(grayImg, 1.3, 5)

                print(faceImg)
                for (x, y, w, h) in faceImg:
                    print(x, y, w, h)
                    face = img[y:y+h, x:x+w]
                    print(face)
                    face = cv2.resize(face ,(size, size))
                    cv2.imwrite("./otherFace/" + str(count) + ".jpg", face)
                    count += 1

                cv2.waitKey(30)

#def detectFace():
    #haar = cv2.CascadeClassifier("/home/crq/opencv-python/opencv/data/haarcascades_cuda/haarcascade_frontalface_default.xml")
    

if __name__ == "__main__":
    name = raw_input('please input other faces dir Name:')
    otherFaceDir = os.path.join('./image', name)
    
    createDir(otherFaceDir)
    processImage("./otherImg")
    #processGifImage()
    #detectFace();
    
