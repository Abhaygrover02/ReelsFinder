#importing libraries
import cv2
import numpy as np
import os

path = 'imgtrain'
orb = cv2.ORB_create()
#reading images
images = []
classNames = ['short', 'notshort' ]
myList = os.listdir(path)
print('Total img Detected', len(myList))

for img in myList:
    imgcur = cv2.imread(f'{path}/{img}',0)
    images.append(imgcur)

def findDes(images):
    desList=[]
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

desList = findDes(images)
print('done')