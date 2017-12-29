from __future__ import print_function
from __future__ import absolute_import

import numpy as np 
from skimage import data_dir, io, transform, color
import cv2
import matplotlib.pyplot as plt 

from skimage import data_dir

data_dir = '/home/iair002/Desktop/SIFT_BoF/data/pattern/' 
# strr = data_dir + '/*.*'
# strr = data_dir + '/*.jpg'
# print(strr)
print(data_dir)

def img_preprocessing(funs, **args):
    rgb = io.imread(funs)
    rgb = cv2.resize(rgb, (256,256))
    # gray = color.rgb2gray(rgb)
    # dst = transform.resize(gray, (256*256))
    # return dst
    return rgb

def img_processing(categories='cat'):
    print(data_dir + categories + '/*.*')
    strr = data_dir + categories
    coll = io.ImageCollection(strr+'/*.*', load_func=img_preprocessing)
    for i in range(len(coll)):
        io.imsave(strr+'/gray/'+np.str(i)+'.jpg',coll[i])
    # print(coll[0].shape)
    # print(len(coll))

    return coll

# imgs = img_processing()
# sift, gray, kp, pic, dataSet = [[[]]], [[[]]], [[[]]], [[[]]], [[[]]]
# for i in range(3):
#     # initiate SIFT detector
#     sift[i] = cv2.xfeatures2d.SIFT_create()
#     gray[i]= cv2.cvtColor(imgs[i],cv2.COLOR_BGR2GRAY)
#     # find the keypoint and descriptors with SIFT
#     kp[i], dataSet[i] = sift[i].detectAndCompute(gray[i],None)
#     pic[i] = cv2.drawKeypoints(gray[i], kp[i], imgs[i])
#     cv2.imwrite('sift_keypoints.jpg', pic[i])

if __name__ == '__main__':
    img_processing()
    