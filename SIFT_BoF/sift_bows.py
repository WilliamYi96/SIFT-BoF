#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib.pyplot as plt
import random
import cv2
import math

from img_preprocessing import *

NUM_IMAGES = 3

def euclDistance(vector1, vector2):
    return sum(abs(vector2 - vector1))

def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape #　numSamples=2493, dim=128
    centroids = np.zeros((k + 1, dim))
    s = set()
    for i in range(1, k + 1):
        while True:
            index = int(random.uniform(0, numSamples))
            if index not in s:
                s.add(index)
                break
        # index = int(random.uniform(0, 2))
        # print "random index:"
        # print index
        centroids[i, :] = dataSet[index, :]
    return centroids

def sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # find the keypoint and descriptors with SIFT
    kp, dataSet = sift.detectAndCompute(np.uint8(img),None)
    # pic = cv2.drawKeypoints(imgs[i], kp, imgs) 
    return kp, dataSet

def kmeans(imgs, k):
    kps, dataSets = [], []
    for i in range(NUM_IMAGES):
        kp, dataSet = sift(imgs[i])
        # for j in range(len(kp)):
        kps.append(kp)
        dataSets.append(dataSet)
        # print 'Hello'
        # print dataSet.shape
    dataSets = np.array(dataSets)
    resultSet = np.vstack((dataSets[0], dataSets[1], dataSets[2]))
    # print  resultSet.shape
    # print(dataSet.shape[0],dataSet.shape[1]) #　(2493, 128)
    numSamples = resultSet.shape[0] #　2493
    clusterAssment = np.mat(np.zeros((numSamples, 2)))
    for i in xrange(numSamples):
        clusterAssment[i, 0] = -1
    clusterChanged = True
    centroids = initCentroids(resultSet, k)
    while clusterChanged:
        clusterChanged = False
        for i in xrange(numSamples):
            minDist = 100000.0
            minIndex = 0
            for j in range(1, k + 1):
                distance = euclDistance(centroids[j, :], resultSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist
            else:
                clusterAssment[i, 1] = minDist
        # print "clusterAssment before:"
        # print clusterAssment
        for j in range(1, k + 1):
            pointsInCluster = resultSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis=0)
            print centroids.shape

    print 'Congratulations, cluster complete!'
    # print centroids
    return centroids, clusterAssment

def sift_main(categories):
        
    # img = cv2.imread("/home/iair002/Downloads/Slide_2.jpg")
    imgs = img_processing(categories)

    print "step 2: clustering..."
    k = 50
    centroids, clusterAssment = kmeans(imgs, k)
    result = np.zeros((k), dtype=np.int16)
    for i in range(clusterAssment.shape[0]):
        categories = int(clusterAssment[i,0]-1)
        # print categories
        result[[categories]] += 1
    # print result
    return sorted(result)

# # bins = np.arange(-2, 52, 1)
# bins = 3
# plt.xlim(0, 50)
# plt.ylim(min(result)-5, max(result)+5)
# plt.hist(result, bins=bins, alpha=0.5)
# plt.title('feature distribution')
# plt.xlabel('feature description')
# plt.ylabel('frequency')
# plt.show()

# print "center:"
# print centroids
# print "clusterAssment:"
# print clusterAssment

# x = xrange(0, 50)
# y = sorted(result)
# plt.plot(x, y)
# plt.show()
# plt.savefig()




