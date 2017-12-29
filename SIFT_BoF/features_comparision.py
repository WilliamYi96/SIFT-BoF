import numpy as np 
import cv2 
import matplotlib.pyplot as plt

import img_preprocessing
from sift_bows import *

# how to use a real number to be the feature

dogs_feature = sift_main('dog')
cats_feature = sift_main('cat')

x = xrange(0,50)

plt.subplot(2,1,1)
plt.title("Features of Dogs")
plt.plot(x, dogs_feature)

plt.subplot(2,1,2)
plt.title("Features of Cats")
plt.plot(x, cats_feature)
plt.show()


