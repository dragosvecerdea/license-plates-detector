import cv2
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


# load the image
image = cv2.imread("Data/output.png")

filterSize = 3
Filter = cv2.GaussianBlur(image,(3, 3),cv2.BORDER_DEFAULT)

laplacian = cv2.Laplacian(Filter,cv2.CV_64F)

plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()
