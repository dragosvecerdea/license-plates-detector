import cv2
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


# load the image
image = cv2.imread("Data/colorOutput.png")

filterSize = 5
Filter = cv2.GaussianBlur(image,(filterSize,filterSize),cv2.BORDER_DEFAULT)

laplacian = cv2.Laplacian(Filter,cv2.CV_64F)

cv2.imwrite("Data/laplacianOutput.png", laplacian)
plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()
