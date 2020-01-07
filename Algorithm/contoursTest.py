import cv2
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

morph_size=(3,3)
# load the image
originalImage = cv2.imread("../Data/contourExtra/contourTest3.jpg")
image = cv2.imread("../Data/laplacianOutput.png")
#imageShow = image.copy()
#imageShow = cv2.cvtColor(imageShow, cv2.COLOR_BGR2GRAY)
imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgGray = cv2.threshold(imgGray, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
erosion = cv2.erode(imgGray, (3,3),iterations=2)
#plt.imshow(erosion)
#plt.show()

## DILATE IMAGE
kernel = np.ones((3,3), np.uint8)
dialate = cv2.dilate(erosion, kernel, iterations=5)
#plt.imshow(dialate)
#plt.show()

imThreshold = cv2.threshold(dialate, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#Find bounding boxed
_ ,contours, hierarchy = cv2.findContours(imThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = []

idx = 0
for contour in contours:
    idx += 1
    (x,y,w,h)= cv2.boundingRect(contour)
    if (2 <= (float(w) / float(h)) < 4):
        cv2.rectangle(originalImage, (x,y), (x+w, y+h), (255), 2)
        roi = originalImage[y:y + h, x:x + w]
        cv2.imwrite("Data/contourExtra/contourTest"+str(idx)+".png", roi)

cv2.imshow('img', originalImage)
cv2.waitKey()

