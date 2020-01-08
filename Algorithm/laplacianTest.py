import cv2
import numpy as np

# load the image
image = cv2.imread("../Data/colorOutput.png")
## DILATE IMAGE
kernel = np.ones((3,3), np.uint8)
image = cv2.dilate(image, kernel, iterations=5)


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


laplacian = auto_canny(image)
cv2.imwrite("Data/laplacianOutput.png", laplacian)
cv2.imshow('img', laplacian)
cv2.waitKey()

