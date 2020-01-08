import cv2
import numpy as np

img = cv2.imread("../Data/contourExtra/contourTest3.jpg")
img = cv2.resize(img, (img.shape[1]*3,img.shape[0]*3), interpolation = cv2.INTER_CUBIC)

img = cv2.normalize(img, None, alpha=0, beta=1.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

img = np.clip(img, 0, 1)
img = (255*img).astype(np.uint8)


def auto_canny(image, sigma=0.1):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

     # return the edged image
    return edged


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
edges = auto_canny(thresh_inv)
kernel = np.ones((3,3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
cv2.imshow("tr", edges)
cv2.imshow("img", edges)
(_, ctrs, hierarchy) = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
img_area = img.shape[0] * img.shape[1]
for i, ctr in enumerate(ctrs):
    if hierarchy[0][i][3] == -1:
        continue
    x, y, w, h = cv2.boundingRect(ctr)
    roi_area = w * h
    roi_ratio = roi_area / img_area
    if ((roi_ratio >= 0.01) and (roi_ratio < 0.1)):
        if ((h > 1.2 * w) and (4 * w >= h)):
            minim = 100
            index = 0
            cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)
            char = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            char = cv2.adaptiveThreshold(char, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
            cv2.imshow('char', char)
            print(index, minim)

cv2.imwrite("../Data/contourExtra/chars.jpg", img )
cv2.waitKey(0)