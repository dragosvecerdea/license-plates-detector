import cv2
import numpy as np
letters = []
numbers = []

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

for idx in range(1,18,1):
    img = cv2.imread("../SameSizeLetters/" + str(idx) + ".bmp")
    img = cv2.copyMakeBorder(img, 10,10,10,10, cv2.BORDER_CONSTANT)
    letterImg = auto_canny(img)
    #letterImg = cv2.cvtColor(letterImg, cv2.COLOR_BGR2GRAY)

    cv2.imshow("img", img)
    cv2.waitKey()
    (_, ctrs , hierarchy) = cv2.findContours(letterImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(ctrs[0])
    letterRoi = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../SameSizeLetters/" + str(idx) + ".jpg", letterRoi)

