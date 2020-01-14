import os

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
import csv
import sys


charPlate = ['B', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'X', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fps = 0
potentialPlates = []
posted = []

def filterColor(input):


    # load the image
    image = input

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # HSV values to define a colour range.
    colorLow = np.array([0, 90, 90])
    colorHigh = np.array([50, 255, 255])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    # Show the first mask
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    image = cv2.bitwise_and(image, image, mask=mask)

    #cv2.imshow("color", image)
    #cv2.waitKey()
    return image




def auto_canny(image, sigma):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def applyCanny(input):
    # load the image
    image = input.copy()
    ## DILATE IMAGE
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=4)
    canny = auto_canny(image, 0.95)

    return canny


def rotate(angle, image):
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def getPlates(input, colorFiltered):

    # load the image
    originalImage = input.copy()
    image = applyCanny(colorFiltered)

    ## DILATE IMAGE
    kernel = np.ones((3, 3), np.uint8)

    image = image.astype('uint8')
    image = cv2.dilate(image, kernel, iterations=3)

    #cv2.imshow("img", image)
    #cv2.waitKey()

    # Find bounding boxed
    (_, contours, hierarchy) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if (1 <= (float(w) / float(h)) < 5):
            angle = cv2.minAreaRect(contour)[-1]
            rotated = rotate(angle,originalImage)
            roisRotated = getPlatesOnRotated(rotated)
            for roii in roisRotated:
                rois.append(roii)
    return rois


def getPlatesOnRotated(input):


    #cv2.imshow("rot", input)
    #cv2.waitKey()
    colorFiltered = filterColor(input)
    # load the image
    originalImage = input.copy()
    image = applyCanny(colorFiltered)

    ## DILATE IMAGE
    kernel = np.ones((3, 3), np.uint8)

    image = image.astype('uint8')
    image = cv2.dilate(image, kernel, iterations=3)

    # Find bounding boxed
    (_, contours, hierarchy) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    idx = 0
    rois = []
    for contour in contours:
        idx += 1
        (x, y, w, h) = cv2.boundingRect(contour)
        if (1 <= (float(w) / float(h)) < 5):
            roi = originalImage[y:y + h, x:x + w]
            roi = roi / 255.0
            roi = cv2.pow(roi, 1.8)
            im_power_law_transformation = cv2.normalize(roi, None, alpha=0, beta=1.9, norm_type=cv2.NORM_MINMAX,
                                                        dtype=cv2.CV_32F)
            rois.append(im_power_law_transformation)
    return rois


def getChars(input):
    img = input
    img = cv2.resize(img, (img.shape[1] * 4, img.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    img = crop_img(img, 0.9, 0.9)

    #img = cv2.normalize(img, None, alpha=0, beta=1.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #thresh_inv = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 25)
    kernel = np.ones((3,3), np.uint8)
    #cv2.imshow("debug", thresh_inv)
    #cv2.waitKey()


    edges = auto_canny(thresh_inv, 0.95)

    edges = cv2.dilate(edges, kernel, iterations=1)

    #cv2.imshow("edges", edges)
    #cv2.waitKey(0)


    (_, ctrs, hierarchy) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0] * img.shape[1]
    chars = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w * h
        roi_ratio = roi_area / img_area
        if ((roi_ratio >= 0.02) and (roi_ratio < 0.2)):
            if ((h > 1 * w) and (5 * w >= h)):
                char = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                char = cv2.threshold(char, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                char = cv2.dilate(char, kernel, iterations=1)
                #cv2.imshow("char", char)
                #cv2.waitKey()
                chars.append(char)
    #.imshow("cha4", img)
    #cv2.waitKey(0)
    PLATE = []
    confident = True
    if len(chars) == 6:
        for char in chars:
            char = cv2.bilateralFilter(char, -1, 20, 20)
            #char = cv2.erode(char, (3, 3), iterations=2)
            curr_confidence = bestMatch(char)
            if curr_confidence == False:
                return
            if curr_confidence[2] < 0.8:
                confident = False
            PLATE.append(bestMatch(char))
        if confident:
            postPlate(PLATE)
        else:
            potentialPlates.append(getPlate(PLATE))
            if potentialPlates.count(getPlate(PLATE)) == 3:
                print(getPlate(PLATE))
                postPlate(PLATE)


def getPlate(plate):
    PLATE = []
    global count
    counter = 0
    PLATE.append(plate[0][0])
    for idx in range(1,6,1):
        if (isLetter(plate[idx]) and not isLetter(plate[idx-1])) or (not isLetter(plate[idx]) and isLetter(plate[idx-1])):
            PLATE.append('-')
            counter += 1
        PLATE.append(plate[idx][0])

    if counter == 1:
        if PLATE[2] == '-':
            PLATE.insert(5,'-')
        else:
            PLATE.insert(2,'-')
    if counter > 2:
        return False
    licensePlate = ""
    for ele in PLATE:
        licensePlate += ele
    return licensePlate

def postPlate(plate):
    licensePlate = getPlate(plate)
    if licensePlate == False:
        return
    posted.append(licensePlate)
    if posted.count(licensePlate) > 1:
        return

    csvRow = [licensePlate, count, (1/fps)*count]
    csvfile = "../results.csv"

    with open(csvfile, "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(csvRow)

def isLetter(plateN):
    if plateN[1] < 17:
        return True
    return False

def getFrames(inputVid):
    # Path to video file
    global count
    global fps
    video = cv2.VideoCapture(inputVid)
    success, image = video.read()
    plateIdx = 0

    headers = ['License plate', 'Frame no.', 'Timestamp(seconds)']

    f = open("../results.csv", "w")
    writer = csv.DictWriter(f, fieldnames=['License plate', 'Frame no.', 'Timestamp(seconds)'])
    writer.writeheader()
    f.close()

    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)

    while success:
        # save frame as JPEG file    
        if count % 7 == 0:
            #if count == 560:
                plates = getPlates(image, filterColor(image))
                for idx, plate in enumerate(plates):
                    getChars(plate)
                    plateIdx += 1
        count += 1
        if count == 1731:
            sys.exit()
        success, image = video.read()

    return count


def crop_img(img, scaleX=1.0, scaleY=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scaleX, img.shape[0] * scaleY
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def bestMatch(image):
    diff = []
    image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    for idx in range(1, 18, 1):
        letterRoi = cv2.imread("../SameSizeLetters/" + str(idx) + ".jpg")
        letterRoi = cv2.resize(letterRoi, (image.shape[1], image.shape[0]))
        letterRoi = cv2.cvtColor(letterRoi, cv2.COLOR_BGR2GRAY)
        letterRoi = cv2.threshold(letterRoi, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        rows, cols = letterRoi.shape

        percentage = matchCheckerDiff(image, letterRoi)
        diff.append(percentage)

    for idx in range(0,10,1):
        numberRoi = cv2.imread("../SameSizeNumbers/" + str(idx) + ".jpg")
        numberRoi = cv2.resize(numberRoi, (image.shape[1], image.shape[0]))

        numberRoi = cv2.cvtColor(numberRoi, cv2.COLOR_BGR2GRAY)
        numberRoi = cv2.threshold(numberRoi, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        rows, cols = numberRoi.shape

        percentage = matchCheckerDiff(image, numberRoi)
        diff.append(percentage)


    result = np.argmax(diff)
    maxx = np.max(diff)
    if maxx > 0.70:
        return (charPlate[result], result, maxx)
    return False


def matchCheckerDiff(character, template):

    (rows, cols) = template.shape
    countOk = 0

    for i in range(rows):
        for j in range(cols):
            if character[i, j] == template[i, j]:
                countOk += 1

    return countOk/(rows*cols)

count = 0
frames = getFrames("../trainingsvideo.avi")


