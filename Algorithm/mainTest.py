import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans


def filterColor(input):
    # load the image
    image = input
    # normalize float versions
    norm_img2 = cv2.normalize(image, None, alpha=0, beta=2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img2 = np.clip(norm_img2, 0, 1)
    norm_img2 = (255 * norm_img2).astype(np.uint8)

    (h, w) = norm_img2.shape[:2]
    im = cv2.cvtColor(norm_img2, cv2.COLOR_BGR2LAB)
    im = im.reshape((im.shape[0] * im.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=12)
    labels = clt.fit_predict(im)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    im = im.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    im = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)

    xp = [0, 64, 128, 192, 255]
    fp = [0, 16, 128, 240, 255]
    x = np.arange(256)
    table = np.interp(x, xp, fp).astype('uint8')
    img = cv2.LUT(quant, table)

    # define the list of boundaries
    boundaries = [
        ([0, 100, 150], [100, 255, 255]),
        # ([0, 0, 0], [50, 50, 50])
    ]

    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)
        return output


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
    image = cv2.dilate(image, kernel, iterations=5)
    canny = auto_canny(image, 0.95)
    return canny


def getPlates(input, colorFiltered):
    morph_size = (3, 3)
    # load the image
    originalImage = input.copy()
    image = applyCanny(colorFiltered)


    image = image.astype('uint8')
    #imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    erosion = cv2.erode(image, (3, 3), iterations=2)

    ## DILATE IMAGE
    kernel = np.ones((3, 3), np.uint8)
    dialate = cv2.dilate(erosion, kernel, iterations=5)

    imThreshold = cv2.threshold(dialate, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]



    # Find bounding boxed
    (_, contours, hierarchy) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    idx = 0
    rois = []
    for contour in contours:
        idx += 1
        (x, y, w, h) = cv2.boundingRect(contour)
        if (2 <= (float(w) / float(h)) < 4):
            cv2.rectangle(originalImage, (x, y), (x + w, y + h), (255), 2)
            roi = originalImage[y:y + h, x:x + w]
            roi = roi / 255.0
            im_power_law_transformation = cv2.pow(roi, 1.5)
            rois.append(im_power_law_transformation)
    return rois


def getChars(input):
    img = input
    img = cv2.resize(img, (img.shape[1] * 5, img.shape[0] * 5), interpolation=cv2.INTER_CUBIC)
    img = crop_img(img, 0.9, 0.7)

    img = cv2.normalize(img, None, alpha=0, beta=1.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    edges = auto_canny(thresh_inv, 0.95)
    kernel = np.ones((3, 3), np.uint8)
    #edges = cv2.erode(edges, kernel, iterations=2)
    #edges = cv2.dilate(edges, kernel, iterations=2)
    (_, ctrs, hierarchy) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0] * img.shape[1]
    chars = []
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w * h
        roi_ratio = roi_area / img_area
        if ((roi_ratio >= 0.02) and (roi_ratio < 0.2)):
            if ((h > 1.2 * w) and (4 * w >= h)):
                cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)
                char = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                char = cv2.threshold(char, 250, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                chars.append(char)
    if len(chars) == 6:
        cv2.imshow("edge", img)
        cv2.waitKey()
        for char in chars:
            print(bestMatch(char))
            char = cv2.medianBlur(char, 7)
            cv2.imshow("char", char)
            cv2.waitKey()
    return chars


def getFrames(inputVid):
    # Path to video file
    video = cv2.VideoCapture(inputVid)
    success, image = video.read()
    count = 0
    plateIdx = 0
    while success:
        # save frame as JPEG file    
        cv2.imwrite("../TestSet/frame%d.jpg" % count, image)
        plates = getPlates(image,filterColor(image))

        for idx, plate in enumerate(plates):
            getChars(plate)
            plateIdx += 1
        print('Read a new frame: ', success)
        count += 1
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
    max = 9999999999
    maxIdx = 1
    diff = []
    for idx in range(1, 18, 1):
        letterRoi = cv2.imread("../SameSizeLetters/" + str(idx) + ".jpg")
        letterRoi = cv2.resize(letterRoi, (image.shape[1], image.shape[0]))
        image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        letterRoi = cv2.cvtColor(letterRoi, cv2.COLOR_BGR2GRAY)
        #letterRoi = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        rows, cols = letterRoi.shape
        # --- take the absolute difference of the images ---
        res = cv2.absdiff(image, letterRoi)
        # --- convert the result to integer type ---
        res = res.astype(np.uint8)
        # --- find percentage difference based on number of pixels that are not zero ---
        percentage = (np.count_nonzero(res) * 100) / res.size
        diff.append(percentage)

    result = np.where(diff == np.amin(diff))
    return result

"""
        countWhile = 0
        countOk = 0
        for i in range(rows):
            for j in range(cols):
                if letterRoi[i,j] == image[i,j]:
                    countOk+=1
        (_, ctrs1, hierarchy) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, ctrs2, hierarchy) = cv2.findContours(letterRoi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        matching = cv2.matchShapes(image, letterRoi, cv2.CONTOURS_MATCH_I1, 0)
        if matching < max:
            max = matching
            maxIdx = idx
    return maxIdx
"""

frames = getFrames("../TrainingSet/Categorie III/Video61_2.avi")


