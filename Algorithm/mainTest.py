import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans


def colorTest(input):
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


def laplacianTest(input):
    # load the image
    image = colorTest(input)
    ## DILATE IMAGE
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel, iterations=5)
    laplacian = auto_canny(image, 0.33)


def contoursTest(input):
    morph_size = (3, 3)
    # load the image
    originalImage = input
    image = laplacianTest(originalImage)

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.threshold(imgGray, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    erosion = cv2.erode(imgGray, (3, 3), iterations=2)

    ## DILATE IMAGE
    kernel = np.ones((3, 3), np.uint8)
    dialate = cv2.dilate(erosion, kernel, iterations=5)

    imThreshold = cv2.threshold(dialate, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find bounding boxed
    _, contours, hierarchy = cv2.findContours(imThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    idx = 0
    for contour in contours:
        idx += 1
        (x, y, w, h) = cv2.boundingRect(contour)
        if (2 <= (float(w) / float(h)) < 4):
            cv2.rectangle(originalImage, (x, y), (x + w, y + h), (255), 2)
            roi = originalImage[y:y + h, x:x + w]
            cv2.imwrite("../Data/contourExtra/contourTest" + str(idx) + ".png", roi)

def charTest(input):
    img = cv2.imread("../Data/contourExtra/contourTest3.jpg")
    img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)

    img = cv2.normalize(img, None, alpha=0, beta=1.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 39, 1)
    edges = auto_canny(thresh_inv, 0.1)
    kernel = np.ones((3, 3), np.uint8)
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

    cv2.imwrite("../Data/contourExtra/chars.jpg", img)

# Path to video file
vidObj = cv2.VideoCapture("../TrainingSet/Categorie III/Video100_2.avi")

# Used as counter variable
count = 0
# checks whether frames were extracted
success = 1

while success:
    success, image = vidObj.read()
    # Saves the frames with frame-count
    cv2.imwrite("TestSet/frame" + str(count) + ".jpg", image)
    bbox = contoursTest("../TestSet/frame" + str(count) + ".jpg")
    count += 1

