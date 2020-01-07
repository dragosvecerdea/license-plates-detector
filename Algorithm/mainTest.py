import numpy as np
import cv2
import sys

def boundingBoxes(inputImage):
    # load the image
    image = cv2.imread(inputImage)

    # define the list of boundaries
    boundaries = [
        ([0, 100, 210], [100, 255, 255])
    ]

    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        colorOutput = cv2.bitwise_and(image, image, mask=mask)

    filterSize = 5
    Filter = cv2.GaussianBlur(colorOutput, (filterSize, filterSize), cv2.BORDER_DEFAULT)

    laplacianOutput = cv2.Laplacian(Filter, cv2.CV_64F)


    morph_size = (3, 3)
    # load the image

    image = laplacianOutput

    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.threshold(imgGray, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    erosion = cv2.erode(imgGray, (3, 3), iterations=2)

    ## DILATE IMAGE
    kernel = np.ones((3, 3), np.uint8)
    dialate = cv2.dilate(erosion, kernel, iterations=5)

    imThreshold = cv2.threshold(dialate, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find bounding boxed
    contours, hierarchy = cv2.findContours(imThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    idx = 0
    for contour in contours:
        idx += 1
        (x, y, w, h) = cv2.boundingRect(contour)
        if (2.5 <= (float(w) / float(h)) < 10):
            cv2.rectangle(inputImage, (x, y), (x + w, y + h), (255), 2)
            if (idx <= 2):
                roi = inputImage[y:y + h, x:x + w]
                #cv2.imwrite("Data/contourTest" + str(idx) + ".png", roi)
                return roi


# Path to video file
vidObj = cv2.VideoCapture("TrainingSet/Categorie III/Video100_2.avi")

# Used as counter variable
count = 0

# checks whether frames were extracted
success = 1

while success:
    success, image = vidObj.read()

    # Saves the frames with frame-count
    cv2.imwrite("TestSet/frame" + str(count) + ".jpg", image)
    count += 1

while count != 0:
    bbox = boundingBoxes("TestSet/frame" + str(count) + ".jpg")
    cv2.imwrite("TestSet/NumberPlates/contourTest" + str(count) + ".jpg", bbox)
    count -= 1




#OpenCV object-Tracker
#USE KCF !!!

"""
# Set up tracker.
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
tracker = cv2.Trac
# Read video
video = cv2.VideoCapture("videos/chaplin.mp4")

# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

# Define an initial bounding box
bbox = boundingBoxes(frame)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
# Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break
    
"""