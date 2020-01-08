import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

# load the image
image = cv2.imread("../TestSet/frame0.jpg")


# normalize float versions

norm_img2 = cv2.normalize(image, None, alpha=0, beta=2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

norm_img2 = np.clip(norm_img2, 0, 1)
norm_img2 = (255*norm_img2).astype(np.uint8)

# write normalized output images

#cv2.imwrite("zelda1_bm20_cm20_normalize2.jpg",norm_img2)

#cv2.imshow("image", np.hstack([image, norm_img2]))
#cv2.waitKey(0)
#cv2.destroyAllWindows()

(h, w) = norm_img2.shape[:2]

# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
im = cv2.cvtColor(norm_img2, cv2.COLOR_BGR2LAB)

# reshape the image into a feature vector so that k-means
# can be applied
im = im.reshape((im.shape[0] * im.shape[1], 3))

# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions
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
#cv2.imwrite("Data/colorOutput.png", img)
# display the images and wait for a keypress
#cv2.imshow("image", np.hstack([im, img]))
#cv2.waitKey(0)



# define the list of boundaries
boundaries = [
	([0,100,150], [100, 255, 255]),
	#([0, 0, 0], [50, 50, 50])
]

for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype="uint8")
	upper = np.array(upper, dtype="uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(img, lower, upper)
	output = cv2.bitwise_and(img, img, mask=mask)
	cv2.imwrite("Data/colorOutput.png", output)
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)

	#save output image
