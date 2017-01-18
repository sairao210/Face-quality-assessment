from skimage import feature
import cv2
import numpy as np

def getLbp_bins(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray,(224,224))
	lbp = feature.local_binary_pattern(gray, 24,8, method="default")
	(hist, _) = np.histogram(lbp.ravel(),
		bins=np.arange(0, 24 + 3),
		range=(0, 24 + 2))
	hist = hist.astype("float")
	hist /= (hist.sum() + 1e-7)
	return hist

def getLbp(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	lbp = feature.local_binary_pattern(gray, 16,2, method="default")
	lbp = np.uint8(lbp)
	hist = cv2.calcHist([lbp],[0],None,[256],[0,256])
	hist /= 256
	# x =  np.uint8(hist)
	return hist

if __name__ == '__main__':
	im = cv2.imread('/home/narshima/Documents/Face Quality/GoodFace/Train/img2075.jpg')
	x = getLbp(im)
	print x
	print len(x)
	# im = cv2.resize(im,(100,167))
	print np.shape(im)
