import os
import cv2
import numpy as np
from get_features import getLbp

def getInputdata():
    x_train = []
    lab_train = []
    x_test = []
    lab_test = []
    for fn in os.listdir('GoodFace/Train'):
        if '.jpg' in fn:
            im = cv2.imread('GoodFace/Train/'+fn)
            f = getLbp(im)
            x_train.append(f)
            lab_train.append([1,0,0])
    print 'GoodFace Train Completed'
    for fn in os.listdir('BadFace/Train'):
        if '.jpg' in fn:
            im = cv2.imread('BadFace/Train/'+fn)
            f = getLbp(im)
            x_train.append(f)
            lab_train.append([0,1,0])
    print 'BadFace Train Completed'
    # for fn in os.listdir('NoFace/Train'):
    #     if '.jpg' in fn:
    #         im = cv2.imread('NoFace/Train/'+fn)
    #         f = getLbp(im)
    #         x_train.append(f)
    #         lab_train.append([0,0,1])
    # print 'NoFace Train Completed'
    # for fn in os.listdir('GoodFace/Test'):
    #     if '.jpg' in fn:
    #         im = cv2.imread('GoodFace/Test/'+fn)
    #         f = getLbp(im)
    #         x_test.append(f)
    #         lab_test.append([1,0,0])
    # print 'GoodFace Test Completed'
    # for fn in os.listdir('BadFace/Test'):
    #     if '.jpg' in fn:
    #         im = cv2.imread('BadFace/Test/'+fn)
    #         f = getLbp(im)
    #         x_test.append(f)
    #         lab_test.append([0,1,0])
    # print 'BadFace Test Completed'
    #
    # for fn in os.listdir('NoFace/Test'):
    #     if '.jpg' in fn:
    #         im = cv2.imread('NoFace/Test/'+fn)
    #         f = getLbp(im)
    #         x_test.append(f)
    #         lab_test.append([0,0,1])
    # print 'NoFace Test Completed'

    return np.array(x_train),np.array(lab_train),np.array(x_test),np.array(lab_test)


if __name__ == '__main__':
    a,b,c,d = getInputdata()
    np.save('Data/enc_x_train.npy',a)
    # np.save('Data/lab_train.npy',b)
    # np.save('Data/x_test.npy',c)
    # np.save('Data/lab_test.npy',d)
