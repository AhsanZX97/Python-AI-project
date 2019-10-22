import numpy as np
import os
import itertools
import operator
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature import hog
from skimage import color, exposure
from scipy.misc import imread,imsave,imresize
import numpy.random as nprnd
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib
import pickle

if __name__ == '__main__':
    #paths for the training samples
    root = './training writing/'

    data = []
    image = []
    target = []

    for path, subdirs, files in sorted(os.walk(root)):
        for name in files:
            image.append(os.path.join(path, name))

    target += [1] * 2
    target += [2] * 6
    target += [3] * 4
    target += [4] * 4
    target += [5] * 6
    target += [6] * 7
    target += [7] * 5
    target += [8] * 6
    target += [9] * 4
    target += ['a'] * 12
    target += ['b'] * 9
    target += ['c'] * 11
    target += ['d'] * 11
    target += ['e'] * 12
    target += ['f'] * 12
    target += ['g'] * 12
    target += ['h'] * 13
    target += ['i'] * 6
    target += ['j'] * 9
    target += ['k'] * 13
    target += ['l'] * 12
    target += ['m'] * 11
    target += ['n'] * 12
    target += ['o'] * 12
    target += ['p'] * 10
    target += ['q'] * 10
    target += ['r'] * 10
    target += ['s'] * 10
    target += ['t'] * 13
    target += ['u'] * 10
    target += ['v'] * 10
    target += ['w'] * 10
    target += ['x'] * 10
    target += ['y'] * 11
    target += ['z'] * 7

    print 'Number of training images ->' + str(len(image))

    #create the list that will hold ALL the data and the labels
    #the labels are needed for the classification task:
    # 0 -> building
    # 1 -> mountain
    # 2 -> highway


    #fill the training dataset
    # the flow is
    # 1) load sample
    # 2) resize it to (200,200) so that we have same size for all the images
    # 3) get the HOG features of the resized image
    # 4) save them in the data list that holds all the hog features
    # 5) also save the label (target) of that sample in the labels list
    for filename in image:
        #read the images
        imagez = imread(filename,1)
        #flatten it
        imagez = imresize(imagez, (200,200))
        hog_features = hog(imagez, orientations=12, pixels_per_cell=(8,8),
                    cells_per_block=(1, 1))
        data.append(hog_features)
    print 'Finished adding buildings samples to dataset'

    print 'Training the SVM'
    #create the SVC
    clf = LinearSVC(dual=False,verbose=1)
    #train the svm
    clf.fit(data, target)

    #pickle it - save it to a file
    pickle.dump( clf, open( "place.detector", "wb" ) )
