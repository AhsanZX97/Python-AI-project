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
    path_brian = './sigs/Brian/'
    path_alfin = './sigs/Alfin/'
    path_seb = './sigs/Seb/'

    #get all the images in the above folders
    #note that we are looking for specific extensions (jpg, bmp and png)
    brian_filenames = sorted([filename for filename in os.listdir(path_brian) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
    alfin_filenames = sorted([filename for filename in os.listdir(path_alfin) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
    seb_filenames = sorted([filename for filename in os.listdir(path_seb) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])

    #add the full path to all the filenames
    brian_filenames = [path_brian+filename for filename in brian_filenames]
    alfin_filenames = [path_alfin+filename for filename in alfin_filenames]
    seb_filenames = [path_seb+filename for filename in seb_filenames]

    print 'Number of training images -> brian: ' + str(len(brian_filenames))
    print 'Number of training images -> alfin: ' + str(len(alfin_filenames))
    print 'Number of training images -> seb: ' + str(len(seb_filenames))

    #create the list that will hold ALL the data and the labels
    #the labels are needed for the classification task:
    # 0 -> building
    # 1 -> mountain
    # 2 -> highway
    data = []
    labels = []

    #fill the training dataset
    # the flow is
    # 1) load sample
    # 2) resize it to (200,200) so that we have same size for all the images
    # 3) get the HOG features of the resized image
    # 4) save them in the data list that holds all the hog features
    # 5) also save the label (target) of that sample in the labels list
    for filename in brian_filenames:
        #read the images
        image = imread(filename,1)
        #flatten it
        image = imresize(image, (200,200))
        hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        data.append(hog_features)
        labels.append(0)
    print 'Finished adding brians samples to dataset'

    for filename in alfin_filenames:
        image = imread(filename,1)
        image = imresize(image, (200,200))
        hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        data.append(hog_features)
        labels.append(1)
    print 'Finished adding alfins samples to dataset'

    for filename in seb_filenames:
        image = imread(filename,1)
        image = imresize(image, (200,200))
        hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        data.append(hog_features)
        labels.append(2)
    print 'Finished adding sebs samples to dataset'

    print 'Training the SVM'
    #create the SVC
    clf = LinearSVC(dual=False,verbose=1)
    #train the svm
    clf.fit(data, labels)

    #pickle it - save it to a file
    pickle.dump( clf, open( "task3.detector", "wb" ) )
