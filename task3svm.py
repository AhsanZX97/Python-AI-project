from skimage.feature import hog
from scipy.misc import imread,imresize
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle

### Remove forced-depreciation warnings about outdated python modules
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
### End warning removal


if __name__ == '__main__':
    #load the detector
    clf = pickle.load( open("task3.detector","rb"))

    #now load a test image and get the hog features.
    test_image = imread('testsigs/Alfin.png',1) # you can modify which image is tested by changing the filename here
    test_image = imresize(test_image, (200,200))

    hog_features = hog(test_image, orientations=12, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
    hog_features = hog_features.reshape(1, -1)

    # result_type returns a number based on whether the image predicted is a building/mountain/highway
    # building = 0
    # mountain = 1
    # highway  = 2
    result_type = clf.predict(hog_features)


    # we now translate the above result into a string, making the result easier to understand
    if result_type == 0:
	    print 'The SVM identifies this signatre as Brians signature'
    elif result_type == 1:
	    print "The SVM identifies this signatre as Alfins signature"
    elif result_type == 2:
	    print "The SVM identifies this signatre as Sebs signature"
    else:
	    print "Something went wrong"

print '\nFinished identifying'
