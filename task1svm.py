from skimage.feature import hog
from scipy.misc import imread,imresize
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
import warnings
from random import randrange
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
from ffnet import ffnet, mlgraph
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage.measure import label
import os, os.path
from PIL import Image
from Tkinter import Tk
from tkFileDialog import askopenfilename
import difflib

### Remove forced-depreciation warnings about outdated python modules
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
### End warning removal


class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename,1)

        #apply threshold in order to make the image binary
        bw = image < 120

        # remove artifacts connected to image border
        cleared = bw.copy()
        #clear_border(cleared)

        # label image regions
        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1


        fig = plt.figure()
        #ax = fig.add_subplot(131)
        #ax.imshow(bw, cmap='jet')

        letters = list()
        order = list()

        for region in regionprops(label_image):
            # 50 for adobe.png
            # 30 for shazam.png
            if region.area > 50:
                minc, minr, maxc, maxr = region.bbox
                # skip small images
                if maxc - minc > len(image)/250: # better to use height rather than area.
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                    order.append(region.bbox)


        #sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        #worst case scenario there can be 1 character per line
        for x in range(len(order)):
            lines.append([])

        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                first_in_line = character
                counter += 1
                lines[counter].append(character)


        for x in range(len(lines)):
            lines[x].sort(key=lambda tup: tup[1])

        final = list()
        prev_tr = 0
        prev_line_br = 0
        lastx = 0;
        lasty = 0;

        for i in range(len(lines)):
            for j in range(len(lines[i])):
                tl_2 = lines[i][j][1]
                bl_2 = lines[i][j][0]
                if tl_2 > prev_tr and bl_2 > prev_line_br:
                    if bl_2 - lasty > 15 and i > 0:
                        final.append(' ')
                    if lastx !=0 and tl_2 - lastx > 40: # 40 adobe.png 30 shazam.png 26 page.png
                        final.append(' ')
                    tl,tr,bl,br = lines[i][j]
                    letter_raw = bw[tl:bl,tr:br]
                    letter_norm = imresize(letter_raw ,(20 ,20))
                    final.append(letter_norm)
                    prev_tr = lines[i][j][3]
                if j == (len(lines[i])-1):
                    prev_line_br = lines[i][j][2]
                if bl_2 - lasty < 15 :
                    lastx = lines[i][j][1]
                else :
                    lastx = 0
                lasty = bl_2

                prev_tr = 0
                tl_2 = 0
        print 'Characters recognized: ' + str(len(final))

        return final


if __name__ == '__main__':
    try:
        extract = Extract_Letters()
        Tk().withdraw()
        filename = askopenfilename() # load the image
        print(filename)

        Tk().withdraw()
        filename2 = askopenfilename() # load the image
        print(filename2)

        letters = extract.extractFile(filename)
        char = ''
        warnings.simplefilter(action='ignore', category=FutureWarning)
        for i in letters:
            if i == ' ' :
                char += ' '
            else :
                found = 0
                letter =0

                clf = pickle.load( open("task1.detector","rb"))

                #now load a test image and get the hog features.
                test_image = imresize(i, (200,200))

                hog_features = hog(test_image, orientations=12, pixels_per_cell=(8, 8),
                                cells_per_block=(1, 1))
                hog_features = hog_features.reshape(1, -1)

                # result_type returns a number based on whether the image predicted is a building/mountain/highway
                # building = 0
                # mountain = 1
                # highway  = 2
                result_type = clf.predict(hog_features)
                char += result_type[0]
        print char
    	orig = "";

        with open(filename2, 'r') as myfile:
    	    orig = myfile.read()
        accuracy = difflib.SequenceMatcher(None,char,orig.lower()).ratio() * 100
        print "%.2f" % accuracy + "%"


    except Exception,e:
        #print 'Cant find input. Have you loaded an image?'
        print "Caught an Exception with details: %s" % e
    except Exception,e:
        #print 'Cant find input. Have you loaded an image?'
        print "Caught an Exception with details: %s" % e

print '\nFinished identifying'
