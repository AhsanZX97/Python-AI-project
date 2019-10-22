import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
#from skimage.morphology import label
from skimage.measure import regionprops
from skimage.measure import label
import os, os.path
from sklearn.datasets import load_digits
from PIL import Image


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

        newx = 0

        for region in regionprops(label_image):
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
                    if lasty != bl_2 and i > 0:
                        final.append('\n')
                    if lastx !=0 and tl_2 - lastx > 40:
                        final.append(' ')
                    tl,tr,bl,br = lines[i][j]
                    letter_raw = bw[tl:bl,tr:br]
                    letter_norm = imresize(letter_raw ,(20 ,20))
                    final.append(letter_norm)
                    prev_tr = lines[i][j][3]
                if j == (len(lines[i])-1):
                    prev_line_br = lines[i][j][2]
                if lasty != bl_2 :
                    lastx = lines[i][j][1]
                else :
                    lastx = 0
                lasty = bl_2
            prev_tr = 0
            tl_2 = 0
        print 'Characters recognized: ' + str(len(final))
        print final
        return final


    def __init__(self):
        print "Extracting characters..."

extract = Extract_Letters()


file = './fuckoffpython.png'
letters = extract.extractFile(file)
name_counter = 600
string_counter = 0
num = 0
#for i in letters:
#	num+=1
#print num

root = "./training_type/"
cpt = sum([len(files) for r, d, files in os.walk(root)])
print cpt
target = []
image = []
data = []
for path, subdirs,files in sorted(os.walk(root)):
	for name in files:
#for name in os.walk(dirnames):
		image.append(os.path.join(path, name))
		im = Image.open(os.path.join(path, name))
		data.append(list(im.getdata()))

folder_string = 'abcdefghijklmnopqrstuvwxyz'

for x in range(1,10):
	target += [x] * 45

for letter in folder_string:
	target += [letter] * 90
