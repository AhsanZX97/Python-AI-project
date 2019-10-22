import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage.measure import label
import os, os.path
from PIL import Image
from Tkinter import Tk
from tkFileDialog import askopenfilename
import difflib


'''
OCR Engine for the Artificial Intelligence Class
================================================
'''


class OCR_Demo:
    #init of our simple class - self.VARIABLE means that the VARIABLE object is an attribute of the class
    #and we can use it later on.
    def __init__(self):

        #split the dataset into a training and a testing part
        root = "./training writing/"
        cpt = sum([len(files) for r, d, files in os.walk(root)])
        target = []
        image = []
        data = []
        for path, subdirs, files in sorted(os.walk(root)):
            for name in files:
                image.append(os.path.join(path, name))
                im = Image.open(os.path.join(path, name))
                data.append(list(im.getdata()))

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

        print len(target)

        self.training_set = { 'images':image,'data':data,'target':target }

        #train the k-nn classifier
        self.knn_classifier = self.train_knn_classifier()

        #show the program
        self.show_gui()

    #gui stuff - create imshow axes + buttons
    def show_gui(self):
        #Gui related stuff - buttons etc.
        main_ocr_figure = plt.figure(num=None, figsize=(5, 10), dpi=80, facecolor='w', edgecolor='k')
        self.digit_axes = main_ocr_figure.add_subplot(211)
        self.digit_thumbnail_axes = main_ocr_figure.add_subplot(10,5,28)
        self.digit_thumbnail_axes.get_xaxis().set_visible(False)
        self.digit_thumbnail_axes.get_yaxis().set_visible(False)
        self.digit_axes.get_xaxis().set_visible(False)
        self.digit_axes.get_yaxis().set_visible(False)
        self.digit_thumbnail_axes.get_xaxis().set_visible(False)
        self.digit_thumbnail_axes.get_yaxis().set_visible(False)
        self.digit_axes.set_title('Unkown image')
        self.digit_thumbnail_axes.set_title('Unkown image thumbail')


        ax_load_digit = plt.axes([0.25, 0.32, 0.55, 0.075])
        load_digit_button = Button(ax_load_digit, 'Test Button Ignore')
        load_digit_button.on_clicked(self.get_random_image)

        ax_knn_result = plt.axes([0.25, 0.12, 0.55, 0.075])
        get_knn_result_button = Button(ax_knn_result, 'Load image')
        get_knn_result_button.on_clicked(self.get_knn_results)
        plt.show()

    def get_random_image(self,event):
            number_of_samples = 322
            rand_image_pick = randrange(number_of_samples)
            ranpick = self.training_set['images'][rand_image_pick]
            im = Image.open(ranpick)
            print list(im.getdata())
            self.unknown_digit_image = imread(ranpick)
            self.digit_axes.imshow(self.unknown_digit_image,cmap = cm.Greys_r)
            self.digit_thumbnail_axes.imshow(self.unknown_digit_image,cmap = cm.Greys_r)
            plt.draw()

    #gets results using a knn classifier that we trained during the __init__ phase
    def get_knn_results(self,event):
        try:
            extract = Extract_Letters()
            folder_string = '123456789abcdefghijklmnopqrstuvwxyz'
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
                elif i == '\n':
                    char+= '\n'
                else :
                    feature_vector =  i.ravel()
                    knn_results = self.knn_classifier.predict_proba([feature_vector])[0]
                    found = 0
                    letter =0
                    for j,result in enumerate(knn_results):
                        if result > found:
                            found = result
                            letter = j
                    char += folder_string[letter]
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

    #training of the knn classifier.
    def train_knn_classifier(self):
        k=64
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(self.training_set['data'], self.training_set['target'])
        print 'Trained the knn-classifier'
        return knn_classifier


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


    def __init__(self):
        print "Extracting characters..."



#main function here.
#just calls the OCR_Demo class.
#See the init method of the class.
def main():
    print __doc__
    OCR_Demo()

if __name__ == '__main__':
    main()
