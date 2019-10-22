# Sample code for AI Coursework
# Extracts the letters from a printed page.

# Keep in mind that after you extract each letter, you have to normalise the size.
# You can do that by using scipy.imresize. It is a good idea to train your classifiers
# using a constant size (for example 20x20 pixels)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize
from skimage.segmentation import clear_border
#from skimage.morphology import label
from skimage.measure import label
from skimage.measure import regionprops
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


image = imread('./Task2/writing2.png',1)

#apply threshold in order to make the image binary
bw = image < 120

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)

# label image regions
label_image = label(cleared,neighbors=8)
borders = np.logical_xor(bw, cleared)
label_image[borders] = -1

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(bw, cmap='jet')


counter =0
prevminc = 0
prevminr = 0
width = 250
for region in regionprops(label_image):
    # skip small images
    if region.area > 50:

        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        counter+=1
print counter

plt.show()
