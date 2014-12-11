import os,sys,pickle
import logging
#import numpypy as np
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import ndimage as nd

from skimage.filter import sobel
from skimage.filter import rank
from skimage.segmentation import * 
from skimage.morphology import watershed, disk
from skimage.color import label2rgb
from skimage import data, img_as_float, graph, data, io, segmentation, color, exposure
from skimage.feature import hog 
import skimage.io
from skimage.morphology import watershed, disk
from skimage import data
from skimage.filter import rank
from skimage.util import img_as_ubyte

import cv2
import sklearn.cluster

from segmentation import IMAGE_FEATURES

################################################################################

def display_labels(labelslist,image,s2wdict):
    num_images = len(image)
    fig, axes = plt.subplots(num_images, figsize=(9, 2.5))

    for i in range(num_images) :
        #replace segment labels by their words.
        sd = s2wdict[i] 
        labels = labelslist[i]
        for label in sd.keys():
            word = sd[label]
            labels[labels==word] = word 
            
        segimg = label2rgb(pow(labels,1), image=image[i], image_alpha=0.1)
        axes[i].imshow(segimg, interpolation='nearest')

    legend = plt.legend()
    for ax in axes:
        ax.axis('off')
    
    plt.show()


################################################################################

def do_predictions(km,s2,one_image):
    nf = 3 
    segment_to_word=dict()
    for k in one_image.keys():
        features = (one_image[k]).reshape([1,nf])
        word = km.predict(features)
        segment_to_word[k] = word 

    print segment_to_word
    return segment_to_word 

def process_one_image(fname,km):
    print fname 
    (im,s2,one_image) = IMAGE_FEATURES(fname) 
    #dictionary
    s2wdict = do_predictions(km,s2, one_image)
    return (s2,im,s2wdict)

    
def main(fnames):
    test_phase = False
    pf = open('Model.pickle')
    km = pickle.load(pf) #Kmeans model
    seglist=[]
    imlist=[]
    sdictlist=[]
    allfiles=os.listdir('../data/testimgs/')
    print allfiles
    randfiles = ['test'+str(x)+'.png' for x in range(106,112)]
    for fname in randfiles :
        #fname = fnames[1] 
        fname = os.path.join('../data/testimgs/', fname)
        (s2,im,s2wdict) = process_one_image(fname,km)
        seglist.append(s2) ; imlist.append(im) ; sdictlist.append(s2wdict)
    
    display_labels(seglist,imlist,sdictlist)
    
    
################################################################################

if __name__ == "__main__":
    main(sys.argv)
