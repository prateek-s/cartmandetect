import os,sys,pickle
import logging
#import numpypy as np
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import ndimage as nd

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
import sklearn.metrics

from segmentation import IMAGE_FEATURES

################################################################################

ALL_FEATURES = []
ALL_LABELS = []


################################################################################

def display_labels(labelslist,image,s2wdict):
    num_images = len(image)
    fig, axes = plt.subplots(num_images, figsize=(9, 2.5))

    for i in range(num_images) :
        #replace segment labels by their words.
        labels = labelslist[i]
        for alabel in np.unique(labels) :
            (x,y) = get_segment_centre(labels,alabel)

            axes[i].text(x,y,str(alabel))
            
        segimg = label2rgb(labels, image=image[i], image_alpha=0.5)
        axes[i].imshow(segimg)

    legend = plt.legend()
    for ax in axes:
        ax.axis('off')
    
    plt.show()



################################################################################


def get_segment_centre(segmentedim, theseg):
    (mx,by) = np.shape(segmentedim)
    
    (Ys,Xs) = np.where(segmentedim == theseg)

    left=np.min(Xs)
    right=np.max(Xs)
    top=np.min(Ys)
    bottom=np.max(Ys)
    cX = (left+right)/2;
    cY = (top+bottom)/2;

    return (cX,cY)

################################################################################



def do_predictions(km,s2,features):
    global ALL_FEATURES
    global ALL_LABELS
    nf = 3 
    segment_to_word=dict()
    all_image_features = np.empty((len(features.keys()),nf))
    i = 0 
    wordlist = []
    for k in features.keys():
        feature_vec = (features[k]).reshape([1,nf])
        all_image_features[i] = feature_vec 
        i = i + 1
        #feature vec is one [r,g,b] value. HSV here?
        word = km.predict(feature_vec)
        
        #print "Predicted :"+str(word) +" for " + str(feature_vec) 
        segment_to_word[k] = word 
        wordlist.append(word)
        s2[s2==k] = word 
        
    #print segment_to_word
    #print all_image_features.shape
    #print np.array(wordlist)
    ALL_FEATURES = np.append(ALL_FEATURES, all_image_features)
    ALL_LABELS = np.append(ALL_LABELS, np.array(wordlist).flatten)
    #silscore = sklearn.metrics.silhouette_samples(all_image_features, np.array(wordlist).flatten())
    #print "SILSCORE " + str(silscore) 
    return (segment_to_word,s2) 

    
def process_one_image(fname,km):
    global DET_SEG_HIST
    #print fname 
    (im,s2,features) = IMAGE_FEATURES(fname) 
    #dictionary
    (s2wdict,s2) = do_predictions(km,s2, features)
    return (s2,im,s2wdict)

    
def main(fnames):
    global ALL_FEATURES
    global ALL_LABELS
    test_phase = False
    pf = open('Model.pickle')
    km = pickle.load(pf) #Kmeans model

    seglist=[]
    imlist=[]
    sdictlist=[]
    allfiles=os.listdir('../data/testimgs/')
    
    randfiles = ['test'+str(x)+'.png' for x in range(106,108)]
    for fname in randfiles :
        #fname = fnames[1] 
        fname = os.path.join('../data/testimgs/', fname)
        (s2,im,s2wdict) = process_one_image(fname,km)
        
        seglist.append(s2) ; imlist.append(im) ; sdictlist.append(s2wdict)
    
    display_labels(seglist,imlist,sdictlist)
    print ALL_FEATURES.shape
    print ALL_LABELS.shape
    #silscore = sklearn.metrics.silhouette_samples(ALL_FEATURES, ALL_LABELS)
    print "SILSCORE " + str(silscore) 

    
    
################################################################################

if __name__ == "__main__":
    main(sys.argv)
