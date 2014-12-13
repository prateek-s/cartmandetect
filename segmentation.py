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
from skimage.util import img_as_ubyte

import cv2

################################################################################

def do_grabcut(f):
    img = cv2.imread(f)
    mask = np.zeros(img.shape[:2],np.uint8)
    (h,w,d) = img.shape
    #print (h,w,d)
    upper_corner=(h/10,w/10)
    lower_corner=(h-(h/10),w-(w/10))
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    #rect = (upper_corner[0],upper_corner[1],lower_corner[0],lower_corner[1])
    rect= (0,0,w-10,h-10)
    #print rect
    iterCount=1
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,iterCount,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
    output = cv2.bitwise_and(img,img,mask=mask2)

    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    # img = img*mask2[:,:,np.newaxis]

    #plt.imshow(img),plt.show()
    cv2.imwrite('gcut.jpg',output)
    return output
    
################################################################################

def quickshift_segmentation(im):
    return quickshift(im)

################################################################################

def watershed_segmentation(im):
    gray=skimage.color.rgb2gray(im)
    denoised = rank.median(gray, disk(2))
    # find continuous region (low gradient) --> markers
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = nd.label(markers)[0]
    seg = watershed(gray,markers)
    return seg

################################################################################

def normcut_segmentations(img):

    #labels1 = segmentation.slic(img, compactness=3, n_segments=50)
    labels1 = segmentation.slic(img,compactness=3,n_segments=20)
    out1 = color.label2rgb(labels1, img)#, kind='avg')
    #return labels1
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    out2 = color.label2rgb(labels2, img,image_alpha=0.2)#, kind='avg')
    return (labels1,labels2)

    
################################################################################
    

def show_all(fname,images,titles,numsegs=1):
    
    num_images = len(images)
    num_titles = len(titles)
    titles += ['']*(num_images-num_titles)
    
    fig, axes = plt.subplots(ncols=num_images, figsize=(9, 2.5))

    im = images[0]

    for i in range(numsegs):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i])
        print titles[i]

    for i in range(numsegs,num_images) :
        j=i #numsegs+i
        segimg = label2rgb(images[j], image=im, image_alpha=0.5)
        axes[j].imshow(segimg, interpolation='nearest')
        axes[j].set_title(titles[j])
        print titles[j]

    for ax in axes:
        ax.axis('off')
    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
    plt.show()
    #plt.savefig(fname+"_seg.jpg")
    print fname+"_seg.jpg"
    return True

################################################################################

def get_feature_vectors(im, thehog, labels):

    (h,w,d) = im.shape
    fimage=im.reshape([h*w,d]) #flattened image
    fhog = thehog.flatten() 

    color_vector=dict()
    hog_vector=dict()
    
    segments = np.unique(labels)
    #Create the dictionary
    for seg in segments :
        color_vector[seg]=[]
        hog_vector[seg]=[]

    for (i,lab) in enumerate(labels.flatten()) :
        color_vector[lab].append(fimage[i])
        #hog_vector[lab].append(fhog(i))
        
    #Now convert to numpy array for faster processing
    color_vector = convert_dict_to_numpy(color_vector)
    #hog_vector = convert_dict_to_numpy(hog_vector)

    c_hist = get_color_histogram(color_vector)
    #get_hog_histogram(hog_vector)
    #print c_hist
    return c_hist
    
################################################################################
    
def get_color_histogram(d):
    for k in d.keys():
        vec = d[k]
        r = np.uint8(np.average(vec[...,0]))
        g = np.uint8(np.average(vec[...,1]))
        b = np.uint8(np.average(vec[...,2]))
        d[k] = np.array((r,g,b))
    return d 

################################################################################
    
def get_hog_histogram(d):
    
    pass

################################################################################

def convert_dict_to_numpy(d):
    for k in d.keys():
        arr = np.array(d[k])        
        d[k] = arr
    return d

################################################################################

def do_hog(im):
    img = color.rgb2gray(im)

    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    hog_image_rescaled = hog_image # exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    return hog_image_rescaled 


################################################################################

def save_features(fname, features) :
    basen = os.path.basename(fname) 
    basen = os.path.splitext(basen)[0]
    basen = str(basen) + '.pickle' 
    
    fp = os.path.join('../features/',basen)
    with open(fp, 'wb') as outfile:
        pickle.dump(features, outfile, protocol=pickle.HIGHEST_PROTOCOL)

################################################################################

def get_segments_hog(fname):
    do_grabcut(fname)
    im=skimage.io.imread(fname)
    gi=skimage.io.imread('gcut.jpg')    
    (s1,s2)=normcut_segmentations(gi)
    hogi = do_hog(gi)
    #s3=watershed_segmentation(gi)
    #s4=quickshift_segmentation(gi)

    return (im,s2,hogi) 
    #show_all(fname, [im,gi,hogi,s2],["Image","Grab","hog","S2"],3)

################################################################################

def IMAGE_FEATURES(fname) :
    """ Features is a dictionary, key==segment label. Value==feature vector.
    Only color features are implemented at the moment"""
    (im,s2,hogi) = get_segments_hog(fname)    
    features = get_feature_vectors(im,hogi, s2)
    return (im,s2,features)

################################################################################
    
def main(fnames):
    save_feats = True
    
    for f in os.listdir(fnames[1]):
        fname = os.path.join(fnames[1],f)
        if not os.path.isfile(fname) :
            print "Error, not exists"+str(fname)
            return False
        print '---------------------'
        print fname
        (im,s2,features) = IMAGE_FEATURES(fname)
        if save_feats:
            save_features(fname, features)

    #grabcutI = do_grabcut(fname)
        #grabcutI = opencv_convert(grabcutI)
        #segmentedI = do_segmentation(fname)
        #normcut_segmentations(skimage.io.imread(fname))
    #pickle the segmented image?
    #display?

################################################################################

if __name__ == "__main__":
    main(sys.argv)
