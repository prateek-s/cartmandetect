import os,sys,pickle
import numpy as np
import sklearn.cluster
import sklearn.lda
import matplotlib.pyplot as plt


def read_all_features(path):
    nf=3
    ALLF=np.empty([1,nf]) #RGB!
    is_first_entry = False

    for fn in os.listdir(path):
        #print f 
        pf = open(os.path.join(path,fn)) 
        one_image = pickle.load(pf)
        for k in one_image.keys():
            #key=segment label, can be ignored
            features = one_image[k] #numpy array
            features = features.reshape([1,nf])
            print "feat: "+str(features)
            if is_first_entry :
                ALLF[0] = features 
                is_first_entry = False
            else:
                ALLF = np.append(ALLF, features,axis=0)

    print ALLF.shape
    print ALLF
    return ALLF
        


def do_clustering(ALLF) :
    km = sklearn.cluster.KMeans(k=50)
    dict_assignments = km.fit_predict(ALLF)
    CC = km.cluster_centers_
    with open('Model.pickle','wb') as outfile:
        pickle.dump(km, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    print CC
    return km,CC,dict_assignments

def lda_clust(ALLF) :
    km = sklearn.lda.LDA()
    km.fit(ALLF)
    #CC = km.cluster_centers_
    with open('Model.pickle','wb') as outfile:
        pickle.dump(km, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        
    print CC
    


def visualize_dictionary(dict_assignments,km,CC) :
    #each element is a cluster centre (r g b 3d)
    #dict_assignments are cluster centres!
    histogram_dict=dict() 
    allc = np.unique(dict_assignments)
    for c in allc:
        histogram_dict[c]=0 ;
    for w in dict_assignments:
        histogram_dict[w]=histogram_dict[w]+1 

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    #colors are just the CC normalized by 255
    colors = CC/255.0 
    ys=[]
    for c in allc:
        ys.append(histogram_dict[c])
    ys = np.array(ys) 
    rects = ax.bar(allc, ys, color=colors)
    ax.set_xlabel("Cluster number")
    ax.set_ylabel("Frequency")
    plt.show()
        
    
def main(fpath) :
    print fpath[1]
    ALLF = read_all_features(fpath[1])
    km,CC,dict_assignments = do_clustering(ALLF)
    visualize_dictionary(dict_assignments,km,CC)
    #lda_clust(ALLF)
    
if __name__ == "__main__":
    main(sys.argv)
