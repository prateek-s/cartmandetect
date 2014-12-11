import os,sys,pickle
import numpy as np
import sklearn.cluster
import sklearn.lda

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
    km = sklearn.cluster.KMeans(k=100)
    km.fit_predict(ALLF)
    CC = km.cluster_centers_
    with open('Model.pickle','wb') as outfile:
        pickle.dump(km, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        
    print CC

def lda_clust(ALLF) :
    km = sklearn.lda.LDA()
    km.fit(ALLF)
    #CC = km.cluster_centers_
    with open('Model.pickle','wb') as outfile:
        pickle.dump(km, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        
    print CC
    
    
def main(fpath) :
    print fpath[1]
    ALLF = read_all_features(fpath[1])
    #do_clustering(ALLF)
    lda_clust(ALLF)
    
if __name__ == "__main__":
    main(sys.argv)
