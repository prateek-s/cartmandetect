import os,sys,pickle
import numpy as np
import sklearn.cluster
import sklearn.lda
import matplotlib.pyplot as plt
import sklearn.metrics

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
        


def do_kmeans_clustering(ALLF) :
    km = sklearn.cluster.KMeans(n_clusters=100,n_init=20)
    labels = km.fit_predict(ALLF)
    CC = km.cluster_centers_
    with open('Model.pickle','wb') as outfile:
        pickle.dump(km, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    print CC
    print CC.shape
    return km,CC,labels



def dbscan_clust(ALLF) :
    dbs = sklearn.cluster.DBSCAN()
    dict_assignments = dbs.fit(ALLF)
    dict_assignments = dbs.fit_predict(ALLF)
    dict_assignments = dbs.labels_ 
    with open('Model_dbs.pickle','wb') as outfile:
        pickle.dump(dbs, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return dbs,[],dict_assignments


def agglo_clust(ALLF) :
    dbs = sklearn.cluster.AgglomerativeClustering(n_clusters=50)
    dict_assignments = dbs.fit(ALLF)
    dict_assignments = dbs.fit_predict(ALLF)
    with open('Model_dbs.pickle','wb') as outfile:
        pickle.dump(dbs, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return dbs,[],dict_assignments

def lda_clust(ALLF) :
    km = sklearn.lda.LDA()
    km.fit(ALLF)
    #CC = km.cluster_centers_
    with open('Model.pickle','wb') as outfile:
        pickle.dump(km, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        
    print CC


def silhoutte_score(ALLF,labels):
    scores =  sklearn.metrics.silhouette_samples(ALLF, labels)
    print scores
    print np.average(scores)
    print np.std(scores)
    print ALLF.shape
    print labels.shape
    return scores 

    
def main(fpath) :
    print fpath[1]
    ALLF = read_all_features(fpath[1])
    km,CC,labels = do_kmeans_clustering(ALLF)
    #km,CC,labels = dbscan_clust(ALLF)
    #km,CC,labels = agglo_clust(ALLF)
    silhoutte_score(ALLF,labels)
    visualize_dictionary(labels,km,CC)
    #lda_clust(ALLF)


def visualize_dictionary(dict_assignments,km,CC) :
    #each element is a cluster centre (r g b 3d)
    #dict_assignments are cluster centres!
    histogram_dict=dict() 
    allc = np.unique(dict_assignments)
    for c in allc:
        if c!=-1:
                histogram_dict[c]=0 ;
    for w in dict_assignments:
        if w != -1 :
                histogram_dict[w]=histogram_dict[w]+1 

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111)
    #colors are just the CC normalized by 255
    
    colors = CC/255.0 
    ys=[]
    for c in allc:
        if c!=-1:
                ys.append(histogram_dict[c])
    ys = np.array(ys) 
    xs = []
    for c in allc:
        if c!=-1 :
            xs.append(c)
    xs = np.array(xs)
    normalize = np.float(np.sum(ys))
    #print "MAX"+str(normalize) 
    ys = np.divide(ys,normalize)
    #print np.column_stack((xs,ys))

    #Need to sort bars by y's. Colors are important here
    #sort (xs,ys) on ys. we'll get color order. remake color array
    sort_hist = False
    if sort_hist :
        xs = list(xs)
        ys = list(ys)
        
        sortedys = zip(xs,ys)
        
        sortedys = sorted(sortedys , key=lambda x: x[1],reverse=True)
        #print sortedys
        
        colors = []
        for (x,y) in sortedys :
            color = CC[x]/255.0 
            colors.append(color)
            
        colors = np.array(colors) 
        (xs,ys) = zip(*sortedys) 
        xs = list(xs) ; ys = list(ys)
        xs = sorted(xs)
    
    rects = ax.bar(xs, ys , color=colors)
    ax.set_xlabel("Cluster number")
    ax.set_ylabel("Frequency")
    plt.show()
        
        
    
    
if __name__ == "__main__":
    main(sys.argv)
