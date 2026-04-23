#========================================================#
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
#========================================================#



#-----------------------------------------------------------#
'''setting random k point to as centroids by random function'''

def intialize_centroids(X,K):
    indices = np.random.choice(X.shape[0],K,replace = False)
    return X[indices]

#-----------------------------------------------------------#
#-----------------------------------------------------------#
'''this will return single vector of nearest to vector point 
to the centroid if first data point is near first centroid it will 
have 0 same 1 & 2 for other returns vector of labels'''
def assign_clusters(X,centroids):
    distances = []
    for i in centroids:
        clus=np.linalg.norm(X-i,axis=1)
        distances.append(clus)
    return np.argmin(distances,axis=0)
#-----------------------------------------------------------#


#-----------------------------------------------------------#
'''Update centroids as mean of vector to that point
if vector not have any point then it will take yet another 
random point as centroid'''

def update_centroids(X,K,labels):
    new_centroids = []
    for i in range(K):
        clus=X[labels==i]

        if len(clus) == 0:
            indx=np.random.randint(0,X.shape[0])
            new_centroids.append(X[indx])
        else:
            new_centroids.append(np.mean(clus,axis=0))

    return np.array(new_centroids)
#------------------------------------------------------------#

#------------------------------------------------------------#
'''Here inertia is nothing but sum of distances squared
from the centroid & total for all centroids'''

def c_inertia(X,centroids,labels):
    inertia =0
    for i,c in enumerate(centroids):
        points = X[labels == i]
        inertia += np.sum((points-c)**2)
    return inertia
#---------------------------------------------------------------#

#---------------------------------------------------------------#
'''Main function accounting for all the iterations
selection of clusterin which has lesser inertia'''

def kmeans(X,K,max_iter,ninr):
    best_centroids = None
    best_labels = None
    best_inertia = float('inf')
    for iter in range(ninr):
        centroids = intialize_centroids(X,K)
        for i in range(max_iter):
            labels = assign_clusters(X,centroids)
            new_centroids = update_centroids(X,K,labels)

            if np.linalg.norm(new_centroids - centroids) < 1e-5:
                print(f"Converged after {i+1} iterations.")
                break
            centroids = new_centroids

        curent_inertia = c_inertia(X,new_centroids,labels)
        if curent_inertia < best_inertia:
            best_centroids = new_centroids
            best_labels = labels
            best_inertia = curent_inertia

    return best_centroids, best_labels
#----------------------------------------------------------------_#

def main():
    # X, y = make_blobs(n_samples=5000,centers=10,random_state=42)
    X = np.array([
        [1, 4],
        [1, 3],
        [0, 4],
        [5, 1],
        [6, 2],
        [4, 0]
    ])
    centroids,clust_label = kmeans(X,2,100,10)


    plt.scatter(X[:, 0],X[:, 1],c=clust_label,cmap='viridis',s=10)
    plt.scatter(centroids[:, 0],centroids[:, 1],c='red',s=50)
    plt.show()

if __name__ == '__main__':
    main()