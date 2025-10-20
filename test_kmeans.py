import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans_scratch import KMeansCLustering
from sklearn import datasets

def test_kmeans():
    #load make blobs dataset
    X, y = datasets.make_blobs(n_samples=300, centers=4, n_features=2, random_state=42) #membuat dataset dengan 4 cluster dan 2 fitur)
    k_values = [2, 3, 4, 5]
    
    # membuat 4 subplot untuk visualisasi hasil clustering
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel() 
    
    for idx, k in enumerate(k_values):
        print(f"\nTesting K = {k}")
        
        #inisiasi model KMeans
        kmeans = KMeansCLustering(k=k)
        #fit model
        kmeans.fit(X)
        
        print(f"Cluster size: {[len(c) for c in kmeans.clusters]}")
        
        #plot
        for i in range(k):
            cluster_points = np.array(kmeans.clusters[i])
            axes[idx].scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"CLuster {i+1}")
            
        centroids = np.array(kmeans.centroids)
        axes[idx].scatter(centroids[:, 0], centroids[:, 1], s=200, c="black", marker="X", label="Centroids")
        axes[idx].set_title(f"K = {k}")
        axes[idx].set_xlabel("Feature 1")
        axes[idx].set_ylabel("Feature 2")
        axes[idx].legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test_kmeans()

            
        
        
        
    
    








