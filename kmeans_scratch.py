import numpy as np
import matplotlib.pyplot as plt


class KMeansCLustering:
    def __init__(self, k, epsilon=1e-4, max_iters=200):
        self.k = k  #jumlah cluster
        self.epsilon = epsilon #threshold konvergensi
        self.max_iters = max_iters  #maksimum iterasi update centroidnya
        self.centroids = []  #list centroid awal
        self.clusters = []  #list cluster awal
    
    #Jarak Data Point ke Centroid dengan Euclidian Distance
    def euclidian_distance(self, X):
        return np.sqrt(np.sum((X - self.centroids)**2, axis=1))
    
    #Jarak Data Point ke Centroid dengan Manhattan Distance
    def manhattan_distance(self, X):
        return np.sum(np.abs((X - self.centroids), axis=1))
    
    #assign tiap data ke cluster terdekatnya
    def cluster_points(self, data):
        self.clusters = [[] for _ in range(self.k)]
        for data_point in data:
            distances = self.euclidian_distance(data_point)
            closest_centroid = np.argmin(distances)
            self.clusters[closest_centroid].append(data_point)
    
    #perbarui centroid berdasarkan rata-rata cluster        
    def update_centroids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clusters[i], axis=0)
    
    #cek apakah sudah konvergen atau belum centroidnya        
    def is_converged(self, old_centroids):
        diff = np.sum(np.abs(self.centroids - old_centroids))
        return diff < self.epsilon
    
    def fit(self, data):
        """
        - inisiasi centroid secara acak berdasarkan data yang ada
        - assign tiap data ke cluster terdekatnya
        - perbarui centroid berdasarkan rata-rata cluster
        - ulangi proses assign dan update hingga konvergen atau telah mencapai maks iterasinya
        """
        random_idx = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = np.array(data[random_idx])
        for i in range(self.max_iters):
            #kita simpan dulu centroid lama untuk nnanti cek konvergensi
            old_centroids = self.centroids.copy()
            #assign 
            self.cluster_points(data)
            #update
            self.update_centroids()
            #cek konvergensi
            if self.is_converged(old_centroids):
                print(f"Converged after {i+1} iterations")
                break
            
        return self.centroids, self.clusters


            
        
            
    
            
        
        
        
        
    