import timeit
import numpy as np
from clustering import DBSCAN, KMeans
from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.cluster import KMeans as skKMeans

def timing(n=1000):
    X = np.random.rand(n, 2)
    dbscan = DBSCAN(X)
    kmeans = KMeans(X)
    
    dbscan_time = timeit.timeit(lambda: dbscan.fit(), number=10)
    kmeans_time = timeit.timeit(lambda: kmeans.fit(), number=10)
    
    print(f"Custom DBSCAN fitting average time: {dbscan_time / 10:.6f} seconds")
    print(f"Custom KMeans fitting average time: {kmeans_time / 10:.6f} seconds")
    print("--------------------------------------------------------------")
    
    skl_dbscan = skDBSCAN()
    skl_kmeans = skKMeans()
    
    skl_dbscan_time = timeit.timeit(lambda: skl_dbscan.fit(X), number=10)
    skl_kmeans_time = timeit.timeit(lambda: skl_kmeans.fit(X), number=10)
    
    print(f"sklearn DBSCAN fitting average time: {skl_dbscan_time / 10:.6f} seconds")
    print(f"sklearn KMeans fitting average time: {skl_kmeans_time / 10:.6f} seconds")
    
if __name__ == "__main__":
    timing()
