import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from KMeans import KMeans


df = pd.read_csv(r'Data/Mall_Customers.csv')


# Initialize KMeans object
kmeans = KMeans(df)

ch_optimal_k, db_optimal_k, silhouette_optimal_k = kmeans.find_optimal_clusters(max_k=10)

# Fit the KMeans model with the optimal number of clusters
kmeans.n_clusters = 6
kmeans.fit()

# Predict cluster labels
labels = kmeans.predict(kmeans.X)

# Plot the results
plt.scatter(kmeans.X[:, 0], kmeans.X[:, 1], c=labels, s=50, cmap='viridis')
centroids = kmeans.centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.75)
plt.show()