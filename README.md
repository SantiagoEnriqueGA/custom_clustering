# Custom Clustering Project

## Goal
This project is primarily educational. It is designed to help understand the workings of clustering algorithms by building them from scratch. The implementations focus on fundamental concepts rather than on optimizing for speed or robustness, using only numpy for array processing and custom datasets for specific tasks.

---

### KMeans.py

This is a custom implementation of the KMeans clustering algorithm along with additional functionalities for evaluating the optimal number of clusters and visualizing the clustering results.

**Classes:**
- KMeans: A class implementing KMeans clustering with methods for fitting the model, predicting cluster labels, and finding the optimal number of clusters.

**Methods:**
- `__init__`: Initializes the KMeans object with parameters such as the data matrix, number of clusters, maximum iterations, and convergence tolerance.
- `fit`: Fits the KMeans model to the data by iteratively updating centroids and cluster assignments until convergence.
- `predict`: Predicts the cluster labels for new data points based on the fitted centroids.
- `find_optimal_clusters`: Implements methods to find the optimal number of clusters using the elbow method, Calinski-Harabasz Index, Davies-Bouldin Index, and Silhouette Score. It also plots the evaluation metrics to aid in determining the optimal k value.
- `calinski_harabasz_index`: Calculates the Calinski-Harabasz Index for evaluating clustering performance.
- `davies_bouldin_index`: Calculates the Davies-Bouldin Index for evaluating clustering performance.
- `silhouette_score`: Calculates the Silhouette Score for evaluating clustering performance.
- `elbow_method`: Implements the elbow method to determine the optimal number of clusters.
- `initialize_centroids`: Randomly initializes the centroids for KMeans clustering.
- `assign_clusters`: Assigns clusters based on the nearest centroid.
- `update_centroids`: Updates centroids based on the current cluster assignments.
- `_handle_categorical`: Handles categorical columns in the input data by one-hot encoding.
- `_convert_to_ndarray`: Converts input data to a NumPy ndarray and handles categorical columns.

#### Additional Information on Evaluation Metrics:

1. **Silhouette Score:**
   - Measures how similar an object is to its own cluster compared to other clusters (ranges from -1 to 1).
   - Formula: $`\text{silhouette\_sample} = \frac{b - a}{\max(a, b)}`$ where $`a`$ is within-cluster distance, $`b`$ is the nearest-cluster distance.

2. **Davies-Bouldin Index:**
   - Measures the average similarity between each cluster and its most similar cluster (lower is better).
   - Formula: $`DB = \frac{1}{n} \sum_{i=1}^{n} \max_{j \neq i} \left( \frac{\text{avg\_distance}_i + \text{avg\_distance}_j}{\text{distance}_{ij}} \right)`$

3. **Calinski-Harabasz Index:**
   - Measures how well separated the clusters are from each other (higher is better).
   - Formula: $`CH = \frac{\text{Bk}}{\text{Wk}} \times \frac{N - k}{k - 1}`$ where $`N`$ is the total number of data points, $`k`$ is the number of clusters.

These metrics are used to evaluate clustering performance and determine the optimal number of clusters for a given dataset.

**Usage Example:**
```python
# Example usage:
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

true_k = 8
# Generate synthetic data for testing
X, y = make_blobs(n_samples=1000, n_features=2, centers=true_k, cluster_std=0.60, random_state=1)

# Initialize KMeans object
kmeans = KMeans()

# Find the optimal number of clusters using the elbow method
ch_optimal_k, db_optimal_k, silhouette_optimal_k = kmeans.find_optimal_clusters(X, max_k=10)

print(f'calinski_harabasz_index number of clusters: {ch_optimal_k}')
print(f'davies_bouldin_index number of clusters: {db_optimal_k}')
print(f'silhouette_score number of clusters: {silhouette_optimal_k}')

# Create a 2x2 subplot for each k value
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Define titles based on which k value method was used
titles = [f'True "make_blobs" K = {true_k}', 
          f'Calinski-Harabasz Index Optimal K = {ch_optimal_k}', 
          f'Davies-Bouldin Index Optimal K = {db_optimal_k}', 
          f'Silhouette Optimal K = {silhouette_optimal_k}']

# Loop through each k value and plot the results of fit()
for i, k in enumerate([true_k, ch_optimal_k, db_optimal_k, silhouette_optimal_k]):
    # Fit the KMeans model with the current k value
    kmeans.n_clusters = k
    kmeans.fit(X)

    # Predict cluster labels
    labels = kmeans.predict(X)

    # Plot the results in the corresponding subplot
    row = i // 2
    col = i % 2
    axs[row, col].scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    centroids = kmeans.centroids
    axs[row, col].scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.75)
    axs[row, col].set_title(titles[i])
    axs[row, col].set_xlabel('Feature 1')
    axs[row, col].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

---

### MallCustomerSegmentation.ipynb

This notebook performs customer segmentation analysis using the Mall Customers dataset. It includes exploratory data analysis, visualization, and clustering using the custom KMeans implementation.

---

### KMeans_exampleUsage.ipynb

This notebook demonstrates an example usage of the custom KMeans clustering algorithm on synthetic data generated using `make_blobs` from `sklearn.datasets`. It showcases how to find the optimal number of clusters and visualize the clustering results.

---

Please make sure to have the necessary libraries installed (e.g., NumPy, pandas, matplotlib, seaborn) to run the code successfully.

