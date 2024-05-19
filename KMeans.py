"""
This is a custom implementation of the KMeans clustering algorithm along with additional functionalities for evaluating the optimal number of clusters and visualizing the clustering results.

Classes:
- KMeans: A class implementing KMeans clustering with methods for fitting the model, predicting cluster labels, and finding the optimal number of clusters.

Methods:
- __init__: Initializes the KMeans object with parameters such as the data matrix, number of clusters, maximum iterations, and convergence tolerance.
- fit: Fits the KMeans model to the data by iteratively updating centroids and cluster assignments until convergence.
- predict: Predicts the cluster labels for new data points based on the fitted centroids.
- find_optimal_clusters: Implements methods to find the optimal number of clusters using the elbow method, Calinski-Harabasz Index, Davies-Bouldin Index, and Silhouette Score. It also plots the evaluation metrics to aid in determining the optimal k value.
- calinski_harabasz_index: Calculates the Calinski-Harabasz Index for evaluating clustering performance.
- davies_bouldin_index: Calculates the Davies-Bouldin Index for evaluating clustering performance.
- silhouette_score: Calculates the Silhouette Score for evaluating clustering performance.
- elbow_method: Implements the elbow method to determine the optimal number of clusters.
- initialize_centroids: Randomly initializes the centroids for KMeans clustering.
- assign_clusters: Assigns clusters based on the nearest centroid.
- update_centroids: Updates centroids based on the current cluster assignments.
- _handle_categorical: Handles categorical columns in the input data by one-hot encoding.
- _convert_to_ndarray: Converts input data to a NumPy ndarray and handles categorical columns.

Usage Example:
- Load data, initialize KMeans object, and find the optimal number of clusters using various evaluation metrics.
- Fit the KMeans model with the optimal number of clusters and visualize the clustering results.

Note: Includes example usage at the end to demonstrate how to use the KMeans class and its functionalities.
"""

import numpy as np

class KMeans:
    def __init__(self, X, n_clusters=3, max_iter=300, tol=1e-4):
        """
        Initialize the KMeans object.

        Parameters:
        - X: The data matrix (numpy array).
        - n_clusters: The number of clusters.
        - max_iter: The maximum number of iterations.
        - tol: The tolerance to declare convergence.
        """
        self.X = self._convert_to_ndarray(X).astype(float)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def _handle_categorical(self, X):
        """
        Handle categorical columns by one-hot encoding.

        Parameters:
        - X: The input data with potential categorical columns.

        Returns:
        - X_processed: The processed data with categorical columns encoded.
        """
        # Identify categorical columns
        cat_columns = [col for col in range(X.shape[1]) if isinstance(X[0, col], str)]
        
        if not cat_columns:
            return X  # No categorical columns found
        
        # Convert categorical columns to one-hot encoding
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        X_cat_encoded = encoder.fit_transform(X[:, cat_columns])
        
        # Replace categorical columns with encoded values
        X_processed = np.delete(X, cat_columns, axis=1)
        X_processed = np.hstack((X_processed, X_cat_encoded))
        
        return X_processed
    
    def _convert_to_ndarray(self, X):
        """
        Convert input data to a NumPy ndarray and handle categorical columns.

        Parameters:
        - X: The input data, which can be a list, DataFrame, or ndarray.

        Returns:
        - X_ndarray: The converted and processed input data as a NumPy ndarray.
        """
        import pandas as pd

        if isinstance(X, np.ndarray):
            X_ndarray = X.copy()
        elif isinstance(X, list):
            X_ndarray = np.array(X)
        elif isinstance(X, pd.DataFrame):
            X_ndarray = X.values
        else:
            raise ValueError("Unsupported input type. Input must be a list, NumPy array, or DataFrame.")
        
        # Handle categorical columns
        X_processed = self._handle_categorical(X_ndarray)
        
        return X_processed

    def initialize_centroids(self):
        """
        Randomly initialize the centroids.

        Returns:
        - centroids: The initialized centroids.
        """
        np.random.seed(1)  # For reproducibility
        random_indices = np.random.permutation(self.X.shape[0])
        centroids = self.X[random_indices[:self.n_clusters]]
        return centroids

    def assign_clusters(self, centroids):
        """
        Assign clusters based on the nearest centroid.

        Parameters:
        - centroids: The current centroids.

        Returns:
        - labels: The cluster assignments for each data point.
        """
        self.X = self.X.astype(float)  # Convert X to float type if necessary
        centroids = centroids.astype(float)  # Ensure centroids are float type


        distances = np.sqrt(((self.X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels

    def update_centroids(self):
        """
        Update the centroids based on the current cluster assignments.

        Returns:
        - centroids: The updated centroids.
        """
        centroids = np.array([self.X[self.labels == i].mean(axis=0) if np.sum(self.labels == i) > 0 else self.X[np.random.choice(len(self.X))] for i in range(self.n_clusters)])
        return centroids


    def fit(self):
        """
        Fit the KMeans model to the data.
        """
        self.centroids = self.initialize_centroids()
        for _ in range(self.max_iter):
            prev_centroids = np.copy(self.centroids)
            self.labels = self.assign_clusters(self.centroids)
            self.centroids = self.update_centroids()
            if np.all(np.abs(prev_centroids - self.centroids) < self.tol):
                break

    def predict(self, new_X):
        """
        Predict the closest cluster each sample in new_X belongs to.

        Parameters:
        - new_X: The data matrix to predict (numpy array).

        Returns:
        - labels: The predicted cluster labels.
        """
        distances = np.sqrt(((new_X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        return labels
    
    def elbow_method(self, max_k=10):
        """
        Implement the elbow method to determine the optimal number of clusters.

        Parameters:
        - X: The data matrix (numpy array).
        - max_k: The maximum number of clusters to test.

        Returns:
        - distortions: A list of distortions for each k.
        """
        distortions = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(self.X, n_clusters=k)
            kmeans.fit()
            # Compute the distortion (inertia)
            distortion = sum(np.min(np.sqrt(((self.X - kmeans.centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)) / self.X.shape[0]
            distortions.append(distortion)
        
        return distortions
    
    def calinski_harabasz_index(self, X, labels, centroids):
        """
        Calculate the Calinski-Harabasz Index for evaluating clustering performance.

        Parameters:
        - X: The data matrix (numpy array).
        - labels: The cluster labels for each data point.
        - centroids: The centroids of the clusters.

        Returns:
        - ch_index: The computed Calinski-Harabasz Index.
        """
        # Get unique clusters and their count
        clusters = np.unique(labels)
        n_clusters = len(clusters)
        N = len(X)  # Total number of data points

        # Handle the case of zero or one cluster
        if n_clusters <= 1:
            return 0

        # Compute the between-cluster dispersion (B(K))
        mean_total = np.mean(X, axis=0)
        Bk = np.sum([len(labels[labels == c]) * np.linalg.norm(centroids[c] - mean_total)**2 for c in clusters])

        # Compute the within-cluster dispersion (W(K))
        Wk = np.sum([np.sum(np.linalg.norm(X[labels == c] - centroids[c], axis=1)**2) for c in clusters])

        # Compute the Calinski-Harabasz Index
        ch_index = (Bk / Wk) * ((N - n_clusters) / (n_clusters - 1))

        return ch_index



    def davies_bouldin_index(self, X, labels, centroids):
        """
        Calculate the Davies-Bouldin Index for evaluating clustering performance.

        Parameters:
        - X: The data matrix (numpy array).
        - labels: The cluster labels for each data point.
        - centroids: The centroids of the clusters.

        Returns:
        - db_index: The computed Davies-Bouldin Index.
        """
        # Get unique clusters and their count
        clusters = np.unique(labels)
        n_clusters = len(clusters)
        cluster_distances = np.zeros((n_clusters, n_clusters))

        # Compute pairwise distances between centroids
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                cluster_distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
                cluster_distances[j, i] = cluster_distances[i, j]

        db_indices = []
        for i in range(n_clusters):
            indices_same_cluster = np.where(labels == clusters[i])[0]
            avg_distance_same = np.mean(np.linalg.norm(X[indices_same_cluster] - centroids[i], axis=1))

            similarity_scores = []
            for j in range(n_clusters):
                if j != i:
                    indices_other_cluster = np.where(labels == clusters[j])[0]
                    avg_distance_other = np.mean(np.linalg.norm(X[indices_other_cluster] - centroids[j], axis=1))
                    avg_distance_centroids = cluster_distances[i, j]
                    if avg_distance_centroids != 0:  # Handle division by zero
                        similarity = (avg_distance_same + avg_distance_other) / avg_distance_centroids
                        similarity_scores.append(similarity)

            if similarity_scores:  # Check if the list is not empty
                db_index = np.max(similarity_scores)
                db_indices.append(db_index)

        if db_indices:  # Check if the list is not empty
            return np.mean(db_indices)
        else:
            return 0  # Return 0 if no valid similarity scores found

    
    def silhouette_score(self, X, labels):
        """
        Calculate the silhouette score for evaluating clustering performance.

        Parameters:
        - X: The data matrix (numpy array).
        - labels: The cluster labels for each data point.

        Returns:
        - silhouette_score: The computed silhouette score.
        """
        n = len(X)
        clusters = np.unique(labels)
        silhouette_scores = []

        for i in range(n):
            a = np.mean(np.linalg.norm(X[labels == labels[i]] - X[i], axis=1))
            # Check if there are valid clusters to compare against
            valid_clusters = [c for c in clusters if c != labels[i]]
            if valid_clusters:
                b = np.min([np.mean(np.linalg.norm(X[labels == c] - X[i], axis=1)) for c in valid_clusters])
                silhouette_scores.append((b - a) / max(a, b))
        
        # Check if silhouette_scores is not empty
        if silhouette_scores:
            return np.mean(silhouette_scores)
        else:
            return 0  # Return 0 if no valid silhouette scores found


    def find_optimal_clusters(self, max_k=10, true_k=None):
        """
        Find the optimal number of clusters using various evaluation metrics and plot the results.

        Parameters:
        - X: The data matrix (numpy array).
        - max_k: The maximum number of clusters to consider.
        - true_k: The true number of clusters in the data.

        Returns:
        - ch_optimal_k: The optimal number of clusters based on the Calinski-Harabasz Index.
        - db_optimal_k: The optimal number of clusters based on the Davies-Bouldin Index.
        - silhouette_optimal_k: The optimal number of clusters based on the Silhouette Score.
        """
        import matplotlib.pyplot as plt
        X = self.X.astype(float)        

        distortions = self.elbow_method(max_k)
        ch_indices = []
        db_indices = []
        silhouette_scores = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(X,n_clusters=k)
            kmeans.fit()
            centroids = kmeans.centroids
            labels = kmeans.predict(X)
            ch_indices.append(self.calinski_harabasz_index(X, labels, centroids))
            db_indices.append(self.davies_bouldin_index(X, labels, centroids))
            silhouette_scores.append(self.silhouette_score(X, labels))

        ch_indices = np.array(ch_indices)
        db_indices = np.array(db_indices)
        silhouette_scores = np.array(silhouette_scores)

        ch_scores = (ch_indices - ch_indices.min()) / (ch_indices.max() - ch_indices.min())
        db_scores = (db_indices - db_indices.min()) / (db_indices.max() - db_indices.min())
        silhouette_scores = (silhouette_scores - silhouette_scores.min()) / (silhouette_scores.max() - silhouette_scores.min())

        ch_diffs = np.diff(ch_scores)
        db_diffs = np.diff(db_scores)
        silhouette_diffs = np.diff(silhouette_scores)

        ch_second_derivatives = np.diff(ch_diffs)
        db_second_derivatives = np.diff(db_diffs)
        silhouette_second_derivatives = np.diff(silhouette_diffs)

        ch_optimal_k = np.argmax(ch_second_derivatives) + 2  
        db_optimal_k = np.argmax(db_second_derivatives) + 2
        silhouette_optimal_k = np.argmax(silhouette_second_derivatives) + 2

        plt.figure(figsize=(15, 4))
        
        # Plot the elbow method
        plt.subplot(1, 4, 1)
        plt.plot(range(1, max_k + 1), distortions, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the Optimal k')
        if true_k: plt.axvline(x=true_k, color='black', linestyle='--', label='true_k')
        plt.legend()
        
        # Plot Calinski-Harabasz Index
        plt.subplot(1, 4, 2)
        plt.plot(range(1, max_k + 1), ch_scores, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title('Calinski-Harabasz Index for Optimal k')
        plt.axvline(x=ch_optimal_k, color='red', linestyle='--', label='optimal_k')
        if true_k: plt.axvline(x=true_k, color='black', linestyle='--', label='true_k')
        plt.legend()
        
        # Plot Davies-Bouldin Index
        plt.subplot(1, 4, 3)
        plt.plot(range(1, max_k + 1), db_scores, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Davies-Bouldin Index')
        plt.title('Davies-Bouldin Index for Optimal k')
        plt.axvline(x=db_optimal_k, color='red', linestyle='--', label='optimal_k')
        if true_k: plt.axvline(x=true_k, color='black', linestyle='--', label='true_k')
        plt.legend()

        # Plot Silhouette Score
        plt.subplot(1, 4, 4)
        plt.plot(range(1, max_k + 1), silhouette_scores, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Optimal k')
        plt.axvline(x=silhouette_optimal_k, color='red', linestyle='--', label='optimal_k')
        if true_k: plt.axvline(x=true_k, color='black', linestyle='--', label='true_k')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return ch_optimal_k, db_optimal_k, silhouette_optimal_k



# # Example usage:
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# true_k = 8
# # Generate synthetic data for testing
# X, y = make_blobs(n_samples=1000, n_features=2, centers=true_k, cluster_std=0.60, random_state=1)

# # Initialize KMeans object
# kmeans = KMeans(X)

# # Find the optimal number of clusters using the elbow method
# ch_optimal_k, db_optimal_k, silhouette_optimal_k = kmeans.find_optimal_clusters(max_k=10, true_k=true_k)

# print(f'calinski_harabasz_index number of clusters: {ch_optimal_k}')
# print(f'davies_bouldin_index number of clusters: {db_optimal_k}')
# print(f'silhouette_score number of clusters: {silhouette_optimal_k}')

# # Create a 2x2 subplot for each k value
# fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# # Define titles based on which k value method was used
# titles = [f'True "make_blobs" K = {true_k}', 
#           f'Calinski-Harabasz Index Optimal K = {ch_optimal_k}', 
#           f'Davies-Bouldin Index Optimal K = {db_optimal_k}', 
#           f'Silhouette Optimal K = {silhouette_optimal_k}']

# # Loop through each k value and plot the results of fit()
# for i, k in enumerate([true_k, ch_optimal_k, db_optimal_k, silhouette_optimal_k]):
#     # Fit the KMeans model with the current k value
#     kmeans.n_clusters = k
#     kmeans.fit()

#     # Predict cluster labels
#     labels = kmeans.predict(X)

#     # Plot the results in the corresponding subplot
#     row = i // 2
#     col = i % 2
#     axs[row, col].scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
#     centroids = kmeans.centroids
#     axs[row, col].scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.75)
#     axs[row, col].set_title(titles[i])
#     axs[row, col].set_xlabel('Feature 1')
#     axs[row, col].set_ylabel('Feature 2')

# plt.tight_layout()
# plt.show()


