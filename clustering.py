"""
This is a custom implementation of the KMeans clustering algorithm along with additional functionalities for evaluating the optimal number of clusters and visualizing the clustering results.

Classes:
- KMeans: A class implementing KMeans clustering with methods for fitting the model, predicting cluster labels, and finding the optimal number of clusters.
- DBSCAN: A class implementing the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm.

Note: Includes example usage at the end to demonstrate how to use the KMeans class and its functionalities.
"""

import numpy as np

class KMeans:
    """
    This class implements the K-Means clustering algorithm along with methods for evaluating the optimal number of clusters 
    and visualizing the clustering results.

    Parameters:
    - X: The data matrix (numpy array).
    - n_clusters: The number of clusters.
    - max_iter: The maximum number of iterations.
    - tol: The tolerance to declare convergence.

    Methods:
    - __init__: Initializes the KMeans object with parameters such as the data matrix, number of clusters, maximum iterations, 
                and convergence tolerance.
    - _handle_categorical: Handles categorical columns in the input data by one-hot encoding.
    - _convert_to_ndarray: Converts input data to a NumPy ndarray and handles categorical columns.
    - initialize_centroids: Randomly initializes the centroids for KMeans clustering.
    - assign_clusters: Assigns clusters based on the nearest centroid.
    - update_centroids: Updates centroids based on the current cluster assignments.
    - fit: Fits the KMeans model to the data by iteratively updating centroids and cluster assignments until convergence.
    - predict: Predicts the closest cluster each sample in new_X belongs to.
    - elbow_method: Implements the elbow method to determine the optimal number of clusters.
    - calinski_harabasz_index: Calculates the Calinski-Harabasz Index for evaluating clustering performance.
    - davies_bouldin_index: Calculates the Davies-Bouldin Index for evaluating clustering performance.
    - silhouette_score: Calculates the Silhouette Score for evaluating clustering performance.
    - find_optimal_clusters: Implements methods to find the optimal number of clusters using the elbow method, 
                             Calinski-Harabasz Index, Davies-Bouldin Index, and Silhouette Score. It also plots the evaluation 
                             metrics to aid in determining the optimal k value.

    Usage Example:
    - Initialize KMeans object with data matrix, number of clusters, and other parameters.
    - Fit the KMeans model to the data and evaluate clustering performance using various metrics.
    - Visualize the clustering results and determine the optimal number of clusters.
    """
    def __init__(self, X, n_clusters=3, max_iter=300, tol=1e-4):
        """
        Initialize the KMeans object.

        Parameters:
        - X: The data matrix (numpy array).
        - n_clusters: The number of clusters.
        - max_iter: The maximum number of iterations.
        - tol: The tolerance to declare convergence.
        """
        self.X = self._convert_to_ndarray(X).astype(float)  # Convert input data to ndarray
        self.n_clusters = n_clusters                        # Number of clusters
        self.max_iter = max_iter                            # Maximum number of iterations
        self.tol = tol                                      # Tolerance for convergence
        self.centroids = None                               # Centroids of the clusters
        self.labels = None                                  # Cluster assignments for each data point

    def _handle_categorical(self, X):
        """
        Handle categorical columns by one-hot encoding.

        Parameters:
        - X: The input data with potential categorical columns.

        Returns:
        - X_processed: The processed data with categorical columns encoded.
        """
        cat_columns = [col for col in range(X.shape[1]) if isinstance(X[0, col], str)]  # Identify categorical columns
        
        if not cat_columns:     # No categorical columns found
            return X  
        
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)                # One-hot encoder
        X_cat_encoded = encoder.fit_transform(X[:, cat_columns])    # Replace categorical columns with encoded values
        
        X_processed = np.delete(X, cat_columns, axis=1)             # Remove categorical columns
        X_processed = np.hstack((X_processed, X_cat_encoded))       # Concatenate encoded columns
        
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

        if isinstance(X, np.ndarray):       # Check if input is already a NumPy array
            X_ndarray = X.copy()            # Create a copy of the array
        
        elif isinstance(X, list):           # Check if input is a list
            X_ndarray = np.array(X)         # Convert list to NumPy array
        
        elif isinstance(X, pd.DataFrame):   # Check if input is a DataFrame
            X_ndarray = X.values            # Convert DataFrame to NumPy array
        
        else:                               # Raise error for unsupported input type
            raise ValueError("Unsupported input type. Input must be a list, NumPy array, or DataFrame.")
        
        X_processed = self._handle_categorical(X_ndarray)   # Handle categorical columns
        
        return X_processed

    def initialize_centroids(self):
        """
        Randomly initialize the centroids.

        Returns:
        - centroids: The initialized centroids.
        """
        np.random.seed(1)                                           # Set random seed for reproducibility
        random_indices = np.random.permutation(self.X.shape[0])     # Randomly shuffle indices
        centroids = self.X[random_indices[:self.n_clusters]]        # Select the first n_clusters points as centroids
        return centroids                        

    def assign_clusters(self, centroids):
        """
        Assign clusters based on the nearest centroid.

        Parameters:
        - centroids: The current centroids.

        Returns:
        - labels: The cluster assignments for each data point.
        """
        self.X = self.X.astype(float)           # Convert X to float type if necessary
        centroids = centroids.astype(float)     # Ensure centroids are float type

        distances = np.sqrt(((self.X - centroids[:, np.newaxis])**2).sum(axis=2))   # Calculate pairwise distances
        labels = np.argmin(distances, axis=0)                                       # Assign labels based on minimum distance
        
        return labels

    def update_centroids(self):
        """
        Update the centroids based on the current cluster assignments.

        Returns:
        - centroids: The updated centroids.
        """
        centroids = np.array([                                                      # Calculate new centroids
            self.X[self.labels == i].mean(axis=0) if np.sum(self.labels == i) > 0   # Mean of points in cluster i
            else self.X[np.random.choice(len(self.X))]                              # Random point if no points in cluster
            for i in range(self.n_clusters)                                         # For each cluster
        ])
        
        return centroids


    def fit(self):
        """
        Fit the KMeans model to the data.
        """
        self.centroids = self.initialize_centroids()                        # Initialize centroids
        
        for _ in range(self.max_iter):                                      # Iterate until convergence or max_iter
            prev_centroids = np.copy(self.centroids)                        # Store previous centroids
            self.labels = self.assign_clusters(self.centroids)              # Assign clusters
            self.centroids = self.update_centroids()                        # Update centroids
            
            if np.all(np.abs(prev_centroids - self.centroids) < self.tol):  # Check for convergence
                break

    def predict(self, new_X):
        """
        Predict the closest cluster each sample in new_X belongs to.

        Parameters:
        - new_X: The data matrix to predict (numpy array).

        Returns:
        - labels: The predicted cluster labels.
        """
        distances = np.sqrt(((new_X - self.centroids[:, np.newaxis])**2).sum(axis=2))   # Calculate pairwise distances
        labels = np.argmin(distances, axis=0)                                           # Assign labels based on minimum distance
        
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
        for k in range(1, max_k + 1):               # Iterate over each k value
            kmeans = KMeans(self.X, n_clusters=k)   # Initialize KMeans with k clusters
            kmeans.fit()                            # Fit the KMeans model
            
            distortion = sum(np.min(np.sqrt(((self.X - kmeans.centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)) / self.X.shape[0]    # Calculate distortion
            distortions.append(distortion)          # Append distortion to list
        
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
        clusters = np.unique(labels)    # Unique cluster labels
        n_clusters = len(clusters)      # Number of clusters
        N = len(X)                      # Total number of data points

        if n_clusters <= 1:             # Check if there is only one cluster
            return 0

        mean_total = np.mean(X, axis=0)         # Mean of all data points

        Bk = np.sum([len(labels[labels == c]) * np.linalg.norm(centroids[c] - mean_total)**2 for c in clusters])    # Compute the between-cluster dispersion (B(K))
        Wk = np.sum([np.sum(np.linalg.norm(X[labels == c] - centroids[c], axis=1)**2) for c in clusters])           # Compute the within-cluster dispersion (W(K))

        ch_index = (Bk / Wk) * ((N - n_clusters) / (n_clusters - 1))   # Compute the Calinski-Harabasz Index

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
        clusters = np.unique(labels)                            # Unique cluster labels
        n_clusters = len(clusters)                              # Number of clusters
        cluster_distances = np.zeros((n_clusters, n_clusters))  # Pairwise distances between centroids

        for i in range(n_clusters):                 # For each cluster
            for j in range(i + 1, n_clusters):      # For each other cluster
                cluster_distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])   # Calculate distance between centroids
                cluster_distances[j, i] = cluster_distances[i, j]                       # Symmetric matrix

        db_indices = []
        for i in range(n_clusters):     # For each cluster
            indices_same_cluster = np.where(labels == clusters[i])[0]                                   # Indices of points in the same cluster
            avg_distance_same = np.mean(np.linalg.norm(X[indices_same_cluster] - centroids[i], axis=1)) # Average distance within cluster

            similarity_scores = []
            for j in range(n_clusters): # For each other cluster
                if j != i:              # Skip the same cluster
                    indices_other_cluster = np.where(labels == clusters[j])[0]                                      # Indices of points in the other cluster
                    avg_distance_other = np.mean(np.linalg.norm(X[indices_other_cluster] - centroids[j], axis=1))   # Average distance to other cluster
                    avg_distance_centroids = cluster_distances[i, j]                                                # Average distance between centroids
                    if avg_distance_centroids != 0:                                                                 # Handle division by zero
                        similarity = (avg_distance_same + avg_distance_other) / avg_distance_centroids              # Calculate similarity score
                        similarity_scores.append(similarity)                                                        # Append similarity score

            if similarity_scores:                       # Check if the list is not empty
                db_index = np.max(similarity_scores)    # Davies-Bouldin Index for the cluster
                db_indices.append(db_index)             # Append DB Index for the cluster

        if db_indices:                  # Check if the list is not empty
            return np.mean(db_indices)  # Return the mean Davies-Bouldin Index
        else:
            return 0                    # Return 0 if no valid similarity scores found

    
    def silhouette_score(self, X, labels):
        """
        Calculate the silhouette score for evaluating clustering performance.

        Parameters:
        - X: The data matrix (numpy array).
        - labels: The cluster labels for each data point.

        Returns:
        - silhouette_score: The computed silhouette score.
        """
        n = len(X)                      # Total number of data points
        clusters = np.unique(labels)    # Unique cluster labels
        silhouette_scores = []          # List to store silhouette scores

        for i in range(n):              # For each data point
            a = np.mean(np.linalg.norm(X[labels == labels[i]] - X[i], axis=1))  # Compute a(i) for the same cluster
        
            valid_clusters = [c for c in clusters if c != labels[i]]            # Other clusters
            if valid_clusters:  
                b = np.min([np.mean(np.linalg.norm(X[labels == c] - X[i], axis=1)) for c in valid_clusters])    # Compute b(i) for other clusters
                silhouette_scores.append((b - a) / max(a, b))                                                   # Compute silhouette score
        
        if silhouette_scores:
            return np.mean(silhouette_scores)   # If silhouette scores are available, return the mean
        else:
            return 0                            # Return 0 if no valid silhouette scores found


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
        
        X = self.X.astype(float)                # Convert X to float type if necessary
        distortions = self.elbow_method(max_k)  # Calculate distortions for each k

        ch_indices = []                         # Initialize lists for calinski_harabasz_index
        db_indices = []                         # Initialize lists for davies_bouldin_index
        silhouette_scores = []                  # Initialize lists for silhouette_score

        for k in range(1, max_k + 1):           # Iterate over each k value
            kmeans = KMeans(X,n_clusters=k)     # Initialize KMeans with k clusters
            kmeans.fit()                        # Fit the KMeans model
            centroids = kmeans.centroids        # Get the centroids
            labels = kmeans.predict(X)          # Predict cluster labels
            
            ch_indices.append(self.calinski_harabasz_index(X, labels, centroids))   # Calculate Calinski-Harabasz Index
            db_indices.append(self.davies_bouldin_index(X, labels, centroids))      # Calculate Davies-Bouldin Index
            silhouette_scores.append(self.silhouette_score(X, labels))              # Calculate Silhouette Score

        # Convert lists to NumPy arrays
        ch_indices = np.array(ch_indices)               
        db_indices = np.array(db_indices)
        silhouette_scores = np.array(silhouette_scores)

        # Normalize the evaluation metrics
        ch_scores = (ch_indices - ch_indices.min()) / (ch_indices.max() - ch_indices.min()) 
        db_scores = (db_indices - db_indices.min()) / (db_indices.max() - db_indices.min())
        silhouette_scores = (silhouette_scores - silhouette_scores.min()) / (silhouette_scores.max() - silhouette_scores.min())

        # Calculate the differences between consecutive scores
        ch_diffs = np.diff(ch_scores)
        db_diffs = np.diff(db_scores)
        silhouette_diffs = np.diff(silhouette_scores)

        # Calculate the second derivatives 
        ch_second_derivatives = np.diff(ch_diffs)
        db_second_derivatives = np.diff(db_diffs)
        silhouette_second_derivatives = np.diff(silhouette_diffs)

        # Find the optimal k value based on the second derivative
        ch_optimal_k = np.argmax(ch_second_derivatives) + 2  
        db_optimal_k = np.argmax(db_second_derivatives) + 2
        silhouette_optimal_k = np.argmax(silhouette_second_derivatives) + 2

        plt.figure(figsize=(15, 4))     # Set the figure size

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

        return ch_optimal_k, db_optimal_k, silhouette_optimal_k     # Return the optimal k values


class DBSCAN:
    """
    This class implements the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm.
    
    Parameters:
    - X: The data matrix (numpy array).
    - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

    Methods:
    - __init__: Initializes the DBSCAN object with the input parameters.
    - fit: Fits the DBSCAN model to the data and assigns cluster labels.
    - predict: Predicts the cluster labels for new data points.
    - fit_predict: Fits the DBSCAN model and returns cluster labels.
    - silhouette_score: Calculates the Silhouette Score for evaluating clustering performance.

    Custom Methods:
    - _handle_categorical: Handles categorical columns by one-hot encoding.
    - _convert_to_ndarray: Converts input data to a NumPy ndarray and handles categorical columns.
    - _custom_distance_matrix: Calculates the pairwise distance matrix using a custom distance calculation method.

    Usage Example:
    - Initialize DBSCAN object with data matrix and hyperparameters.
    - Fit the DBSCAN model to the data and evaluate clustering performance using silhouette score.
    """
    def __init__(self, X, eps=0.5, min_samples=5):
        """
        Initialize the DBSCAN object.

        Parameters:
        - X: The data matrix (numpy array).
        - eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.X = self._convert_to_ndarray(X).astype(float)  # Convert input data to ndarray
        self.eps = eps                                      # Maximum distance between samples
        self.min_samples = min_samples                      # Minimum number of samples for core
        self.labels = None                                  # Cluster labels for each data point

    def _handle_categorical(self, X):
        """
        Handle categorical columns by one-hot encoding.

        Parameters:
        - X: The input data with potential categorical columns.

        Returns:
        - X_processed: The processed data with categorical columns encoded.
        """
        cat_columns = [col for col in range(X.shape[1]) if isinstance(X[0, col], str)]  # Identify categorical columns
        
        if not cat_columns: # If no categorical columns found
            return X        # Return the input data
        
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)                # One-hot encoder
        X_cat_encoded = encoder.fit_transform(X[:, cat_columns])    # Replace categorical columns with encoded values
        
        X_processed = np.delete(X, cat_columns, axis=1)         # Remove categorical columns
        X_processed = np.hstack((X_processed, X_cat_encoded))   # Concatenate encoded columns
        
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

        if isinstance(X, np.ndarray):   # Check if input is already a NumPy array
            X_ndarray = X.copy()        # Create a copy of the array
        
        elif isinstance(X, list):       # Check if input is a list
            X_ndarray = np.array(X)     # Convert list to NumPy array
        
        elif isinstance(X, pd.DataFrame):   # Check if input is a DataFrame
            X_ndarray = X.values            # Convert DataFrame to NumPy array
        
        else:                           # Raise error for unsupported input type
            raise ValueError("Unsupported input type. Input must be a list, NumPy array, or DataFrame.")
        
        X_processed = self._handle_categorical(X_ndarray)   # Handle categorical columns
        
        return X_processed
    
    def _custom_distance_matrix(self, X1, X2):
        """
        Calculate the pairwise distance matrix between two sets of data points using a custom distance calculation method.

        Parameters:
        - X1: The first data matrix (numpy array).
        - X2: The second data matrix (numpy array).

        Returns:
        - dist_matrix: The pairwise distance matrix between data points in X1 and X2.
        """
        def euclidean_distance(a, b):               # Custom distance calculation function
            return np.sqrt(np.sum((a - b) ** 2))    # Euclidean distance

        n_samples1 = X1.shape[0]                            # Number of samples in X1
        n_samples2 = X2.shape[0]                            # Number of samples in X2
        dist_matrix = np.zeros((n_samples1, n_samples2))    # Initialize distance matrix

        for i in range(n_samples1):         # For each sample in X1
            for j in range(n_samples2):     # For each sample in X2
                dist_matrix[i, j] = euclidean_distance(X1[i], X2[j])    # Calculate pairwise distance

        return dist_matrix  

    def fit(self):
        """
        Fit the DBSCAN model to the data.

        Algorithm Steps:
        1. Calculate the distance matrix between all points in the dataset.
        2. Identify core points based on the minimum number of neighbors within eps distance.
        3. Assign cluster labels using depth-first search (DFS) starting from core points.

        Returns:
        - labels: The cluster labels for each data point.
        """
        dist_matrix = self._custom_distance_matrix(self.X,self.X)   # Calculate pairwise distance matrix

        core_points = np.zeros(self.X.shape[0], dtype=bool)     # Initialize core points
        labels = -1 * np.ones(self.X.shape[0], dtype=int)       # Initialize labels
        cluster_id = 0                                          # Initialize cluster ID

        for i in range(self.X.shape[0]):                        # For each data point
            neighbors = np.where(dist_matrix[i] < self.eps)[0]  # Find neighbors within eps distance
            
            if len(neighbors) >= self.min_samples:  
                core_points[i] = True               # If core point, mark as True

        for i in range(self.X.shape[0]):            # For each data point
            if labels[i] == -1 and core_points[i]:  # If core point and unassigned
                labels[i] = cluster_id              # Assign cluster ID
                stack = [i]                         # Initialize stack with core point
                
                while stack:                                                # Depth-first search (DFS)
                    point = stack.pop()                                     # Pop the top element
                    neighbors = np.where(dist_matrix[point] < self.eps)[0]  # Find neighbors within eps distance
                    
                    if len(neighbors) >= self.min_samples:                  # If core point    
                        for neighbor in neighbors:                          # For each neighbor
                            if labels[neighbor] == -1:                      # If unassigned
                                labels[neighbor] = cluster_id               # Assign cluster ID
                                stack.append(neighbor)                      # Add to stack
                cluster_id += 1     # Increment cluster ID

        self.labels = labels        # Store cluster labels
        return labels               

    def predict(self, new_X):
        """
        Predict the cluster labels for new data points.
        Note: DBSCAN does not naturally support predicting new data points.

        Parameters:
        - new_X: The data matrix to predict (numpy array).

        Returns:
        - labels: The predicted cluster labels (-1 for noise).
        """
        new_X = self._convert_to_ndarray(new_X).astype(float)       # Convert input data to ndarray        
        dist_matrix = self._custom_distance_matrix(new_X, self.X)   # Calculate distance matrix
        labels = -1 * np.ones(new_X.shape[0], dtype=int)            # Initialize labels for new data points as noise (-1)

        for i in range(new_X.shape[0]):                             # For each new data point
            neighbors = np.where(dist_matrix[i] < self.eps)[0]      # Find neighbors within eps distance
            
            if len(neighbors) > 0:                          # If neighbors found
                labels[i] = self.labels[neighbors[0]]       # Assign the cluster label of the first neighbor

        return labels

    def fit_predict(self):
        """
        Fit the DBSCAN model to the data and return the cluster labels.

        Returns:
        - labels: The cluster labels for the data.
        """
        return self.fit()   # Fits the DBSCAN model and return cluster labels

    def silhouette_score(self):
        """
        Calculate the silhouette score for evaluating clustering performance.

        Returns:
        - silhouette_score: The computed silhouette score.
        """
        if self.labels is None or len(set(self.labels)) <= 1:   # If cluster labels are available and at least two unique clusters
            return -1
        
        a = np.zeros(len(self.X))   # Initialize a(i) for each data point
        b = np.zeros(len(self.X))   # Initialize b(i) for each data point

        for i in range(len(self.X)):                        # For each data point
            same_cluster = self.labels == self.labels[i]    # Indices of points in the same cluster as point i
            other_clusters = self.labels != self.labels[i]  # Indices of points in other clusters

            if np.sum(same_cluster) > 1:                                                    # If more than one point in the same cluster
                a[i] = np.mean(np.linalg.norm(self.X[same_cluster] - self.X[i], axis=1))    # Compute a(i)
            else:
                a[i] = 0                                    # Set a(i) to 0 if only one point in the same cluster
            
            # Compute b(i) for each cluster different from point i's cluster
            b[i] = np.min([np.mean(np.linalg.norm(self.X[other_clusters & (self.labels == label)] - self.X[i], axis=1)) 
                           for label in set(self.labels) if label != -1 and label != self.labels[i]])

        silhouette_scores = (b - a) / np.maximum(a, b)  # Compute silhouette scores for all points
        silhouette_score = np.mean(silhouette_scores)   # Compute mean silhouette score
        
        return silhouette_score
    


# # Example usage:
# ------------------------
# DBSCAN
# ------------------------
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
# import numpy as np

# # Generate synthetic data for testing
# true_k = 8
# X, y = make_blobs(n_samples=1000, n_features=2, centers=true_k, cluster_std=0.60, random_state=1)

# # Initialize DBSCAN object
# eps = 0.5
# min_samples = 10
# dbscan = DBSCAN(X, eps=eps, min_samples=min_samples)

# # Fit the DBSCAN model to the data
# labels = dbscan.fit_predict()

# # Calculate the silhouette score for evaluation
# silhouette_score = dbscan.silhouette_score()
# print(f'Silhouette Score: {silhouette_score}')

# # Plot the DBSCAN results
# plt.figure(figsize=(8, 8))

# # Scatter plot of the data points colored by their cluster label
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.title(f'DBSCAN Clustering (eps={eps}, min_samples={min_samples})')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# # Plotting core points and noise points
# core_samples_mask = np.zeros_like(labels, dtype=bool)
# core_samples_mask[dbscan.labels != -1] = True

# # Plot core points
# core_points = X[core_samples_mask]
# plt.scatter(core_points[:, 0], core_points[:, 1], c=labels[core_samples_mask], s=50, cmap='viridis', edgecolors='k')

# # Plot noise points
# noise_points = X[~core_samples_mask]
# plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=50, label='Noise')

# plt.legend()
# plt.show()




# # Example usage:
# ------------------------
# KMeans
# ------------------------
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
