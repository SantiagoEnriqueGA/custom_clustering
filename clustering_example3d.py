from clustering import DBSCAN

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Genetrate 3D data
true_k = 4
X, _ = make_blobs(n_samples=300, centers=true_k, n_features=3, cluster_std=0.60, random_state=0)

# Initialize DBSCAN
eps = 0.5
min_samples = 10
dbscan = DBSCAN(X, eps=eps, min_samples=min_samples)


# Auto Eps
best_eps, scores_dict = dbscan.auto_eps(min=0.1, max=2.0, precision=0.01, verbose=True, return_scores=True)
print('Auto Eps:', best_eps)
dbscan.eps = best_eps

# Fit and predict
labels = dbscan.fit_predict()

# Calculate Silhouette Score
silhouette_score = dbscan.silhouette_score()
print(f'Silhouette Score: {silhouette_score}')


# Plot the DBSCAN results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the data points colored by their cluster label
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50, alpha=0.5)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.title('DBSCAN clustering')
plt.show()