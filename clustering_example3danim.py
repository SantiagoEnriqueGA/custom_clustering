from clustering import DBSCAN

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Genetrate 3D data
true_k = 4
X, _ = make_blobs(n_samples=300, centers=true_k, n_features=3, cluster_std=0.60, random_state=0)

# Initialize DBSCAN
eps = 0.5
min_samples = 10
dbscan = DBSCAN(X, eps=eps, min_samples=min_samples)


# Auto Eps
best_eps, scores_dict = dbscan.auto_eps(min=0.1, max=1.1, precision=0.1, verbose=True, return_scores=True)
print('Auto Eps:', best_eps)
dbscan.eps = best_eps

# Store labels for each eps value
labels_list = []
eps_values = []
for eps, score in scores_dict.items():
    dbscan = DBSCAN(X, eps=eps, min_samples=min_samples)
    labels_list.append(dbscan.fit_predict())
    eps_values.append(eps)

# Animate plot the DBSCAN results of all eps results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()
    labels = labels_list[frame]
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o')
    ax.set_title(f'DBSCAN with eps={eps_values[frame]:.2f}')
    
    # Create legend
    handles, _ = scatter.legend_elements()
    ax.legend(handles, [f'Cluster {i}' for i in range(len(handles))], title="Clusters")

ani = FuncAnimation(fig, update, frames=len(labels_list), interval=500, repeat=True)
plt.show()