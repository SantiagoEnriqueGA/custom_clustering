import unittest
import numpy as np
from clustering import KMeans
from clustering import DBSCAN

class TestKMeans(unittest.TestCase):

    def setUp(self):
        # Generate synthetic data for testing
        self.true_k = 3
        self.X = np.array([
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
            [8.0, 8.0], [1.0, 0.6], [9.0, 11.0],
            [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]
        ])
        self.kmeans = KMeans(self.X, n_clusters=self.true_k)

    def test_initialization(self):
        self.assertEqual(self.kmeans.n_clusters, self.true_k)
        self.assertEqual(self.kmeans.max_iter, 300)
        self.assertEqual(self.kmeans.tol, 1e-4)
        self.assertIsNone(self.kmeans.centroids)
        self.assertIsNone(self.kmeans.labels)

    def test_fit(self):
        self.kmeans.fit()
        self.assertEqual(len(self.kmeans.centroids), self.true_k)
        self.assertEqual(len(self.kmeans.labels), len(self.X))

    def test_predict(self):
        self.kmeans.fit()
        new_X = np.array([[0.0, 0.0], [12.0, 3.0]])
        labels = self.kmeans.predict(new_X)
        self.assertEqual(len(labels), len(new_X))

    def test_elbow_method(self):
        distortions = self.kmeans.elbow_method(max_k=5)
        self.assertEqual(len(distortions), 5)

    def test_find_optimal_clusters(self):
        ch_optimal_k, db_optimal_k, silhouette_optimal_k = self.kmeans.find_optimal_clusters(max_k=5)
        self.assertTrue(1 <= ch_optimal_k <= 5)
        self.assertTrue(1 <= db_optimal_k <= 5)
        self.assertTrue(1 <= silhouette_optimal_k <= 5)

class TestDBSCAN(unittest.TestCase):
    
    def setUp(self):
        # Generate synthetic data for testing
        self.X = np.array([
            [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
            [8.0, 8.0], [1.0, 0.6], [9.0, 11.0],
            [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]
        ])
        self.dbscan = DBSCAN(self.X, eps=1.5, min_samples=2)

    def test_initialization(self):
        self.assertEqual(self.dbscan.eps, 1.5)
        self.assertEqual(self.dbscan.min_samples, 2)
        self.assertIsNone(self.dbscan.labels)

    def test_fit(self):
        labels = self.dbscan.fit()
        self.assertEqual(len(labels), len(self.X))
        self.assertTrue(all(label >= -1 for label in labels))

    def test_predict(self):
        self.dbscan.fit()
        new_X = np.array([[0.0, 0.0], [12.0, 3.0]])
        labels = self.dbscan.predict(new_X)
        self.assertEqual(len(labels), len(new_X))
        self.assertTrue(all(label >= -1 for label in labels))

    def test_silhouette_score(self):
        self.dbscan.fit()
        score = self.dbscan.silhouette_score()
        self.assertTrue(-1 <= score <= 1)

    def test_auto_eps(self):
        best_eps, scores_dict = self.dbscan.auto_eps(return_scores=True)
        self.assertTrue(0.1 <= best_eps <= 1.1)
        self.assertTrue(all(-1 <= score <= 1 for score in scores_dict.values()))

if __name__ == "__main__":
    unittest.main()