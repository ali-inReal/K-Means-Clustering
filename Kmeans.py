import numpy as np
import matplotlib.pyplot as plt


def distance(val1, val2):
    return np.sqrt(np.sum((val1-val2)**2))


class Kmeans:
    def __init__(self, k, max_iterations):
        self.k = k
        self.max_iterations = max_iterations

        # Clusters

        self.clusters = [[] for _ in range(self.k)]

        self.centroids = []

    def predict(self, X):
        self.X = X
        self.noOfSamples, self.nFeatures = X.shape

        # intialize random centroids

        random_indices = np.random.choice(
            self.noOfSamples, self.k, replace=False)

        self.centroids = [self.X[idx] for idx in random_indices]

        for i in range(self.max_iterations):
            # assigning samples to clusters
            self.clusters = self.createCluster(self.centroids)
            oldCentroids = self.centroids
            self.centroids = self.calculateCentroid(self.clusters)
            flag = self.centroidsReapeated(oldCentroids, self.centroids)
            if(flag):
                break

        return self.assignLabels(self.clusters)

    def assignLabels(self, clusters):
        labels = np.empty(self.noOfSamples)
        for cl_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cl_idx
        return labels

    def createCluster(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, value in enumerate(self.X):
            centroidIndex = self.closestCentroid(value, centroids)
            clusters[centroidIndex].append(idx)
        return clusters

    def closestCentroid(self, value, centroids):
        distances = [distance(value, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def calculateCentroid(self, clusters):
        centroids = np.zeros((self.k, self.nFeatures))
        for index, cluster in enumerate(clusters):
            clusterMean = np.mean(self.X[cluster])
            centroids[index] = clusterMean
        return centroids

    def centroidsReapeated(self, oldCentroids, centroids):
        flag = False
        for x in range(self.k):
            if(distance(oldCentroids[x], centroids[x]) == 0):
                flag = True
            else:
                flag = False
        return flag

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()


def plotData(X):
    fig, ax = plt.subplots(figsize=(12, 8))
    point = X.T
    ax.scatter(*point)
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)
    plotData(X)
    clusters = len(np.unique(y))
    print(clusters)

    k = Kmeans(k=clusters, max_iterations=150)
    y_pred = k.predict(X)

    k.plot()
