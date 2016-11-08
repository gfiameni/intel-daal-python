# Intel DAAL related imports
from daal.data_management import HomogenNumericTable

# Helpersfor getArrayFromNT and printNT. See utils.py
from utils import *

# Import numpy, matplotlib, seaborn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plotting configurations
%config InlineBackend.figure_format = 'retina'
plt.rcParams["figure.figsize"] = (12, 9)

from sklearn.datasets import load_iris

iris = load_iris()
print('Shape:', iris.data.shape)
print('Features:', iris.feature_names)
print('Labels: ', iris.target_names)

import daal.algorithms.kmeans as kmeans
from daal.algorithms.kmeans import init


class KMeans:

    def __init__(self, nclusters, randomseed = None):
        """Initialize class parameters
        
        Args:
           nclusters: Number of clusters
           randomseed: An integer used to seed the random number generator
        """

        self.nclusters_ = nclusters
        self.seed_ = 1234 if randomseed is None else randomseed
        self.centroids_ = None
        self.assignments_ = None
        self.goalfunction_ = None
        self.niterations_ = None


    def compute(self, data, centroids = None, maxiters = 100):
        """Compute K-Means clustering for the input data

        Args:
           data: Input data to be clustered
           centroids: User defined input centroids. If None then initial
               centroids will be randomly chosen
           maxiters: The maximum number of iterations
        """

        if centroids is None:
            # Create an algorithm object for centroids initialization
            init_alg = init.Batch_Float64RandomDense(self.nclusters_)
            # Set input
            init_alg.input.set(init.data, data)
            # Set parameters
            init_alg.parameter.seed = self.seed_
            # Compute initial centroids
            self.centroids_ = init_alg.compute().get(init.centroids)
        else:
            self.centroids_ = centroids

        # Create an algorithm object for clustering
        clustering_alg = kmeans.Batch_Float64LloydDense(
                self.nclusters_,
                maxiters)
        # Set input
        clustering_alg.input.set(kmeans.data, data)
        clustering_alg.input.set(kmeans.inputCentroids, self.centroids_)
        # compute
        result = clustering_alg.compute()
        self.centroids_ = result.get(kmeans.centroids)
        self.assignments_ = result.get(kmeans.assignments)
        self.goalfunction_ = result.get(kmeans.goalFunction)
        self.niterations_ = result.get(kmeans.nIterations)
		
		
# Create a NumericTable from the Iris dataframe
iris_data = HomogenNumericTable(iris.data.astype(dtype=np.double))

# The number of clusters is 3, as there're 3 labels
nclusters = len(np.unique(iris.target))

# K-Means clustering
clustering = KMeans(nclusters)
clustering.compute(iris_data)
assignments = getArrayFromNT(clustering.assignments_).flatten().astype(np.int)

# Visualize 3 clusters using a 3D plot
from mpl_toolkits.mplot3d import Axes3D

plt.set_cmap(plt.cm.prism)
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(iris.data[:, 3], iris.data[:, 0], iris.data[:, 2], c=assignments)
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()

