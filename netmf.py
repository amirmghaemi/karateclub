import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from karateclub.estimator import Estimator

from time import time

class NetMF(Estimator):
    r"""An implementation of `"NetMF" <https://keg.cs.tsinghua.edu.cn/jietang/publications/WSDM18-Qiu-et-al-NetMF-network-embedding.pdf>`_
    from the WSDM '18 paper "Network Embedding as Matrix Factorization: Unifying
    DeepWalk, LINE, PTE, and Node2Vec". The procedure uses sparse truncated SVD to
    learn embeddings for the pooled powers of the PMI matrix computed from powers
    of the normalized adjacency matrix.

    Args:
        dimensions (int): Number of embedding dimension. Default is 32.
        iteration (int): Number of SVD iterations. Default is 10.
        order (int): Number of PMI matrix powers. Default is 2.
        negative_samples (in): Number of negative samples. Default is 1.
        seed (int): SVD random seed. Default is 42.
    """
    def __init__(self, dimensions=32, iteration=10, order=2, negative_samples=1, seed=42):
        self.dimensions       = dimensions
        self.iterations       = iteration
        self.order            = order
        self.negative_samples = negative_samples
        self.seed             = seed

    def _create_D_inverse(self, graph):
        """
        Creating a sparse inverse degree matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **D_inverse** *(Scipy array)* - Diagonal inverse degree matrix.
        """
        index     = np.arange(graph.number_of_nodes())
        values    = np.array([1.0/graph.degree[node] for node in range(graph.number_of_nodes())])
        shape     = (graph.number_of_nodes(), graph.number_of_nodes())
        D_inverse = sparse.csr_matrix((values, (index, index)), shape=shape) # @ANN:init():csr
        return D_inverse

    def _create_base_matrix(self, graph):
        """
        Creating the normalized adjacency matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **(A_hat, A_hat, A_hat, D_inverse)** *(SciPy arrays)* - Normalized adjacency matrices.
        """
        A         = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes())) # @ANN:convert(nx):csr
        D_inverse = self._create_D_inverse(graph)
        A_hat     = D_inverse @ A # @ANN:matmul(csr, csr):csr
        return (A_hat, A_hat, A_hat, D_inverse)

    def _create_target_matrix(self, graph):
        """
        Creating a log transformed target matrix.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.

        Return types:
            * **target_matrix** *(SciPy array)* - The shifted PMI matrix.
        """
        A_pool, A_tilde, A_hat, D_inverse = self._create_base_matrix(graph)
        for _ in range(self.order - 1):
            A_tilde = A_tilde @ A_hat  # @ANN:matmul(csr, csr):csr
            A_pool  = A_pool + A_tilde # @ANN:add(csr, csr):csr
        
        A_pool.data *= graph.number_of_edges() / (self.order * self.negative_samples)
        A_pool = A_pool @ D_inverse  # @ANN:matmul(csr, csr):csr

        A_pool.data[A_pool.data < 1.0] = 1.0
        A_pool.data = np.log(A_pool.data)
        
        return A_pool

    def _create_embedding(self, target_matrix):
        """
        Fitting a truncated SVD embedding of a PMI matrix.
        """
        svd = TruncatedSVD(n_components=self.dimensions,
                           n_iter=self.iterations,
                           random_state=self.seed) # @ANN:truncated_svd(coo):dense
        
        return svd.fit_transform(target_matrix)

    def fit(self, graph):
        """
        Fitting a NetMF model.
    
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be embedded.
        """
        self._check_graph(graph)
        target_matrix   = self._create_target_matrix(graph)
        self._embedding = self._create_embedding(target_matrix)

    def get_embedding(self):
        r"""Getting the node embedding.
    
        Return types:
            * **embedding** *(Numpy array)* - The embedding of nodes.
        """
        return self._embedding
