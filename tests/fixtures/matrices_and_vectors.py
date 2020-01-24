import numpy as np

BASE_GRAPH = np.array([[0, 1, 1, 0],
                       [1, 0, 1, 0],
                       [1, 1, 0, 1],
                       [0, 0, 1, 0]])
BASE_GRAPH_NODE_FEATURES = np.array([[1, 2], [1, 1], [2, 0.5], [0.5, 0.5]])
BASE_UNITY_MATRIX = np.ones((BASE_GRAPH_NODE_FEATURES.shape[1], BASE_GRAPH_NODE_FEATURES.shape[1]))
BASE_ZEROS_MATRIX = np.zeros((BASE_GRAPH_NODE_FEATURES.shape[1], BASE_GRAPH_NODE_FEATURES.shape[1]))

BASE_GRAPH_EDGE_FEATURES = np.array([[[0.0, 0.0],   [1.0, 2.0], [2.0, 0.5],   [0.0, 0.0]],
                                     [[1.0, 2.0],   [0.0, 0.0], [1.0, 1.0],   [0.0, 0.0]],
                                     [[2.0, 0.5],   [1.0, 1.0], [0.0, 0.0],   [0.5, 0.5]],
                                     [[0.0, 0.0],   [0.0, 0.0], [0.5, 0.5],   [0.0, 0.0]]])
BASE_GRAPH_NODES_NUMBER = BASE_GRAPH.shape[0]
BASE_GRAPH_EDGES_NUMBER = 2 * BASE_GRAPH_EDGE_FEATURES.shape[0]
BASE_W_MATRIX = np.array([[BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX],
                          [BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX],
                          [BASE_UNITY_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX],
                          [BASE_ZEROS_MATRIX, BASE_ZEROS_MATRIX, BASE_UNITY_MATRIX, BASE_ZEROS_MATRIX]])
BASE_U_MATRIX = np.array([BASE_UNITY_MATRIX,
                          BASE_UNITY_MATRIX,
                          BASE_UNITY_MATRIX,
                          BASE_UNITY_MATRIX])
