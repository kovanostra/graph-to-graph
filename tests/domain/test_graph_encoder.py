from unittest import TestCase

import numpy as np

from src.domain.graph_encoder import GraphEncoder
from src.domain.rnn_messenger import RNNMessenger
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    BASE_U_MATRIX, BASE_GRAPH_EDGE_FEATURES, BASE_W_MATRIX


class TestGraphEncoder(TestCase):

    def setUp(self) -> None:
        self.rnn_messenger = RNNMessenger()
        self.rnn_messenger.w_graph_node_features = 0.1 * BASE_W_MATRIX
        self.rnn_messenger.w_graph_edge_features = 0.1 * BASE_W_MATRIX
        self.rnn_messenger.w_graph_neighbor_messages = 0.1 * BASE_W_MATRIX
        self.graph_encoder = GraphEncoder(self.rnn_messenger)
        self.graph_encoder.u_graph_node_features = 0.1 * BASE_U_MATRIX
        self.graph_encoder.u_graph_neighbor_messages = 0.1 * BASE_U_MATRIX

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Given
        time_steps = 1
        node = 0
        node_encoding_expected = np.array([0.53, 0.53])

        # When
        node_encoding = self.graph_encoder.encode_graph(BASE_GRAPH,
                                                        BASE_GRAPH_NODE_FEATURES,
                                                        BASE_GRAPH_EDGE_FEATURES,
                                                        time_steps)[node]

        # Then
        self.assertTrue(np.allclose(node_encoding_expected, node_encoding))

    def test_encode_graph_returns_the_expected_shape(self):
        # Given
        time_steps = 1
        encoded_graph_shape_expected = BASE_GRAPH_NODE_FEATURES.shape

        # When
        encoded_graph_shape = self.graph_encoder.encode_graph(BASE_GRAPH,
                                                              BASE_GRAPH_NODE_FEATURES,
                                                              BASE_GRAPH_EDGE_FEATURES,
                                                              time_steps).shape

        # Then
        self.assertTrue(np.allclose(encoded_graph_shape_expected, encoded_graph_shape))
