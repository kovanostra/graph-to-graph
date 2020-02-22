from unittest import TestCase

import numpy as np

from src.domain.graph import Graph
from src.domain.graph_encoder import GraphEncoder
from src.domain.messenger_rnn import MessengerRNN
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    BASE_U_MATRIX, BASE_GRAPH_EDGE_FEATURES, BASE_W_MATRIX


class TestGraphEncoder(TestCase):

    def setUp(self) -> None:
        self.rnn_messenger = MessengerRNN()
        self.rnn_messenger.w_graph_node_features = 0.1 * BASE_W_MATRIX
        self.rnn_messenger.w_graph_edge_features = 0.1 * BASE_W_MATRIX
        self.rnn_messenger.w_graph_neighbor_messages = 0.1 * BASE_W_MATRIX
        self.graph_encoder = GraphEncoder(messenger=self.rnn_messenger, time_steps=1)
        self.graph_encoder.u_graph_node_features = 0.1 * BASE_U_MATRIX
        self.graph_encoder.u_graph_neighbor_messages = 0.1 * BASE_U_MATRIX

    def test_encode_graph_returns_the_expected_encoding_for_a_node_after_one_time_step(self):
        # Given
        node = 0
        node_encoding_expected = np.array([0.53, 0.53])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)

        # When
        node_encoding = self.graph_encoder.encode(graph)[node]

        # Then
        self.assertTrue(np.allclose(node_encoding_expected, node_encoding))

    def test_encode_graph_returns_the_expected_shape(self):
        # Given
        encoded_graph_shape_expected = BASE_GRAPH_NODE_FEATURES.shape
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)

        # When
        encoded_graph_shape = self.graph_encoder.encode(graph).shape

        # Then
        self.assertTrue(np.allclose(encoded_graph_shape_expected, encoded_graph_shape))
