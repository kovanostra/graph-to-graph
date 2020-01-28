from unittest import TestCase

import numpy as np

from tests.fixtures.matrices_and_vectors import BASE_GRAPH_NODES_NUMBER, BASE_GRAPH_NODE_FEATURES, \
    BASE_GRAPH_EDGE_FEATURES, BASE_GRAPH, BASE_W_MATRIX
from src.domain.rnn_messenger import RNNMessenger


class TestRNNMessenger(TestCase):
    def setUp(self) -> None:
        self.rnn_messenger = RNNMessenger()
        self.rnn_messenger.w_graph_node_features = 0.1 * BASE_W_MATRIX
        self.rnn_messenger.w_graph_edge_features = 0.1 * BASE_W_MATRIX
        self.rnn_messenger.w_graph_neighbor_messages = 0.1 * BASE_W_MATRIX

    def test_get_the_messages_matrix_with_same_dimensions_as_the_graph(self):
        # Given
        messages_expected = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                      BASE_GRAPH_NODES_NUMBER,
                                      BASE_GRAPH_NODE_FEATURES.shape[1]))

        # When
        messages = self.rnn_messenger.compose_messages_from_nodes_to_targets(BASE_GRAPH,
                                                                             BASE_GRAPH_NODE_FEATURES,
                                                                             BASE_GRAPH_EDGE_FEATURES,
                                                                             messages_expected)

        # Then
        self.assertTrue(np.array_equal(messages_expected.shape, messages.shape))

    def test_get_the_messages_matrix_with_the_same_links_as_the_graph(self):
        # Given
        messages_initial = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODE_FEATURES.shape[1]))
        messages_non_zero_expected = np.nonzero(BASE_GRAPH)[1]

        # When
        messages = self.rnn_messenger.compose_messages_from_nodes_to_targets(BASE_GRAPH,
                                                                             BASE_GRAPH_NODE_FEATURES,
                                                                             BASE_GRAPH_EDGE_FEATURES,
                                                                             messages_initial)
        messages_non_zero = np.nonzero(np.sum(messages, axis=2))[1]

        # Then
        self.assertTrue(np.array_equal(messages_non_zero_expected, messages_non_zero))

    def test_get_the_expected_messages_from_the_a_node_after_one_time_step(self):
        messages_initial = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODE_FEATURES.shape[1]))
        node_expected = 0
        messages_from_node_expected = np.array([[0., 0.],
                                                [0.6, 0.6],
                                                [0.55, 0.55],
                                                [0., 0.]])

        # When
        messages_from_node = self.rnn_messenger.compose_messages_from_nodes_to_targets(BASE_GRAPH,
                                                                                       BASE_GRAPH_NODE_FEATURES,
                                                                                       BASE_GRAPH_EDGE_FEATURES,
                                                                                       messages_initial)[node_expected]

        # Then
        self.assertTrue(np.allclose(messages_from_node_expected, messages_from_node))
