from unittest import TestCase

import numpy as np

from src.domain.graph import Graph
from src.domain.message_passing.messenger_rnn import MessengerRNN
from tests.fixtures.matrices_and_vectors import BASE_GRAPH_NODES_NUMBER, BASE_GRAPH_NODE_FEATURES, \
    BASE_GRAPH_EDGE_FEATURES, BASE_GRAPH, BASE_W_MATRIX, MULTIPLICATION_FACTOR


class TestMessengerRNN(TestCase):
    def setUp(self) -> None:
        self.messenger_rnn = MessengerRNN()
        self.messenger_rnn.w_graph_node_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_rnn.w_graph_edge_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_rnn.w_graph_neighbor_messages = MULTIPLICATION_FACTOR * BASE_W_MATRIX

    def test_initialize_returns_matrices_of_the_correct_shape(self):
        # Given
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)
        expected_shape = BASE_W_MATRIX.shape

        # When
        self.messenger_rnn.initialize(graph=graph, weight=MULTIPLICATION_FACTOR)

        # Then
        self.assertEqual(expected_shape, self.messenger_rnn.w_graph_node_features.shape)
        self.assertEqual(expected_shape, self.messenger_rnn.w_graph_edge_features.shape)
        self.assertEqual(expected_shape, self.messenger_rnn.w_graph_neighbor_messages.shape)

    def test_get_the_messages_matrix_with_same_dimensions_as_the_graph(self):
        # Given
        messages_expected = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                      BASE_GRAPH_NODES_NUMBER,
                                      BASE_GRAPH_NODE_FEATURES.shape[1]))
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)

        # When
        messages = self.messenger_rnn.compose_messages_from_nodes_to_targets(graph, messages_expected)

        # Then
        self.assertTrue(np.array_equal(messages_expected.shape, messages.shape))

    def test_get_the_messages_matrix_with_the_same_links_as_the_graph(self):
        # Given
        messages_initial = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODE_FEATURES.shape[1]))
        messages_non_zero_expected = np.nonzero(BASE_GRAPH)[1]
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)

        # When
        messages = self.messenger_rnn.compose_messages_from_nodes_to_targets(graph, messages_initial)
        messages_non_zero = np.nonzero(np.sum(messages, axis=2))[1]

        # Then
        self.assertTrue(np.array_equal(messages_non_zero_expected, messages_non_zero))

    def test_get_the_expected_messages_from_a_node_after_one_time_step(self):
        messages_initial = np.zeros((BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODES_NUMBER,
                                     BASE_GRAPH_NODE_FEATURES.shape[1]))
        node_expected = 0
        messages_from_node_expected = np.array([[0., 0.],
                                                [0.6, 0.6],
                                                [0.55, 0.55],
                                                [0., 0.]])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)

        # When
        messages_from_node = self.messenger_rnn.compose_messages_from_nodes_to_targets(graph,
                                                                                       messages_initial)[node_expected]

        # Then
        self.assertTrue(np.allclose(messages_from_node_expected, messages_from_node))
