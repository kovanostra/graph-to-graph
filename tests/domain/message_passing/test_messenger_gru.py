from unittest import TestCase

import numpy as np

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.message_passing.message_gru import MessageGRU
from src.domain.message_passing.messenger_gru import MessengerGRU
from src.domain.node import Node
from tests.fixtures.matrices_and_vectors import BASE_W_MATRIX, BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    MULTIPLICATION_FACTOR, BASE_B_VECTOR, BASE_GRAPH_EDGE_FEATURES


class TestMessengerGRU(TestCase):
    def setUp(self) -> None:
        self.messenger_gru = MessengerGRU()
        self.messenger_gru.w_tree_update_gate_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_gru.w_tree_forget_gate_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_gru.w_tree_current_memory_message_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_gru.u_tree_update_gate = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_gru.u_tree_forget_gate = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_gru.u_tree_current_memory_message = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.messenger_gru.b_tree_update_gate = MULTIPLICATION_FACTOR * BASE_B_VECTOR
        self.messenger_gru.b_tree_forget_gate = MULTIPLICATION_FACTOR * BASE_B_VECTOR
        self.messenger_gru.b_tree_current_memory_message = MULTIPLICATION_FACTOR * BASE_B_VECTOR

    def test_initialize_returns_matrices_of_the_correct_shape(self):
        # Given
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)
        expected_matrix_shape = BASE_W_MATRIX.shape
        expected_vector_shape = BASE_B_VECTOR.shape

        # When
        self.messenger_gru.initialize(graph=graph, weight=MULTIPLICATION_FACTOR)

        # Then
        self.assertEqual(expected_matrix_shape, self.messenger_gru.w_tree_update_gate_features.shape)
        self.assertEqual(expected_matrix_shape, self.messenger_gru.w_tree_forget_gate_features.shape)
        self.assertEqual(expected_matrix_shape, self.messenger_gru.w_tree_current_memory_message_features.shape)
        self.assertEqual(expected_matrix_shape, self.messenger_gru.u_tree_update_gate.shape)
        self.assertEqual(expected_matrix_shape, self.messenger_gru.u_tree_forget_gate.shape)
        self.assertEqual(expected_matrix_shape, self.messenger_gru.u_tree_current_memory_message.shape)
        self.assertEqual(expected_vector_shape, self.messenger_gru.b_tree_update_gate.shape)
        self.assertEqual(expected_vector_shape, self.messenger_gru.b_tree_forget_gate.shape)
        self.assertEqual(expected_vector_shape, self.messenger_gru.b_tree_current_memory_message.shape)

    def test_calculate_sum_of_messages_coming_from_neighbors_except_target_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)
        start_node = Node(graph, 2)
        end_node = Node(graph, 1)
        edge = Edge(graph, start_node, end_node)
        messages_sum_expected = MessageGRU()
        messages_sum_expected.value = 0.1 * np.array([5, 3])

        # When
        messages_sum = self.messenger_gru._get_messages_from_all_node_neighbors_except_target_summed(messages,
                                                                                                     start_node,
                                                                                                     edge)

        # Then
        self.assertTrue(np.array_equal(messages_sum_expected.value, messages_sum.value))

    def test_calculate_update_gate_output_for_a_single_node_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)
        start_node = Node(graph, 2)
        end_node = Node(graph, 1)
        edge = Edge(graph, start_node, end_node)
        update_gate_output_expected = np.array([0.60587367, 0.60587367])

        # When
        update_gate_output = self.messenger_gru._pass_through_update_gate(messages, start_node, edge, graph)

        # Then
        self.assertTrue(np.allclose(update_gate_output_expected, update_gate_output))

    def test_calculate_reset_gate_output_for_a_single_node_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)
        start_node = Node(graph, 2)
        end_node = Node(graph, 0)
        reset_edge = Edge(graph, start_node, end_node)
        reset_gate_output_expected = np.array([0.5914589784327802, 0.5914589784327802])

        # When
        reset_gate_output = self.messenger_gru._pass_through_reset_gate(messages, start_node, reset_edge, graph)

        # Then
        self.assertTrue(np.allclose(reset_gate_output_expected, reset_gate_output))

    def test_calculate_current_message_output_for_a_single_node_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)
        start_node = Node(graph, 2)
        end_node = Node(graph, 0)
        edge = Edge(graph, start_node, end_node)
        current_memory_message_expected = np.array([0.344843, 0.344843])

        # When
        current_memory_message = self.messenger_gru._get_current_memory_message(messages, start_node, edge, graph)

        # Then
        self.assertTrue(np.allclose(current_memory_message_expected, current_memory_message))

    def test_compose_messages_from_a_node_to_a_target_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        node_expected = 2
        target_expected = 0
        messages_expected = np.array([0.405994, 0.327169])
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)

        # When
        messages = self.messenger_gru.compose_messages_from_nodes_to_targets(graph,
                                                                             messages)[node_expected, target_expected]

        # Then
        self.assertTrue(np.allclose(messages_expected, messages))
