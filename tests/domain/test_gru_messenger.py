from unittest import TestCase

import numpy as np

from src.domain.gru_message import GRUMessage
from src.domain.gru_messenger import GRUMessenger
from src.domain.node import Node
from tests.fixtures.matrices_and_vectors import BASE_W_MATRIX, BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, \
    MULTIPLICATION_FACTOR, BASE_B_VECTOR


class TestGRUMessenger(TestCase):
    def setUp(self) -> None:
        self.gru_messenger = GRUMessenger()
        self.gru_messenger.w_tree_update_gate_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.gru_messenger.w_tree_forget_gate_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.gru_messenger.w_tree_current_memory_message_features = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.gru_messenger.u_tree_update_gate = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.gru_messenger.u_tree_forget_gate = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.gru_messenger.u_tree_current_memory_message = MULTIPLICATION_FACTOR * BASE_W_MATRIX
        self.gru_messenger.b_tree_update_gate = MULTIPLICATION_FACTOR * BASE_B_VECTOR
        self.gru_messenger.b_tree_forget_gate = MULTIPLICATION_FACTOR * BASE_B_VECTOR
        self.gru_messenger.b_tree_current_memory_message = MULTIPLICATION_FACTOR * BASE_B_VECTOR

    def test_calculate_sum_of_messages_coming_from_neighbors_except_target_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        current_node = Node(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, 2)
        target_node_index = 1
        messages_sum_expected = GRUMessage()
        messages_sum_expected.value = 0.1 * np.array([5, 3])

        # When
        messages_sum = self.gru_messenger._get_messages_from_all_node_neighbors_except_target_summed(messages,
                                                                                                     current_node,
                                                                                                     target_node_index)

        # Then
        self.assertTrue(np.array_equal(messages_sum_expected.value, messages_sum.value))

    def test_calculate_update_gate_output_for_a_single_node_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        current_node = Node(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, 2)
        target_node_index = 1
        current_node.set_target(target_node_index)
        update_gate_output_expected = np.array([0.60587367, 0.60587367])

        # When
        update_gate_output = self.gru_messenger._pass_through_update_gate(messages, current_node, target_node_index,
                                                                          BASE_GRAPH_NODE_FEATURES)

        # Then
        self.assertTrue(np.allclose(update_gate_output_expected, update_gate_output))

    def test_calculate_reset_gate_output_for_a_single_node_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        current_node = Node(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, 2)
        target_node_index = 0
        current_node.set_target(target_node_index)
        reset_node_index = 1
        reset_gate_output_expected = np.array([0.5914589784327802, 0.5914589784327802])

        # When
        reset_gate_output = self.gru_messenger._pass_through_reset_gate(messages, current_node,
                                                                        BASE_GRAPH_NODE_FEATURES, reset_node_index)

        # Then
        self.assertTrue(np.allclose(reset_gate_output_expected, reset_gate_output))

    def test_calculate_current_message_output_for_a_single_node_after_one_time_step(self):
        # Given
        messages = MULTIPLICATION_FACTOR * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                                     [[1, 1], [0, 0], [1, 1], [0, 0]],
                                                     [[1, 1], [1, 1], [0, 0], [4, 2]],
                                                     [[0, 0], [0, 0], [4, 2], [0, 0]]])
        current_node = Node(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, 2)
        target_node_index = 0
        current_node.set_target(target_node_index)
        current_memory_message_expected = np.array([0.344843, 0.344843])

        # When
        current_memory_message = self.gru_messenger._get_current_memory_message(messages, current_node,
                                                                                BASE_GRAPH_NODE_FEATURES,
                                                                                target_node_index)

        # Then
        self.assertTrue(np.allclose(current_memory_message_expected, current_memory_message))
