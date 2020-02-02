from unittest import TestCase

import numpy as np

from src.domain.gru_message import GRUMessage
from src.domain.gru_messenger import GRUMessenger
from src.domain.node import Node
from tests.fixtures.matrices_and_vectors import BASE_W_MATRIX, BASE_GRAPH, BASE_GRAPH_NODE_FEATURES


class TestGRUMessenger(TestCase):
    def setUp(self) -> None:
        self.gru_messenger = GRUMessenger()
        self.gru_messenger.w_tree_update_gate_features = 0.1 * BASE_W_MATRIX
        self.gru_messenger.w_tree_forget_gate_features = 0.1 * BASE_W_MATRIX
        self.gru_messenger.w_tree_current_message_features = 0.1 * BASE_W_MATRIX
        self.gru_messenger.u_tree_update_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.u_tree_forget_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.u_tree_current_message = 0.1 * BASE_W_MATRIX
        self.gru_messenger.b_tree_update_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.b_tree_forget_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.b_tree_current_message = 0.1 * BASE_W_MATRIX

    def test_calculate_sum_of_messages_coming_from_neighbors_except_target_after_one_time_step(self):
        # Given
        messages = 0.1 * np.array([[[0, 0], [1, 1], [1, 1], [0, 0]],
                                   [[1, 1], [0, 0], [1, 1], [0, 0]],
                                   [[1, 1], [1, 1], [0, 0], [4, 2]],
                                   [[0, 0], [0, 0], [4, 2], [0, 0]]])
        current_node = Node(BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, 2)
        target_node_index = 1
        messages_sum_expected = GRUMessage()
        messages_sum_expected.value = 0.1 * np.array([5, 3])

        # When
        messages_sum = self.gru_messenger._get_messages_from_all_node_neighbors_except_target(messages, current_node,
                                                                                              target_node_index)

        # Then
        self.assertTrue(np.array_equal(messages_sum_expected.value, messages_sum.value))
