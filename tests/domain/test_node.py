import os
from unittest import TestCase

import numpy as np

print(os.getcwd())
from tests.fixtures.matrices_and_vectors import BASE_GRAPH_NODE_FEATURES, BASE_GRAPH
from src.domain.node import Node


class TestNode(TestCase):
    def setUp(self) -> None:
        self.current_node_id = 2
        self.node = Node(self.current_node_id, BASE_GRAPH_NODE_FEATURES, BASE_GRAPH)

    def test_set_target(self):
        # Given
        target_index = 0
        target_expected = 0

        # When
        self.node.set_target(target_index)

        # Then
        self.assertEqual(target_expected, self.node.current_target)

    def test_get_slice_to_target(self):
        # Given
        target_index = 1
        self.node.set_target(target_index)
        slice_expected = (self.current_node_id, self.node.current_target)

        # When
        actual_slice = self.node.get_slice_to_target()

        # Then
        self.assertEqual(slice_expected, actual_slice)

    def test_get_slice_to_neighbors_without_current_target(self):
        # Given
        target_index = 1
        self.node.set_target(target_index)
        slice_expected = (np.array([0, 3]), 2)

        # When
        actual_slice = self.node.get_slice_to_neighbors_without_current_target(target_index)

        # Then
        self.assertTrue(np.array_equal(slice_expected[0], actual_slice[0]))
        self.assertTrue(np.array_equal(slice_expected[1], actual_slice[1]))

    def test_to_string(self):
        # Given
        node_features_expected = np.array([2, 0.5])
        node_neighbors_expected = np.array([0, 1, 3])
        target_index = 1
        self.node.set_target(target_index)
        current_target_expected = 1
        node_to_string_expected = "Node: " + str(self.current_node_id) + \
                                  ". Features: " + str(node_features_expected) + \
                                  ". Neighbors: " + str(node_neighbors_expected) + \
                                  ". Current target: " + str(current_target_expected)
        # When

        # Then
        self.assertEqual(node_to_string_expected, self.node.to_string())
