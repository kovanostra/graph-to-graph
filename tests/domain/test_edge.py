from unittest import TestCase

import numpy as np

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.node import Node
from tests.fixtures.matrices_and_vectors import BASE_GRAPH, BASE_GRAPH_NODE_FEATURES, BASE_GRAPH_EDGE_FEATURES


class TestEdge(TestCase):
    def setUp(self) -> None:
        graph = Graph(BASE_GRAPH,
                      BASE_GRAPH_NODE_FEATURES,
                      BASE_GRAPH_EDGE_FEATURES)
        start_node = Node(graph, 2)
        end_node = Node(graph, 0)
        self.edge = Edge(graph, start_node, end_node)

    def test_get_edge_slice(self):
        # Given
        edge_slice_expected = (2, 0)

        # When
        edge_slice = self.edge.get_edge_slice()

        # Then
        self.assertEqual(edge_slice_expected, edge_slice)

    def test_get_start_node_neighbors_without_end_node(self):
        # Given
        start_node_neighbors_expected = np.array([1, 3])

        # When
        start_node_neighbors = self.edge.get_start_node_neighbors_without_end_node()[0]

        # Then
        self.assertTrue(np.array_equal(start_node_neighbors_expected, start_node_neighbors))
