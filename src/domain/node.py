from typing import Tuple

import numpy as np


class Node:
    def __init__(self, graph: np.ndarray, all_nodes_features: np.ndarray, node_id: int):
        self.graph = graph
        self.node_id = node_id
        self.features = self._get_node_features(all_nodes_features)
        self.neighbors = self._get_neighbors(graph)
        self.neighbors_count = len(self.neighbors)
        self.current_target = None

    def set_target(self, target_index: int):
        self.current_target = self.neighbors[target_index]

    def get_slice_to_target(self) -> Tuple:
        node_slice = None
        if self.current_target is not None:
            node_slice = (self.node_id, self.current_target)
        return node_slice

    def get_slice_to_neighbors_without_current_target(self, target_node_index: int) -> Tuple:
        return self._remove_current_target_from_neighbors(target_node_index), self.node_id

    def to_string(self):
        return "Node: " + str(self.node_id) + ". Features: " + str(self.features) + \
               ". Neighbors: " + str(self.neighbors) + ". Current target: " + str(self.current_target)

    def _get_node_features(self, all_features: np.ndarray) -> np.ndarray:
        return all_features[self.node_id]

    def _get_neighbors(self, graph: np.ndarray) -> np.ndarray:
        return np.nonzero(graph[self.node_id])[0]

    def _remove_current_target_from_neighbors(self, target_node_index: int) -> np.ndarray:
        return np.delete(self.neighbors, target_node_index)
