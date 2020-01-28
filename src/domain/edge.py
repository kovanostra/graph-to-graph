import numpy as np


class Edge:
    def __init__(self, node_id: int, current_target: int, features: np.ndarray):
        self.node_id = node_id
        self.current_target = current_target
        self.features = self._extract_edge_features(features)

    def _extract_edge_features(self, features: np.ndarray) -> np.ndarray:
        return features[self.node_id, self.current_target]
