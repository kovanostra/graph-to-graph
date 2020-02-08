import numpy as np

from src.domain.Graph import Graph


class Edge:
    def __init__(self, start_node: int, end_node: int, graph: Graph) -> None:
        self.start_node = start_node
        self.end_node = end_node
        self.features = self._extract_edge_features(graph)

    def _extract_edge_features(self, graph: Graph) -> np.ndarray:
        return graph.edge_features[self.start_node, self.end_node]
