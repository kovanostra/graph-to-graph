import numpy as np


class RNNMessage:
    def __init__(self):
        self.node_input = None
        self.edge_input = None
        self.neighbors_input = None
        self.value = None

    def compose(self):
        self.value = np.transpose(self._relu(self.node_input + self.edge_input + self.neighbors_input))

    @staticmethod
    def _relu(vector: np.ndarray) -> np.ndarray:
        return np.maximum(0, vector)
