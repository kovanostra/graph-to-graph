import numpy as np

from src.domain.interface.message import Message


class RNNMessage(Message):
    def __init__(self):
        super().__init__()
        self.node_input = None
        self.edge_input = None
        self.neighbors_input = None

    def compose(self) -> None:
        self.value = np.transpose(self._relu(self.node_input + self.edge_input + self.neighbors_input))

    @staticmethod
    def _relu(vector: np.ndarray) -> np.ndarray:
        return np.maximum(0, vector)
