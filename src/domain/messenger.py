from abc import ABCMeta, abstractmethod

import numpy as np


class Messenger(metaclass=ABCMeta):
    @abstractmethod
    def compose_messages_from_nodes_to_targets(self, graph: np.array, node_features: np.array, edge_features: np.array,
                                               messages: np.array) -> np.array:
        pass
