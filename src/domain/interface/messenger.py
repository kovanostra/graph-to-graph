from abc import ABCMeta, abstractmethod

import numpy as np

from src.domain.graph import Graph


class Messenger(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def compose_messages_from_nodes_to_targets(self, graph: Graph, messages: np.array) -> np.array:
        pass

    @abstractmethod
    def initialize(self, graph: Graph, weight: float) -> None:
        pass
