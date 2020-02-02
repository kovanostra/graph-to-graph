import numpy as np

from src.domain.gru_message import GRUMessage
from src.domain.message import Message
from src.domain.messenger import Messenger
from src.domain.node import Node


class GRUMessenger(Messenger):
    def __init__(self):
        self.w_tree_update_gate_features = None
        self.w_tree_forget_gate_features = None
        self.w_tree_current_message_features = None
        self.u_tree_update_gate = None
        self.u_tree_forget_gate = None
        self.u_tree_current_message = None
        self.b_tree_update_gate = None
        self.b_tree_forget_gate = None
        self.b_tree_current_message = None

    def compose_messages_from_nodes_to_targets(self, graph: np.array, node_features: np.array, edge_features: np.array,
                                               messages: np.array) -> np.array:
        pass

    def _get_messages_from_all_node_neighbors_except_target(self,
                                                            messages: np.ndarray,
                                                            current_node: Node,
                                                            target_node_index: int) -> Message:
        messages_from_the_other_neighbors = self._create_message()
        messages_from_the_other_neighbors.value = np.zeros(current_node.features.shape[0])
        if current_node.neighbors_count > 1:
            neighbors_slice = current_node.get_slice_to_neighbors_without_current_target(target_node_index)
            messages_from_the_other_neighbors.value = np.sum(messages[neighbors_slice], axis=0)
        return messages_from_the_other_neighbors

    @staticmethod
    def _create_message() -> Message:
        return GRUMessage()
