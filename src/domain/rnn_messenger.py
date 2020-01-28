import numpy as np

from src.domain.rnn_message import RNNMessage
from src.domain.edge import Edge
from src.domain.node import Node


class RNNMessenger:
    def __init__(self):
        self.w_graph_node_features = None
        self.w_graph_edge_features = None
        self.w_graph_neighbor_messages = None

    def compose_messages_from_nodes_to_targets(self,
                                               graph: np.ndarray,
                                               node_features: np.ndarray,
                                               edge_features: np.ndarray,
                                               messages: np.ndarray) -> np.ndarray:
        number_of_graph_nodes = node_features.shape[0]
        for node_id in range(number_of_graph_nodes):
            current_node = self._create_node(graph, node_features, node_id)
            for target_node_index in range(current_node.neighbors_count):
                current_node.set_target(target_node_index)
                current_edge = self._create_edge(edge_features, current_node)
                node_slice = current_node.get_slice_to_target()
                message = self._get_message_inputs(messages, current_node, target_node_index, current_edge)
                message.compose()
                messages[node_slice] = message.value
        return messages

    def _get_message_inputs(self, messages: np.ndarray, current_node, target_node_index: int, current_edge):
        message = self._create_message()
        node_slice = current_node.get_slice_to_target()
        message.node_input = self.w_graph_node_features[node_slice].dot(current_node.features)
        message.edge_input = self.w_graph_edge_features[node_slice].dot(current_edge.features)
        messages_from_the_other_neighbors = self._get_messages_from_all_node_neighbors_except_target(messages,
                                                                                                     current_node,
                                                                                                     target_node_index)
        message.neighbors_input = self.w_graph_neighbor_messages[node_slice].dot(
            messages_from_the_other_neighbors.value)
        return message

    def _get_messages_from_all_node_neighbors_except_target(self,
                                                            messages: np.ndarray,
                                                            current_node,
                                                            target_node_index: int):
        messages_from_the_other_neighbors = self._create_message()
        messages_from_the_other_neighbors.value = np.zeros(current_node.features.shape[0])
        if current_node.neighbors_count > 1:
            neighbors_slice = current_node.get_slice_to_neighbors_without_current_target(target_node_index)
            messages_from_the_other_neighbors.value = self.w_graph_neighbor_messages[neighbors_slice][0].dot(
                messages[neighbors_slice][0])
        return messages_from_the_other_neighbors

    def _create_node(self, graph: np.ndarray, node_features: np.ndarray, start_node: int):
        return Node(graph, node_features, start_node)

    @staticmethod
    def _create_edge(edge_features: np.ndarray, current_node):
        return Edge(current_node.node_id, current_node.current_target, edge_features)

    @staticmethod
    def _create_message():
        return RNNMessage()
