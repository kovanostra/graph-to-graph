import numpy as np

from domain.interface.message import Message
from domain.interface.messenger import Messenger
from src.domain.Graph import Graph
from src.domain.edge import Edge
from src.domain.node import Node
from src.domain.rnn_message import RNNMessage


class RNNMessenger(Messenger):
    def __init__(self):
        super().__init__()
        self.w_graph_node_features = None
        self.w_graph_edge_features = None
        self.w_graph_neighbor_messages = None

    def compose_messages_from_nodes_to_targets(self, graph: Graph, messages: np.ndarray) -> np.ndarray:
        new_messages = np.zeros_like(messages)
        for node_id in range(graph.number_of_nodes):
            current_node = self._create_node(graph, node_id)
            for target_node_index in range(current_node.neighbors_count):
                current_node.set_target(target_node_index)
                current_edge = self._create_edge(graph, current_node)
                node_slice = current_node.get_slice_to_target()
                message = self._get_message_inputs(messages, current_node, target_node_index, current_edge)
                message.compose()
                new_messages[node_slice] = message.value
        return new_messages

    def _get_message_inputs(self, messages: np.ndarray, current_node, target_node_index: int, current_edge) -> Message:
        message = self._create_message()
        node_slice = current_node.get_slice_to_target()
        message.node_input = self.w_graph_node_features[node_slice].dot(current_node.features)
        message.edge_input = self.w_graph_edge_features[node_slice].dot(current_edge.features)
        messages_from_the_other_neighbors_summed = self._get_messages_from_all_node_neighbors_except_target(messages,
                                                                                                            current_node,
                                                                                                            target_node_index)
        message.neighbors_input = self.w_graph_neighbor_messages[node_slice].dot(
            messages_from_the_other_neighbors_summed.value)
        return message

    def _get_messages_from_all_node_neighbors_except_target(self,
                                                            messages: np.ndarray,
                                                            current_node: Node,
                                                            target_node_index: int) -> Message:
        messages_from_the_other_neighbors_summed = self._create_message()
        messages_from_the_other_neighbors_summed.value = np.zeros(current_node.features.shape[0])
        if current_node.neighbors_count > 1:
            neighbors_slice = current_node.get_slice_to_neighbors_without_current_target(target_node_index)
            messages_from_the_other_neighbors_summed.value = self.w_graph_neighbor_messages[neighbors_slice][0].dot(
                messages[neighbors_slice][0])
        return messages_from_the_other_neighbors_summed

    @staticmethod
    def _create_node(graph: Graph, node_id: int) -> Node:
        return Node(graph, node_id)

    @staticmethod
    def _create_edge(graph: Graph, current_node: Node) -> Edge:
        return Edge(current_node.node_id, current_node.current_target, graph)

    @staticmethod
    def _create_message() -> Message:
        return RNNMessage()
