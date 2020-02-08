import numpy as np

from src.domain.edge import Edge
from src.domain.graph import Graph
from src.domain.interface.message import Message
from src.domain.interface.messenger import Messenger
from src.domain.message_rnn import MessageRNN
from src.domain.node import Node


class MessengerRNN(Messenger):
    def __init__(self):
        super().__init__()
        self.w_graph_node_features = None
        self.w_graph_edge_features = None
        self.w_graph_neighbor_messages = None

    def compose_messages_from_nodes_to_targets(self, graph: Graph, messages: np.ndarray) -> np.ndarray:
        new_messages = np.zeros_like(messages)
        for node_id in range(graph.number_of_nodes):
            node = self._create_node(graph, node_id)
            for end_node_id in node.neighbors:
                end_node = self._create_node(graph, end_node_id)
                edge = self._create_edge(graph, node, end_node)
                edge_slice = edge.get_edge_slice()
                message = self._get_message_inputs(messages, node, edge)
                message.compose()
                new_messages[edge_slice] = message.value
        return new_messages

    def _get_message_inputs(self, messages: np.ndarray, node: Node, edge: Edge) -> Message:
        message = self._create_message()
        edge_slice = edge.get_edge_slice()
        message.node_input = self.w_graph_node_features[edge_slice].dot(node.features)
        message.edge_input = self.w_graph_edge_features[edge_slice].dot(edge.features)
        messages_from_the_other_neighbors_summed = self._get_messages_from_all_node_neighbors_except_target(messages,
                                                                                                            node,
                                                                                                            edge)
        message.neighbors_input = self.w_graph_neighbor_messages[edge_slice].dot(
            messages_from_the_other_neighbors_summed.value)
        return message

    def _get_messages_from_all_node_neighbors_except_target(self,
                                                            messages: np.ndarray,
                                                            node: Node,
                                                            edge: Edge) -> Message:
        messages_from_the_other_neighbors_summed = self._create_message()
        messages_from_the_other_neighbors_summed.value = np.zeros(node.features.shape[0])
        if node.neighbors_count > 1:
            neighbors_slice = edge.get_start_node_neighbors_without_end_node()
            messages_from_the_other_neighbors_summed.value = self.w_graph_neighbor_messages[neighbors_slice][0].dot(
                messages[neighbors_slice][0])
        return messages_from_the_other_neighbors_summed

    @staticmethod
    def _create_node(graph: Graph, node_id: int) -> Node:
        return Node(graph, node_id)

    @staticmethod
    def _create_edge(graph: Graph, start_node: Node, end_node: Node) -> Edge:
        return Edge(graph, start_node, end_node)

    @staticmethod
    def _create_message() -> Message:
        return MessageRNN()
