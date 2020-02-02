import numpy as np

from src.domain.gru_message import GRUMessage
from src.domain.message import Message
from src.domain.messenger import Messenger
from src.domain.node import Node


class GRUMessenger(Messenger):
    def __init__(self):
        self.w_tree_update_gate_features = None
        self.w_tree_forget_gate_features = None
        self.w_tree_current_memory_message_features = None
        self.u_tree_update_gate = None
        self.u_tree_forget_gate = None
        self.u_tree_current_memory_message = None
        self.b_tree_update_gate = None
        self.b_tree_forget_gate = None
        self.b_tree_current_memory_message = None

    def compose_messages_from_nodes_to_targets(self, graph: np.array, node_features: np.array, edge_features: np.array,
                                               messages: np.array) -> np.array:
        number_of_graph_nodes = node_features.shape[0]
        new_messages = np.zeros_like(messages)
        for node_id in range(number_of_graph_nodes):
            current_node = self._create_node(graph, node_features, node_id)
            for target_node_index in range(current_node.neighbors_count):
                current_node.set_target(target_node_index)
                node_slice = current_node.get_slice_to_target()
                message = self._get_message_inputs(messages, current_node, node_features, target_node_index)
                message.compose()
                new_messages[node_slice] = message.value
        return new_messages

    def _get_message_inputs(self, messages: np.ndarray, current_node: Node, node_features: np.ndarray,
                            target_node_index: int) -> Message:
        message = self._create_message()
        message.previous_messages = self._get_messages_from_all_node_neighbors_except_target_summed(messages,
                                                                                                    current_node,
                                                                                                    target_node_index).value
        message.update_gate = self._pass_through_update_gate(messages, current_node, node_features, target_node_index)
        message.current_memory = self._get_current_memory_message(messages, current_node, node_features,
                                                                  target_node_index)
        return message

    def _get_messages_from_all_node_neighbors_except_target_summed(self,
                                                                   messages: np.ndarray,
                                                                   current_node: Node,
                                                                   target_node_index: int) -> Message:
        messages_from_the_other_neighbors = self._create_message()
        messages_from_the_other_neighbors.value = np.zeros(current_node.features.shape[0])
        if current_node.neighbors_count > 1:
            neighbors_slice = current_node.get_slice_to_neighbors_without_current_target(target_node_index)
            messages_from_the_other_neighbors.value = np.sum(messages[neighbors_slice], axis=0)
        return messages_from_the_other_neighbors

    def _pass_through_update_gate(self, messages: np.array, current_node: Node, node_features: np.array,
                                  target_node_index: int) -> np.array:
        message_from_a_neighbor_other_than_target = self._get_messages_from_all_node_neighbors_except_target_summed(
            messages,
            current_node,
            target_node_index)
        node_slice = current_node.get_slice_to_target()
        update_gate_output = self._sigmoid(
            self.w_tree_update_gate_features[node_slice].dot(node_features[current_node.node_id]) +
            self.u_tree_update_gate[node_slice].dot(message_from_a_neighbor_other_than_target.value) +
            self.b_tree_update_gate)
        return update_gate_output

    def _get_current_memory_message(self, messages: np.ndarray, current_node: Node, node_features: np.ndarray,
                                    target_node_index: int) -> np.ndarray:
        messages_passed_through_reset_gate = self._keep_or_reset_messages(messages, current_node, target_node_index,
                                                                          node_features)
        node_slice = current_node.get_slice_to_target()
        current_memory_message = self.w_tree_current_memory_message_features[node_slice].dot(
            node_features[current_node.node_id]) + \
                                 self.u_tree_current_memory_message[node_slice].dot(
                                     messages_passed_through_reset_gate) + \
                                 self.b_tree_current_memory_message
        return self._tanh(current_memory_message)

    def _keep_or_reset_messages(self, messages: np.ndarray, current_node: Node, target_node_index: int,
                                node_features: np.ndarray) -> np.ndarray:
        messages_from_the_other_neighbors_summed = self._create_message()
        messages_from_the_other_neighbors_summed.value = np.zeros(current_node.features.shape[0])
        neighbors_slice = current_node.get_slice_to_neighbors_without_current_target(target_node_index)[0]
        node_slice = (current_node.node_id, target_node_index)
        for reset_node_index in neighbors_slice:
            reset_node_slice = (current_node.node_id, reset_node_index)
            reset_gate_output = self._pass_through_reset_gate(messages, current_node, node_features, reset_node_index)
            messages_from_the_other_neighbors_summed.value += reset_gate_output * messages[reset_node_slice]
        return self.u_tree_current_memory_message[node_slice].dot(messages_from_the_other_neighbors_summed.value)

    def _pass_through_reset_gate(self, messages: np.array, current_node: Node, node_features: np.array,
                                 reset_node_index: int) -> np.array:
        node_slice = (current_node.node_id, reset_node_index)
        message_from_a_neighbor_other_than_target = messages[node_slice]
        reset_gate_output = self._sigmoid(
            self.w_tree_update_gate_features[node_slice].dot(node_features[current_node.node_id]) +
            self.u_tree_update_gate[node_slice].dot(message_from_a_neighbor_other_than_target) +
            self.b_tree_update_gate)
        return reset_gate_output

    @staticmethod
    def _create_message() -> Message:
        return GRUMessage()

    @staticmethod
    def _sigmoid(vector: np.ndarray) -> np.ndarray:
        return np.exp(vector) / (np.exp(vector) + 1)

    @staticmethod
    def _tanh(vector: np.ndarray) -> np.ndarray:
        return np.tanh(vector)

    @staticmethod
    def _create_node(graph: np.ndarray, node_features: np.ndarray, start_node: int) -> Node:
        return Node(graph, node_features, start_node)
