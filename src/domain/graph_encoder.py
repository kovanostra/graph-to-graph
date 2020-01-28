import numpy as np

from src.domain.rnn_messenger import RNNMessenger


class GraphEncoder:
    def __init__(self):
        self.u_graph_node_features = None
        self.u_graph_neighbor_messages = None

    def encode_graph(self, graph: np.ndarray, node_features: np.ndarray, edge_features: np.ndarray,
                     time_steps: int) -> np.ndarray:
        messages = self._send_messages(graph, node_features, edge_features, time_steps)
        encoded_graph = self._encode_nodes(messages, node_features)
        return encoded_graph

    def _send_messages(self, graph: np.ndarray, node_features: np.ndarray, edge_features: np.ndarray,
                       time_steps: int) -> np.ndarray:
        number_of_graph_nodes = self.u_graph_node_features.shape[0]
        messages = np.zeros((number_of_graph_nodes,
                             number_of_graph_nodes,
                             self.u_graph_node_features.shape[2]))
        messenger = RNNMessenger()
        for step in range(time_steps):
            messages = messenger.compose_messages_from_nodes_to_targets(graph, node_features, edge_features, messages)
        return messages

    def _encode_nodes(self, messages: np.ndarray, node_features: np.ndarray) -> np.ndarray:
        encoded_graph = np.zeros(node_features.shape)
        for node_id in range(node_features.shape[0]):
            encoded_graph[node_id] += self._apply_recurrent_layer_for_node(messages, node_features, node_id)
        return encoded_graph

    def _apply_recurrent_layer_for_node(self, messages: np.ndarray, node_features: np.ndarray,
                                        node_id: int) -> np.ndarray:
        node_encoding_features = self.u_graph_node_features[node_id].dot(node_features[node_id])
        node_encoding_messages = self.u_graph_neighbor_messages[node_id].dot(np.sum(messages[node_id], axis=0))
        return self._relu(node_encoding_features + node_encoding_messages)

    @staticmethod
    def _relu(vector: np.ndarray) -> np.ndarray:
        return np.maximum(0, vector)
