import numpy as np

from src.domain.rnn_messenger import RNNMessenger


class GraphEncoder:
    def __init__(self):
        self.u_graph_node_features = None
        self.u_graph_neighbor_messages = None

    def encode_graph(self, graph: np.ndarray, node_features: np.ndarray, edge_features: np.ndarray, time_steps: int) -> np.ndarray:
        messages = self._send_messages(graph, node_features, edge_features, time_steps)
        encoded_graph = self._get_encoded_graph(messages, node_features)
        return encoded_graph

    def _get_encoded_graph(self, messages: np.ndarray, node_features: np.ndarray) -> np.ndarray:
        encoded_graph = np.zeros(node_features.shape)
        for start_node in range(node_features.shape[0]):
            encoded_graph[start_node] += self._encode_node(messages, node_features, start_node)
        return encoded_graph

    def _encode_node(self, messages: np.ndarray, node_features: np.ndarray, start_node: int) -> np.ndarray:
        node_encoding_features = self.u_graph_node_features[start_node].dot(node_features[start_node])
        node_encoding_messages = self.u_graph_neighbor_messages[start_node].dot(np.sum(messages[start_node], axis=0))
        return self._relu(node_encoding_features + node_encoding_messages)

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

    @staticmethod
    def _relu(vector: np.ndarray) -> np.ndarray:
        return np.maximum(0, vector)
