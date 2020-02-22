from typing import Tuple

import numpy as np
import rdkit
from rdkit import Chem

from domain.message_passing.messenger_gru import MessengerGRU
from domain.message_passing.messenger_rnn import MessengerRNN
from src.domain.graph import Graph
from src.domain.graph_encoder import GraphEncoder
from src.domain.interface.messenger import Messenger
from src.domain.tree_decomposer import TreeDecomposer


class GraphToGraph:
    def __init__(self, messenger_for_graph_encoder: Messenger,
                 messenger_for_tree_encoder: Messenger,
                 initialization_weight: float,
                 time_steps: int):
        self.messenger_for_graph_encoder = messenger_for_graph_encoder
        self.graph_encoder = GraphEncoder(self.messenger_for_graph_encoder, time_steps=time_steps)
        self.encoded_graph = None
        self.messenger_for_tree_encoder = messenger_for_tree_encoder
        self.tree_encoder = GraphEncoder(self.messenger_for_tree_encoder, time_steps=time_steps)
        self.encoded_tree = None
        self.matrices_initialization_weight = initialization_weight
        self.tree_decomposer = TreeDecomposer()

    def fit(self, molecules_in_smiles: Tuple[str]) -> None:
        for molecule_in_smiles in molecules_in_smiles:
            molecule = Chem.MolFromSmiles(molecule_in_smiles)
            molecular_graph = self._initialize_molecular_graph(molecule)
            junction_tree = self.tree_decomposer.decompose(molecule)
            junction_tree = self._set_node_features(junction_tree)
            self._initialize_model_weights(molecular_graph, junction_tree)
            self.encoded_graph = self.graph_encoder.encode(molecular_graph)
            self.encoded_tree = self.tree_encoder.encode(junction_tree)

    def _initialize_molecular_graph(self, molecule: rdkit.Chem.rdchem.Mol) -> Graph:
        adjacency_matrix = Chem.GetAdjacencyMatrix(molecule)
        number_of_features = 2
        features_shape = (adjacency_matrix.shape[0], number_of_features)
        node_features = self._initialize_node_features_matrix(weight=self.matrices_initialization_weight,
                                                              features_shape=features_shape)
        edge_features = self._initialize_edge_features_matrix(adjacency_matrix=adjacency_matrix,
                                                              weight=self.matrices_initialization_weight,
                                                              features_length=features_shape[1])
        graph = self._create_graph(adjacency_matrix, node_features, edge_features)
        return graph

    def _initialize_model_weights(self, molecular_graph: Graph, junction_tree: Graph) -> None:
        self.messenger_for_graph_encoder.initialize(graph=molecular_graph, weight=self.matrices_initialization_weight)
        self.messenger_for_tree_encoder.initialize(graph=junction_tree, weight=self.matrices_initialization_weight)
        self.graph_encoder.initialize(graph=molecular_graph, weight=self.matrices_initialization_weight)
        self.tree_encoder.initialize(graph=junction_tree, weight=self.matrices_initialization_weight)

    def _set_node_features(self, junction_tree: Graph) -> Graph:
        number_of_features = 2
        features_shape = (junction_tree.adjacency_matrix.shape[0], number_of_features)
        junction_tree.node_features = self._initialize_node_features_matrix(weight=self.matrices_initialization_weight,
                                                                            features_shape=features_shape)
        junction_tree.number_of_node_features = number_of_features
        return junction_tree

    @staticmethod
    def _initialize_node_features_matrix(weight: float, features_shape: tuple) -> np.array:
        return np.array(weight * np.random.random(features_shape))

    @staticmethod
    def _initialize_edge_features_matrix(adjacency_matrix: np.ndarray,
                                         weight: float,
                                         features_length: int) -> np.ndarray:
        return np.array([[row[column_index] * weight * np.random.random(features_length) for column_index in
                          range(adjacency_matrix.shape[1])] for row in adjacency_matrix])

    @staticmethod
    def _create_graph(adjacency_matrix: np.ndarray, node_features: np.ndarray, edge_features: np.ndarray) -> Graph:
        return Graph(adjacency_matrix, node_features, edge_features)


if __name__ == "__main__":
    message_passing_time_steps = 10
    matrices_initialization_weight = 0.1
    messenger_rnn = MessengerRNN()
    messenger_gru = MessengerGRU()
    graph_to_graph = GraphToGraph(messenger_for_graph_encoder=messenger_rnn,
                                  messenger_for_tree_encoder=messenger_gru,
                                  initialization_weight=matrices_initialization_weight,
                                  time_steps=message_passing_time_steps)
    molecules = ('C12C3C(C(C1)CCC2)(C4CC3CC4)C(C)CC',)
    graph_to_graph.fit(molecules)
    print(graph_to_graph.encoded_graph, "\n\n")
    print(graph_to_graph.encoded_tree, "\n\n")
