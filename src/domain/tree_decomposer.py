import numpy as np
import rdkit
from rdkit import Chem

from src.domain.graph import Graph


class TreeDecomposer:
    def __init__(self) -> None:
        pass

    def decompose(self, molecule: rdkit.Chem.rdchem.Mol) -> Graph:
        bonds_not_in_rings = self.extract_all_bonds_not_in_molecular_rings(molecule)
        rings = self._get_smallest_set_of_smallest_rings(molecule)
        junction_tree = self._create_junction_tree(bonds_not_in_rings, rings)
        return junction_tree

    @staticmethod
    def _get_smallest_set_of_smallest_rings(molecule: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.GetSymmSSSR:
        rings = []
        for ring in Chem.GetSymmSSSR(molecule):
            rings.append(list(ring))
        return rings

    @staticmethod
    def extract_all_bonds_not_in_molecular_rings(molecule: rdkit.Chem.rdchem.Mol) -> list:
        bonds_not_in_molecular_rings = []
        for bond in molecule.GetBonds():
            if not bond.IsInRing():
                bonds_not_in_molecular_rings.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        return bonds_not_in_molecular_rings

    def _create_junction_tree(self, bonds_not_in_rings: list, rings: rdkit.Chem.GetSymmSSSR) -> Graph:
        all_clusters = sorted(bonds_not_in_rings + rings)
        tree_nodes_count = len(all_clusters)
        adjacency_matrix = np.zeros((tree_nodes_count, tree_nodes_count))
        for current_cluster_index in range(tree_nodes_count - 1):
            current_cluster = all_clusters[current_cluster_index]
            for next_cluster_index in range(current_cluster_index, tree_nodes_count):
                next_cluster = all_clusters[next_cluster_index]
                if current_cluster[-1] == next_cluster[0]:
                    adjacency_matrix[current_cluster_index, next_cluster_index] = 1
                    adjacency_matrix[next_cluster_index, current_cluster_index] = 1

        return self._create_graph(adjacency_matrix)

    @staticmethod
    def _create_graph(adjacency_matrix: np.array) -> Graph:
        return Graph(adjacency_matrix, np.array([[]]), np.array([[]]))
