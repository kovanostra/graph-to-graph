import numpy as np
import rdkit
from rdkit import Chem

from src.domain.graph import Graph


class TreeDecomposer:
    def __init__(self) -> None:
        pass

    def decompose(self, molecule: rdkit.Chem.rdchem.Mol) -> Graph:
        bonds_not_in_rings = self.extract_all_bonds_not_in_molecular_rings(molecule)
        rings = self._get_smallest_set_of_smallest_rings_merged_if_they_share_more_than_two_atoms(molecule)
        junction_tree = self._create_junction_tree(bonds_not_in_rings, rings)
        return junction_tree

    def _get_smallest_set_of_smallest_rings_merged_if_they_share_more_than_two_atoms(self,
                                                                                     molecule: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.GetSymmSSSR:
        rings = []
        for ring in Chem.GetSymmSSSR(molecule):
            rings.append(list(ring))
        if len(rings) > 1:
            rings = self._merge_rings_sharing_more_than_two_atoms(sorted(rings))
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
        print(all_clusters)
        tree_nodes_count = len(all_clusters)
        adjacency_matrix = np.zeros((tree_nodes_count, tree_nodes_count))
        for current_cluster_index in range(tree_nodes_count - 1):
            current_cluster = all_clusters[current_cluster_index]
            for next_cluster_index in range(current_cluster_index, tree_nodes_count):
                next_cluster = all_clusters[next_cluster_index]
                if next_cluster_index != current_cluster_index:
                    if len(current_cluster) == 2 and (current_cluster[-1] in next_cluster
                                                      or current_cluster[0] in next_cluster) \
                            or self._clusters_share_two_atoms(current_cluster, next_cluster):
                        adjacency_matrix[current_cluster_index, next_cluster_index] = 1
                        adjacency_matrix[next_cluster_index, current_cluster_index] = 1

        return self._create_graph(adjacency_matrix)

    @staticmethod
    def _create_graph(adjacency_matrix: np.array) -> Graph:
        return Graph(adjacency_matrix, np.array([[]]), np.array([[]]))

    def _merge_rings_sharing_more_than_two_atoms(self, rings: list) -> list:
        merged_rings = []
        ring_index = 0
        while ring_index < len(rings) - 1:
            current_ring = rings[ring_index]
            next_ring = rings[ring_index + 1]
            if self._rings_share_more_than_two_atoms(current_ring, next_ring):
                merged_rings.append(self._merge(current_ring, next_ring))
                ring_index += 2
            else:
                merged_rings.append(current_ring)
                ring_index += 1
        if ring_index == 1:
            merged_rings.append(next_ring)
        return merged_rings

    @staticmethod
    def _merge(current_ring: list, next_ring: list) -> list:
        all_common_elements_in_rings = set(next_ring) - set(current_ring)
        merged_ring = current_ring + list(all_common_elements_in_rings)
        return merged_ring

    @staticmethod
    def _rings_share_more_than_two_atoms(current_ring: list, next_ring: list) -> bool:
        return len(set(current_ring) & set(next_ring)) > 2

    @staticmethod
    def _clusters_share_two_atoms(current_cluster: list, next_cluster: list) -> bool:
        return len(current_cluster) > 2 and len(next_cluster) > 2 and len(set(current_cluster) & set(next_cluster)) == 2
