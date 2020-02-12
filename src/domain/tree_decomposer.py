import itertools
import numpy as np
import rdkit
from rdkit import Chem

from src.domain.graph import Graph


class TreeDecomposer:
    def __init__(self) -> None:
        pass

    def decompose(self, molecule: rdkit.Chem.rdchem.Mol) -> Graph:
        bonds_not_in_rings = self._extract_all_bonds_not_in_molecular_rings(molecule)
        rings_in_molecular_graph = self._get_smallest_set_of_smallest_rings_merged_if_they_share_more_than_two_atoms(
            molecule)
        all_clusters = sorted(bonds_not_in_rings + rings_in_molecular_graph)
        rings_in_cluster_graph = self._get_rings_in_cluster_graph(all_clusters, molecule)
        rings_to_break = self._break_rings_in_cluster_graph(all_clusters, rings_in_cluster_graph)
        junction_tree = self._create_junction_tree(all_clusters, rings_to_break)
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
    def _extract_all_bonds_not_in_molecular_rings(molecule: rdkit.Chem.rdchem.Mol) -> list:
        bonds_not_in_molecular_rings = []
        for bond in molecule.GetBonds():
            if not bond.IsInRing():
                bonds_not_in_molecular_rings.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        return bonds_not_in_molecular_rings

    def _create_junction_tree(self, all_clusters: list, rings_to_break: list) -> Graph:
        print(all_clusters)
        print(rings_to_break)
        tree_nodes_count = len(all_clusters)
        adjacency_matrix = np.zeros((tree_nodes_count, tree_nodes_count))
        for current_cluster_index in range(tree_nodes_count - 1):
            current_cluster = all_clusters[current_cluster_index]
            for next_cluster_index in range(current_cluster_index, tree_nodes_count):
                next_cluster = all_clusters[next_cluster_index]
                if next_cluster_index != current_cluster_index:
                    if (len(current_cluster) == 2 and (current_cluster[-1] in next_cluster
                                                       or current_cluster[0] in next_cluster)
                        or self._clusters_share_two_atoms(current_cluster, next_cluster)) and \
                            ([current_cluster_index, next_cluster_index] not in rings_to_break and
                             [next_cluster_index, current_cluster_index] not in rings_to_break):
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
    def _number_of_shared_atoms(current_ring: list, next_ring: list) -> int:
        return len(set(current_ring) & set(next_ring))

    @staticmethod
    def _clusters_share_two_atoms(current_cluster: list, next_cluster: list) -> bool:
        return len(current_cluster) > 2 and len(next_cluster) > 2 and len(set(current_cluster) & set(next_cluster)) == 2

    @staticmethod
    def _get_rings_in_cluster_graph(all_clusters: list,
                                    molecule: rdkit.Chem.rdchem.Mol) -> list:
        rings = []
        for atom in range(len(Chem.GetAdjacencyMatrix(molecule))):
            ring = []
            for cluster_index in range(len(all_clusters)):
                cluster = all_clusters[cluster_index]
                if atom in cluster:
                    ring.append(cluster_index)
            if len(ring) > 2:
                rings.append(ring)
        return rings

    def _break_rings_in_cluster_graph(self, all_clusters: list, rings_in_cluster_graph: list) -> list:
        rings_to_break = []
        for ring in rings_in_cluster_graph:
            ring_members = np.array(all_clusters)[ring]
            max_number_of_shared_atoms = -1
            min_number_of_shared_atoms = 10
            edges_to_break = []
            for pair in itertools.combinations(ring_members, 2):
                shared_atoms_of_pair = self._number_of_shared_atoms(pair[0], pair[1])
                pair_with_shared_atoms = list(pair)
                pair_with_shared_atoms.append([shared_atoms_of_pair])
                edges_to_break.append(pair_with_shared_atoms)
                if shared_atoms_of_pair > max_number_of_shared_atoms:
                    max_number_of_shared_atoms = shared_atoms_of_pair
                if shared_atoms_of_pair < min_number_of_shared_atoms:
                    min_number_of_shared_atoms = shared_atoms_of_pair
            edges_to_break = sorted(edges_to_break, key=lambda x: x[2], reverse=True)
            if max_number_of_shared_atoms - min_number_of_shared_atoms > 0:
                pair_cluster_index = [index for index in range(len(all_clusters)) if
                                      all_clusters[index] == edges_to_break[-1][0] or all_clusters[index] ==
                                      edges_to_break[-1][1]]
                rings_to_break.append(pair_cluster_index)
        return rings_to_break
