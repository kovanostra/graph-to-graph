from typing import List

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
        all_clusters = self._detect_and_break_rings_in_cluster_graph(all_clusters, molecule)
        junction_tree = self._create_junction_tree(all_clusters)
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

    def _create_junction_tree(self, all_clusters: list) -> Graph:
        tree_nodes_count = len(all_clusters)
        adjacency_matrix = np.zeros((tree_nodes_count, tree_nodes_count))
        for current_cluster_index in range(tree_nodes_count - 1):
            current_cluster = all_clusters[current_cluster_index]
            for next_cluster_index in range(current_cluster_index, tree_nodes_count):
                next_cluster = all_clusters[next_cluster_index]
                if self._clusters_share_at_least_one_atom(current_cluster, next_cluster):
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
    def _clusters_share_at_least_one_atom(current_cluster: list, next_cluster: list) -> bool:
        return current_cluster != next_cluster and len(set(current_cluster) & set(next_cluster)) >= 1

    def _detect_and_break_rings_in_cluster_graph(self,
                                                 all_clusters: List[list],
                                                 molecule: rdkit.Chem.rdchem.Mol) -> list:
        atoms_in_molecule = len(Chem.GetAdjacencyMatrix(molecule))
        new_nodes = []
        for atom in range(len(Chem.GetAdjacencyMatrix(molecule))):
            ring_candidate = []
            for cluster_index in range(len(all_clusters)):
                cluster = all_clusters[cluster_index]
                if atom in cluster:
                    ring_candidate.append(cluster_index)
                    if len(ring_candidate) > 2:
                        new_nodes, atoms_in_molecule = self._intersect_node(all_clusters,
                                                                            atom,
                                                                            atoms_in_molecule,
                                                                            new_nodes,
                                                                            ring_candidate)
                        for current_cluster_index, next_cluster_index in itertools.combinations(ring_candidate, 2):
                            if self._clusters_share_at_least_one_atom(all_clusters[current_cluster_index],
                                                                      all_clusters[next_cluster_index]):
                                shared_atoms = self._get_shared_atoms(all_clusters[current_cluster_index],
                                                                      all_clusters[next_cluster_index])
                                all_clusters = self._remove_shared_atoms(all_clusters, current_cluster_index,
                                                                         next_cluster_index, shared_atoms)
        all_clusters.extend(new_nodes)
        return sorted(all_clusters)

    def _intersect_node(self, all_clusters, atom, atoms_in_molecule, new_nodes, ring_candidate):
        new_atoms_in_molecule = atoms_in_molecule + len(ring_candidate)
        intersecting_node = self._create_node_with_new_atoms(atoms_in_molecule, new_atoms_in_molecule)
        for ring_member_index in range(len(intersecting_node)):
            node_to_modify = ring_candidate[ring_member_index]
            all_clusters[node_to_modify].remove(atom)
            all_clusters[node_to_modify].append(intersecting_node[ring_member_index])
        new_nodes.append(intersecting_node)
        atoms_in_molecule = new_atoms_in_molecule
        return new_nodes, atoms_in_molecule

    @staticmethod
    def _create_node_with_new_atoms(atoms_in_molecule, new_atoms_in_molecule):
        return [dummy_atom for dummy_atom in
                range(atoms_in_molecule, new_atoms_in_molecule)]

    @staticmethod
    def _get_shared_atoms(current_cluster: list, next_cluster: list) -> list:
        return [atom for atom in current_cluster if atom in next_cluster]

    @staticmethod
    def _remove_shared_atoms(all_clusters: list, current_cluster_index: int, next_cluster_index: int,
                             shared_atoms: list) -> list:
        for atom in shared_atoms:
            all_clusters[current_cluster_index].remove(atom)
            all_clusters[next_cluster_index].remove(atom)
        return all_clusters
