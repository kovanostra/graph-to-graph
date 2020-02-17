from typing import List, Tuple

import itertools
import numpy as np
import rdkit
from rdkit import Chem

from src.domain.graph import Graph


class TreeDecomposer:
    def __init__(self) -> None:
        self.molecule = None
        self.bonds = None
        self.rings = None
        self.atoms_in_molecule = None

    def set_molecule_and_properties(self, molecule: rdkit.Chem.rdchem.Mol) -> None:
        self.molecule = molecule
        self.bonds = molecule.GetBonds()
        self.rings = [list(ring) for ring in Chem.GetSymmSSSR(self.molecule)]
        self.atoms_in_molecule = len(Chem.GetAdjacencyMatrix(self.molecule))

    def decompose(self) -> Graph:
        bonds_not_in_rings = self._extract_all_bonds_not_in_molecular_rings()
        self._merge_cluster_rings_sharing_more_than_two_atoms()
        cluster_graph = self._create_cluster_graph(bonds_not_in_rings)
        cluster_graph = self._get_cluster_graph_without_rings(cluster_graph)
        junction_tree = self._create_junction_tree(cluster_graph)
        return junction_tree

    def _extract_all_bonds_not_in_molecular_rings(self) -> list:
        bonds_not_in_molecular_rings = []
        for bond in self.bonds:
            if not bond.IsInRing():
                bonds_not_in_molecular_rings.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        return bonds_not_in_molecular_rings

    def _create_cluster_graph(self, bonds_not_in_rings):
        return sorted(bonds_not_in_rings + self.rings)

    def _get_cluster_graph_without_rings(self, cluster_graph: List[list]) -> list:
        atoms_in_molecule = self.atoms_in_molecule
        new_nodes = []
        for atom in range(self.atoms_in_molecule):
            ring_candidate = []
            for cluster_index in range(len(cluster_graph)):
                cluster = cluster_graph[cluster_index]
                if atom in cluster:
                    ring_candidate.append(cluster_index)
                    cluster_graph, new_nodes, atoms_in_molecule = self._break_rings_in_cluster_graph(cluster_graph,
                                                                                                     atom,
                                                                                                     atoms_in_molecule,
                                                                                                     new_nodes,
                                                                                                     ring_candidate)
        cluster_graph.extend(new_nodes)
        return sorted(cluster_graph)

    def _create_junction_tree(self, cluster_graph: list) -> Graph:
        tree_nodes_count = len(cluster_graph)
        adjacency_matrix = np.zeros((tree_nodes_count, tree_nodes_count))
        for current_cluster_index in range(tree_nodes_count - 1):
            current_cluster = cluster_graph[current_cluster_index]
            for next_cluster_index in range(current_cluster_index, tree_nodes_count):
                next_cluster = cluster_graph[next_cluster_index]
                if self._clusters_share_at_least_one_atom(current_cluster, next_cluster):
                    adjacency_matrix[current_cluster_index, next_cluster_index] = 1
                    adjacency_matrix[next_cluster_index, current_cluster_index] = 1

        return self._create_graph(adjacency_matrix)

    @staticmethod
    def _create_graph(adjacency_matrix: np.array) -> Graph:
        return Graph(adjacency_matrix, np.array([[]]), np.array([[]]))

    def _merge_cluster_rings_sharing_more_than_two_atoms(self) -> None:
        merged_rings = []
        ring_index = 0
        while ring_index < len(self.rings) - 1:
            current_ring, next_ring = self.rings[ring_index], self.rings[ring_index + 1]
            if self._rings_share_more_than_two_atoms(current_ring, next_ring):
                merged_rings.append(self._merge(current_ring, next_ring))
                ring_index += 2
            else:
                merged_rings.append(current_ring)
                ring_index += 1
        if ring_index == 1:
            merged_rings.append(next_ring)
        if merged_rings:
            self.rings = merged_rings

    @staticmethod
    def _merge(current_ring: list, next_ring: list) -> list:
        all_common_elements_in_rings = set(next_ring) - set(current_ring)
        merged_ring = current_ring + list(all_common_elements_in_rings)
        return merged_ring

    @staticmethod
    def _rings_share_more_than_two_atoms(current_ring: list, next_ring: list) -> bool:
        return len(set(current_ring) & set(next_ring)) > 2

    @staticmethod
    def _clusters_share_at_least_one_atom(current_cluster: list, next_cluster: list) -> bool:
        return current_cluster != next_cluster and len(set(current_cluster) & set(next_cluster)) >= 1

    def _break_rings_in_cluster_graph(self, cluster_graph: list, atom: int, atoms_in_molecule: int, new_nodes: list,
                                      ring_candidate: list) -> Tuple[list, list, int]:
        if len(ring_candidate) > 2:
            new_nodes, atoms_in_molecule = self._intersect_node(cluster_graph,
                                                                atom,
                                                                atoms_in_molecule,
                                                                new_nodes,
                                                                ring_candidate)
            cluster_graph = self._remove_remaining_bonds(cluster_graph, ring_candidate)
        return cluster_graph, new_nodes, atoms_in_molecule

    def _remove_remaining_bonds(self, cluster_graph, ring_candidate):
        for current_cluster_index, next_cluster_index in itertools.combinations(ring_candidate, 2):
            if self._clusters_share_at_least_one_atom(cluster_graph[current_cluster_index],
                                                      cluster_graph[next_cluster_index]):
                shared_atoms = self._get_shared_atoms(cluster_graph[current_cluster_index],
                                                      cluster_graph[next_cluster_index])
                cluster_graph = self._remove_shared_atoms(cluster_graph, current_cluster_index,
                                                          next_cluster_index, shared_atoms)
        return cluster_graph

    def _intersect_node(self, cluster_graph: list, atom: int, atoms_in_molecule: int, new_nodes: list,
                        ring_candidate: list) -> Tuple[list, int]:
        new_atoms_in_molecule = atoms_in_molecule + len(ring_candidate)
        intersecting_node = self._create_node_with_new_atoms(atoms_in_molecule, new_atoms_in_molecule)
        for ring_member_index in range(len(intersecting_node)):
            node_to_modify = ring_candidate[ring_member_index]
            cluster_graph[node_to_modify].remove(atom)
            cluster_graph[node_to_modify].append(intersecting_node[ring_member_index])
        new_nodes.append(intersecting_node)
        atoms_in_molecule = new_atoms_in_molecule
        return new_nodes, atoms_in_molecule

    @staticmethod
    def _create_node_with_new_atoms(atoms_in_molecule: int, new_atoms_in_molecule: int) -> list:
        return [dummy_atom for dummy_atom in
                range(atoms_in_molecule, new_atoms_in_molecule)]

    @staticmethod
    def _get_shared_atoms(current_cluster: list, next_cluster: list) -> list:
        return [atom for atom in current_cluster if atom in next_cluster]

    @staticmethod
    def _remove_shared_atoms(cluster_graph: list, current_cluster_index: int, next_cluster_index: int,
                             shared_atoms: list) -> list:
        for atom in shared_atoms:
            cluster_graph[current_cluster_index].remove(atom)
            cluster_graph[next_cluster_index].remove(atom)
        return cluster_graph
