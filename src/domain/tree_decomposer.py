from typing import Tuple

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
        self.cluster_graph = None

    def decompose(self, molecule: rdkit.Chem.rdchem.Mol) -> Graph:
        self._set_molecular_properties(molecule)
        self._extract_all_bonds_not_in_molecular_rings()
        self._merge_cluster_rings_sharing_more_than_two_atoms()
        self._create_cluster_graph()
        self._get_cluster_graph_without_rings()
        return self._create_junction_tree()

    def _set_molecular_properties(self, molecule: rdkit.Chem.rdchem.Mol) -> None:
        self.molecule = molecule
        self.bonds = molecule.GetBonds()
        self.rings = [list(ring) for ring in Chem.GetSymmSSSR(self.molecule)]
        self.atoms_in_molecule = len(Chem.GetAdjacencyMatrix(self.molecule))

    def _extract_all_bonds_not_in_molecular_rings(self) -> None:
        bonds_not_in_molecular_rings = []
        for bond in self.bonds:
            if not bond.IsInRing():
                bonds_not_in_molecular_rings.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        self.bonds = bonds_not_in_molecular_rings

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

    def _create_cluster_graph(self):
        self.cluster_graph = sorted(self.bonds + self.rings)

    def _get_cluster_graph_without_rings(self) -> None:
        atoms_in_molecule = self.atoms_in_molecule
        new_nodes = []
        for atom in range(self.atoms_in_molecule):
            ring_candidate = []
            for cluster_index in range(len(self.cluster_graph)):
                cluster = self.cluster_graph[cluster_index]
                if atom in cluster:
                    ring_candidate.append(cluster_index)
                    new_nodes, atoms_in_molecule = self._break_rings_in_cluster_graph(atom,
                                                                                      atoms_in_molecule,
                                                                                      new_nodes,
                                                                                      ring_candidate)
        self.cluster_graph.extend(new_nodes)
        self.cluster_graph = sorted(self.cluster_graph)

    def _create_junction_tree(self) -> Graph:
        tree_nodes_count = len(self.cluster_graph)
        adjacency_matrix = np.zeros((tree_nodes_count, tree_nodes_count))
        for current_cluster_index in range(tree_nodes_count - 1):
            for next_cluster_index in range(current_cluster_index, tree_nodes_count):
                if self._clusters_share_at_least_one_atom(current_cluster_index, next_cluster_index):
                    adjacency_matrix[current_cluster_index, next_cluster_index] = 1
                    adjacency_matrix[next_cluster_index, current_cluster_index] = 1

        return self._create_graph(adjacency_matrix)

    def _clusters_share_at_least_one_atom(self, current_cluster_index: int, next_cluster_index: int) -> bool:
        current_cluster, next_cluster = self.cluster_graph[current_cluster_index], self.cluster_graph[
            next_cluster_index]
        return current_cluster != next_cluster and len(set(current_cluster) & set(next_cluster)) >= 1

    def _break_rings_in_cluster_graph(self, atom: int, atoms_in_molecule: int, new_nodes: list,
                                      ring_candidate: list) -> Tuple[list, int]:
        if len(ring_candidate) > 2:
            new_nodes, atoms_in_molecule = self._intersect_node(atom,
                                                                atoms_in_molecule,
                                                                new_nodes,
                                                                ring_candidate)
            self._remove_remaining_bonds(ring_candidate)
        return new_nodes, atoms_in_molecule

    def _remove_remaining_bonds(self, ring_candidate: list) -> None:
        for current_cluster_index, next_cluster_index in itertools.combinations(ring_candidate, 2):
            if self._clusters_share_at_least_one_atom(current_cluster_index, next_cluster_index):
                shared_atoms = self._get_shared_atoms(current_cluster_index, next_cluster_index)
                self._remove_shared_atoms(current_cluster_index, next_cluster_index, shared_atoms)

    def _intersect_node(self, atom: int, atoms_in_molecule: int, new_nodes: list, ring_candidate: list) \
            -> Tuple[list, int]:
        new_atoms_in_molecule = atoms_in_molecule + len(ring_candidate)
        intersecting_node = self._create_node_with_new_atoms(atoms_in_molecule, new_atoms_in_molecule)
        for ring_member_index in range(len(intersecting_node)):
            node_to_modify = ring_candidate[ring_member_index]
            self.cluster_graph[node_to_modify].remove(atom)
            self.cluster_graph[node_to_modify].append(intersecting_node[ring_member_index])
        new_nodes.append(intersecting_node)
        atoms_in_molecule = new_atoms_in_molecule
        return new_nodes, atoms_in_molecule

    def _remove_shared_atoms(self, current_cluster_index: int, next_cluster_index: int,
                             shared_atoms: list) -> None:
        for atom in shared_atoms:
            self.cluster_graph[current_cluster_index].remove(atom)
            self.cluster_graph[next_cluster_index].remove(atom)

    def _get_shared_atoms(self, current_cluster_index: int, next_cluster_index: int) -> list:
        current_cluster, next_cluster = self.cluster_graph[current_cluster_index], self.cluster_graph[
            next_cluster_index]
        return [atom for atom in current_cluster if atom in next_cluster]

    @staticmethod
    def _create_graph(adjacency_matrix: np.array) -> Graph:
        return Graph(adjacency_matrix, np.array([[]]), np.array([[]]))

    @staticmethod
    def _merge(current_ring: list, next_ring: list) -> list:
        all_common_elements_in_rings = set(next_ring) - set(current_ring)
        merged_ring = current_ring + list(all_common_elements_in_rings)
        return merged_ring

    @staticmethod
    def _rings_share_more_than_two_atoms(current_ring: list, next_ring: list) -> bool:
        return len(set(current_ring) & set(next_ring)) > 2

    @staticmethod
    def _create_node_with_new_atoms(atoms_in_molecule: int, new_atoms_in_molecule: int) -> list:
        return [dummy_atom for dummy_atom in
                range(atoms_in_molecule, new_atoms_in_molecule)]
