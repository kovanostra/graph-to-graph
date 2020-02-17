import itertools
import numpy as np
import rdkit
from rdkit import Chem

from src.domain.graph import Graph


class TreeDecomposer:
    def __init__(self) -> None:
        self.molecule = None
        self.molecular_bonds = None
        self.molecular_rings = None
        self.number_of_atoms_in_molecule = None
        self.cluster_graph = None
        self.number_of_nodes_in_cluster_graph = None
        self.junction_tree = None
        self.number_of_nodes_in_junction_tree = None

    def decompose(self, molecule: rdkit.Chem.rdchem.Mol) -> Graph:
        self._set_molecular_properties(molecule)
        self._extract_molecular_bonds_not_in_molecular_rings()
        self._merge_molecular_rings_sharing_more_than_two_atoms()
        self._create_cluster_graph()
        self._create_junction_tree()
        return self._convert_junction_tree_to_graph()

    def _set_molecular_properties(self, molecule: rdkit.Chem.rdchem.Mol) -> None:
        self.molecule = molecule
        self.molecular_bonds = molecule.GetBonds()
        self.molecular_rings = [list(ring) for ring in Chem.GetSymmSSSR(self.molecule)]
        self.number_of_atoms_in_molecule = len(Chem.GetAdjacencyMatrix(self.molecule))

    def _extract_molecular_bonds_not_in_molecular_rings(self) -> None:
        self.molecular_bonds = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
                                for bond in self.molecular_bonds
                                if not bond.IsInRing()]

    def _merge_molecular_rings_sharing_more_than_two_atoms(self) -> None:
        molecular_rings_merged = []
        current_ring, next_ring = None, None
        ring_index = 0
        while ring_index < len(self.molecular_rings) - 1:
            current_ring, next_ring = self.molecular_rings[ring_index], self.molecular_rings[ring_index + 1]
            if self._rings_share_more_than_two_atoms(current_ring, next_ring):
                molecular_rings_merged.append(self._merge(current_ring, next_ring))
                ring_index += 2
            else:
                molecular_rings_merged.append(current_ring)
                ring_index += 1
        if self._molecule_contains_only_two_rings(ring_index):
            molecular_rings_merged.append(next_ring)
        if molecular_rings_merged:
            self.molecular_rings = molecular_rings_merged

    def _create_cluster_graph(self):
        self.cluster_graph = sorted(self.molecular_bonds + self.molecular_rings)
        self.number_of_nodes_in_cluster_graph = len(self.cluster_graph)

    def _create_junction_tree(self) -> None:
        new_nodes = []
        for atom in range(self.number_of_atoms_in_molecule):
            new_nodes = self._detect_and_break_rings_containing_an_atom(atom, new_nodes)
        self._update_cluster_graph_with_intersecting_nodes(new_nodes)

    def _convert_junction_tree_to_graph(self) -> Graph:
        adjacency_matrix = self._initialize_adjacency_matrix()
        for current_cluster_index in range(self.number_of_nodes_in_junction_tree - 1):
            for next_cluster_index in range(current_cluster_index, self.number_of_nodes_in_junction_tree):
                if self._clusters_share_at_least_one_atom(self.junction_tree, current_cluster_index,
                                                          next_cluster_index):
                    adjacency_matrix[current_cluster_index, next_cluster_index] = 1
                    adjacency_matrix[next_cluster_index, current_cluster_index] = 1

        return self._create_graph(adjacency_matrix)

    def _detect_and_break_rings_containing_an_atom(self, atom, new_nodes):
        ring_candidate = []
        for cluster_index in range(self.number_of_nodes_in_cluster_graph):
            if atom in self.cluster_graph[cluster_index]:
                ring_candidate.append(cluster_index)
                new_nodes = self._break_rings_in_cluster_graph(atom, new_nodes, ring_candidate)
        return new_nodes

    def _update_cluster_graph_with_intersecting_nodes(self, new_nodes):
        self.junction_tree = sorted(self.cluster_graph + new_nodes)
        self.number_of_nodes_in_junction_tree = len(self.junction_tree)

    def _initialize_adjacency_matrix(self):
        return np.zeros((self.number_of_nodes_in_junction_tree, self.number_of_nodes_in_junction_tree))

    def _break_rings_in_cluster_graph(self, atom: int, new_nodes: list, ring_candidate: list) -> list:
        if self._ring_candidate_is_ring(ring_candidate):
            ring = ring_candidate
            new_nodes = self._intersect_node_to_break_a_ring(atom, new_nodes, ring)
            self._remove_remaining_bonds_from_ring(ring)
        return new_nodes

    def _remove_remaining_bonds_from_ring(self, ring_candidate: list) -> None:
        for current_cluster_index, next_cluster_index in itertools.combinations(ring_candidate, 2):
            if self._clusters_share_at_least_one_atom(self.cluster_graph, current_cluster_index, next_cluster_index):
                self._remove_shared_atoms(current_cluster_index, next_cluster_index)

    def _intersect_node_to_break_a_ring(self, atom: int, new_nodes: list, ring: list) -> list:
        number_of_atoms_in_molecule = self.number_of_atoms_in_molecule + len(ring)
        intersecting_node = self._create_node_with_dummy_atoms(number_of_atoms_in_molecule)
        for node_index in range(len(intersecting_node)):
            self._replace_atom_in_cluster(atom, intersecting_node, ring, node_index)
        new_nodes.append(intersecting_node)
        self.number_of_atoms_in_molecule = number_of_atoms_in_molecule
        return new_nodes

    def _replace_atom_in_cluster(self, atom: int, intersecting_node: list, ring: list, node_index: int) -> None:
        cluster_to_modify = ring[node_index]
        self.cluster_graph[cluster_to_modify].remove(atom)
        self.cluster_graph[cluster_to_modify].append(intersecting_node[node_index])

    def _remove_shared_atoms(self, current_cluster_index: int, next_cluster_index: int) -> None:
        shared_atoms = self._get_shared_atoms(current_cluster_index, next_cluster_index)
        for atom in shared_atoms:
            self.cluster_graph[current_cluster_index].remove(atom)
            self.cluster_graph[next_cluster_index].remove(atom)

    def _create_node_with_dummy_atoms(self, new_atoms_in_molecule: int) -> list:
        return [dummy_atom for dummy_atom in range(self.number_of_atoms_in_molecule, new_atoms_in_molecule)]

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
        return current_ring + list(all_common_elements_in_rings)

    @staticmethod
    def _rings_share_more_than_two_atoms(current_ring: list, next_ring: list) -> bool:
        return len(set(current_ring) & set(next_ring)) > 2

    @staticmethod
    def _clusters_share_at_least_one_atom(graph: list, current_cluster_index: int, next_cluster_index: int) -> bool:
        current_cluster, next_cluster = graph[current_cluster_index], graph[next_cluster_index]
        return current_cluster != next_cluster and len(set(current_cluster) & set(next_cluster)) >= 1

    @staticmethod
    def _ring_candidate_is_ring(ring_candidate: list) -> bool:
        return len(ring_candidate) > 2

    @staticmethod
    def _molecule_contains_only_two_rings(ring_index: int):
        return ring_index == 1
