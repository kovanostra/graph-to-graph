from unittest import TestCase

import numpy as np
from rdkit import Chem

from src.domain.graph import Graph
from src.domain.tree_decomposer import TreeDecomposer


class TestTreeDecomposer(TestCase):
    def setUp(self) -> None:
        self.tree_decomposer = TreeDecomposer()

    def test_tree_decomposer_when_molecule_contains_one_bond_and_one_ring(self):
        # Given
        molecule_in_smiles = 'Cc1ccccc1'
        molecule = Chem.MolFromSmiles(molecule_in_smiles)
        junction_tree_expected = Graph(np.array([[0, 1],
                                                 [1, 0]]),
                                       Chem.GetAdjacencyMatrix(molecule),
                                       Chem.GetAdjacencyMatrix(molecule))

        # When
        junction_tree = self.tree_decomposer.decompose(molecule)

        # Then
        self.assertTrue(np.array_equal(junction_tree_expected.adjacency_matrix, junction_tree.adjacency_matrix))

    def test_tree_decomposer_when_molecule_contains_one_bond_and_two_rings_that_share_more_than_two_atoms(self):
        # Given
        molecule_in_smiles = 'Cc1cc2C=Cc(c1)c2'
        molecule = Chem.MolFromSmiles(molecule_in_smiles)
        junction_tree_expected = Graph(np.array([[0, 1],
                                                 [1, 0]]),
                                       Chem.GetAdjacencyMatrix(molecule),
                                       Chem.GetAdjacencyMatrix(molecule))

        # When
        junction_tree = self.tree_decomposer.decompose(molecule)

        # Then
        self.assertTrue(np.array_equal(junction_tree_expected.adjacency_matrix, junction_tree.adjacency_matrix))

    def test_tree_decomposer_when_molecule_contains_one_bond_and_two_rings_that_share_only_two_atoms(self):
        # Given
        molecule_in_smiles = 'CC1=CC2=CC=CC=C2C=C1'
        molecule = Chem.MolFromSmiles(molecule_in_smiles)
        junction_tree_expected = Graph(np.array([[0, 1, 0],
                                                 [1, 0, 1],
                                                 [0, 1, 0]]),
                                       Chem.GetAdjacencyMatrix(molecule),
                                       Chem.GetAdjacencyMatrix(molecule))

        # When
        junction_tree = self.tree_decomposer.decompose(molecule)

        # Then
        self.assertTrue(np.array_equal(junction_tree_expected.adjacency_matrix, junction_tree.adjacency_matrix))

    def test_tree_decomposer_when_molecule_contains_multiple_bonds_and_rings_that_share_more_than_two_atoms(self):
        # Given
        molecule_in_smiles = 'C12C3C(C(C1)CCC2)(C4CC3CC4)C(C)CC'
        molecule = Chem.MolFromSmiles(molecule_in_smiles)
        junction_tree_expected = Graph(np.array([[1, 1, 0, 1, 1, 0],
                                                 [1, 0, 1, 0, 0, 0],
                                                 [0, 1, 1, 0, 0, 0],
                                                 [0, 0, 1, 0, 1, 0],
                                                 [0, 0, 1, 1, 0, 1],
                                                 [0, 0, 0, 0, 1, 0]]),
                                       Chem.GetAdjacencyMatrix(molecule),
                                       Chem.GetAdjacencyMatrix(molecule))

        # When
        junction_tree = self.tree_decomposer.decompose(molecule)

        # Then
        print(junction_tree.adjacency_matrix)
        self.assertTrue(np.array_equal(junction_tree_expected.adjacency_matrix, junction_tree.adjacency_matrix))
