from unittest import TestCase

import numpy as np
from rdkit import Chem

from src.domain.graph import Graph
from src.domain.tree_decomposer import TreeDecomposer


class TestTreeDecomposer(TestCase):
    def setUp(self) -> None:
        self.tree_decomposer = TreeDecomposer()

    def test_tree_decomposer_when_one_ring_and_one_bond(self):
        # Given
        smiles = 'Cc1ccccc1'
        molecule = Chem.MolFromSmiles(smiles)
        junction_tree_expected = Graph(np.array([[0, 1],
                                                 [1, 0]]),
                                       Chem.GetAdjacencyMatrix(molecule),
                                       Chem.GetAdjacencyMatrix(molecule))

        # When
        junction_tree = self.tree_decomposer.decompose(molecule)

        # Then
        self.assertTrue(np.array_equal(junction_tree_expected.adjacency_matrix, junction_tree.adjacency_matrix))
