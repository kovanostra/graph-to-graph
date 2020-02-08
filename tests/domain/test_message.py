from unittest import TestCase

import numpy as np

from src.domain.message_rnn import MessageRNN


class TestMessage(TestCase):
    def setUp(self) -> None:
        self.message = MessageRNN()
        self.message.node_input = np.array([1, 2])
        self.message.edge_input = np.array([-2, 2])
        self.message.neighbors_input = np.array([-1, 1])

    def test_compose(self):
        # Given
        message_value_expected = np.array([0, 5])

        # When
        self.message.compose()

        # Then
        self.assertTrue(np.array_equal(message_value_expected, self.message.value))
