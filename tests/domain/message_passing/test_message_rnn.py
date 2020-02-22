from unittest import TestCase

import numpy as np

from src.domain.message_passing.message_rnn import MessageRNN


class TestMessageRNN(TestCase):
    def setUp(self) -> None:
        self.message_rnn = MessageRNN()
        self.message_rnn.node_input = np.array([1, 2])
        self.message_rnn.edge_input = np.array([-2, 2])
        self.message_rnn.neighbors_input = np.array([-1, 1])

    def test_compose(self):
        # Given
        message_value_expected = np.array([0, 5])

        # When
        self.message_rnn.compose()

        # Then
        self.assertTrue(np.array_equal(message_value_expected, self.message_rnn.value))
