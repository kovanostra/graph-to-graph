from unittest import TestCase

from src.domain.gru_messenger import GRUMessenger
from tests.fixtures.matrices_and_vectors import BASE_W_MATRIX


class TestGRUMessenger(TestCase):
    def setUp(self) -> None:
        self.gru_messenger = GRUMessenger()
        self.gru_messenger.w_tree_update_gate_features = 0.1 * BASE_W_MATRIX
        self.gru_messenger.w_tree_forget_gate_features = 0.1 * BASE_W_MATRIX
        self.gru_messenger.w_tree_current_message_features = 0.1 * BASE_W_MATRIX
        self.gru_messenger.u_tree_update_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.u_tree_forget_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.u_tree_current_message = 0.1 * BASE_W_MATRIX
        self.gru_messenger.b_tree_update_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.b_tree_forget_gate = 0.1 * BASE_W_MATRIX
        self.gru_messenger.b_tree_current_message = 0.1 * BASE_W_MATRIX
